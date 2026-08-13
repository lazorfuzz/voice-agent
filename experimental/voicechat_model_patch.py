import asyncio
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
from collections import Counter, OrderedDict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import warnings

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*ffmpeg.*", module="pydub")

from checkpoint_utils import ModelComponentsBase, load_nemotron_voicechat
from data_types import InferenceOptions, InferenceResult, SeqInfo
from infer.utils import (
    _log_time,
    _time_block,
    format_tool_response_for_nemo,
    resolve_single_token_id,
    sample_text_token,
)
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.modules.ear_tts_vae_codec import CausalConv1dCache
from nemo.collections.speechlm2.parts.precision import fp32_precision
from runtime_setup import (
    DTYPE,
    FORCE_TURN_TAKING_THRESHOLD,
    MAX_TOOL_TOKENS,
    MODEL_OUTPUT_SAMPLE_RATE,
    RNNT_BARGE_IN_FRAMES,
    RNNT_BOU_MIN_FRAMES,
    RNNT_BOU_MIN_FRAMES_FIRST_TURN,
    RNNT_DENSITY_ALPHA,
    RNNT_DENSITY_LOW_MIN,
    RNNT_DENSITY_THRESHOLD,
    RNNT_DISPLAY_FALLBACK_CLEAR_FRAMES,
    RNNT_DISPLAY_GATE_FRAMES,
    RNNT_DISPLAY_MAX_SYMBOLS,
    RNNT_EOS_SILENCE_FRAMES,
    RNNT_FC_INTERRUPT_FRAMES,
    RNNT_MAX_AGENT_RESPONSE_FRAMES,
    RNNT_MAX_SYMBOLS,
    RNNT_NOISE_RESET_FRAMES,
    RNNT_PUNCT_BIAS_ENABLED,
    RNNT_PUNCT_BIAS_INCREMENT,
    RNNT_PUNCT_BIAS_MIN_SILENCE_FRAMES,
    RNNT_PUNCT_BIAS_TOKENS,
    RNNT_TTS_MIN_TOKENS,
    RNNT_TTS_RATIO_CAP,
)
from sequence_manager import SequenceManager
from torch.utils.dlpack import from_dlpack

logging.getLogger("nemo_logger").setLevel(logging.ERROR)

_re_available_tools_pattern = re.compile(r"<AVAILABLE_TOOLS>(.*?)</AVAILABLE_TOOLS>", re.DOTALL)
_re_tool_ack_messages_pattern = re.compile(r"<TOOL_ACK_MESSAGES>(.*?)</TOOL_ACK_MESSAGES>", re.DOTALL)


class GatheredCausalConv1dCache:
    """Gathers per-item cache state for the current batch; each item keeps its own causal state."""

    def __init__(self, batch_item_ids: List[int], per_item_caches: Dict[int, CausalConv1dCache],) -> None:
        self.batch_item_ids = batch_item_ids
        self.per_item_caches = per_item_caches

    def update(
        self,
        states: torch.Tensor,
        layer_id: Union[int, str],
        padding: int,
        padding_value: int = 0,
        flush: bool = False,
    ) -> torch.Tensor:
        device = states.device
        dtype = states.dtype
        b, c, t = states.size()

        for bid in self.batch_item_ids:
            if bid not in self.per_item_caches:
                self.per_item_caches[bid] = CausalConv1dCache()

        padding_list = []
        for i in range(b):
            bid = self.batch_item_ids[i]
            cache = self.per_item_caches[bid]
            if layer_id in cache.cache:
                padding_list.append(cache.cache[layer_id])
            else:
                padding_list.append(torch.zeros((1, c, padding), dtype=dtype, device=device) + padding_value)
        padding_tensor = torch.cat(padding_list, dim=0)
        padded_states = torch.cat([padding_tensor, states], dim=2)
        tail = padded_states[:, :, -padding:]

        for i in range(b):
            bid = self.batch_item_ids[i]
            self.per_item_caches[bid].cache[layer_id] = tail[i : i + 1].detach().clone()

        if flush:
            for bid in self.batch_item_ids:
                cache = self.per_item_caches[bid]
                if layer_id in cache.cache:
                    del cache.cache[layer_id]

        return padded_states


# ---------------------------------------------------------------------------
# RNNT hidden-state helpers for batched EOU detection
# ---------------------------------------------------------------------------


def _rnnt_slice_hidden(hidden, idx: int):
    """Extract hidden state for one sequence at position idx from a batched hidden state."""
    if hidden is None:
        return None
    if isinstance(hidden, (tuple, list)):
        return type(hidden)(h[:, idx : idx + 1, :] for h in hidden)
    return hidden[:, idx : idx + 1, :]


def _rnnt_stack_hidden(hiddens: list):
    """Stack list of per-sequence (num_layers, 1, H) hidden states into (num_layers, B, H)."""
    if all(h is None for h in hiddens):
        return None
    if isinstance(hiddens[0], (tuple, list)):
        stacked = [torch.cat([h[i] for h in hiddens], dim=1) for i in range(len(hiddens[0]))]
        return type(hiddens[0])(stacked)
    return torch.cat(hiddens, dim=1)


def _rnnt_masked_update_hidden(is_blank: torch.Tensor, old_hidden, new_hidden):
    """For non-blank sequences, replace old_hidden with new_hidden; blank sequences keep old."""
    if old_hidden is None:
        return new_hidden
    B = is_blank.shape[0]
    mask = is_blank.view(1, B, 1)  # broadcasts over (num_layers, B, hidden_size)
    if isinstance(old_hidden, (tuple, list)):
        updated = [torch.where(mask, o, n) for o, n in zip(old_hidden, new_hidden)]
        return type(old_hidden)(updated)
    return torch.where(mask, old_hidden, new_hidden)


class TritonPythonModel:
    @_log_time("Model initialization", debug_only=False)
    def initialize(self, args):
        logger = pb_utils.Logger
        logger.log_info(f"Initializing model")

        self._event_loop = asyncio.new_event_loop()
        self._event_loop_thread = threading.Thread(
            target=self._event_loop.run_forever, daemon=True, name="s2s_event_loop"
        )
        self._event_loop_thread.start()

        logger = pb_utils.Logger

        self.options = InferenceOptions()

        self.parameters = json.loads(args["model_config"])["parameters"]
        self.default_system_prompt = (
            self.parameters["system_prompt"]["string_value"] if "system_prompt" in self.parameters else ""
        )
        self.model_repo_path = os.path.join(args["model_repository"], args["model_version"])

        self.model: ModelComponentsBase = load_nemotron_voicechat(self.model_repo_path)
        self._seq_mgr = SequenceManager(
            device=self.model.device,
            num_audio_quantizers=self.model.model_cfg.model.speech_generation.model.tts_config.num_quantizers,
        )
        self.pad_id = get_pad_id(self.model.tokenizer)
        self._tts_silence_codes = self._compute_tts_silence_codes()
        self.user_bos_id = resolve_single_token_id(self.model.tokenizer, "^")
        self._prompt_embed_cache: OrderedDict[str, Tuple[list, torch.Tensor, int]] = OrderedDict()
        self._prompt_embed_cache_max = 8

        # Sampling parameters
        self.top_p = float(os.environ.get("LLM_TOP_P", "1.0"))
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
        self.repetition_penalty = float(os.environ.get("LLM_REPETITION_PENALTY", "1.0"))
        self.special_token_ids: set = {
            self.model.tokenizer.pad_id,
            self.model.tokenizer.bos_id,
            self.model.tokenizer.eos_id,
        }
        self._rnnt_vocab: List[str] = []
        self._rnnt_punct_ids: List[int] = []
        self._rnnt_punct_ids_set: set = set()
        self._init_rnnt_display_vocab()
        logger.log_info(
            f"Sampling params — top_p={self.top_p}, temperature={self.temperature}, "
            f"repetition_penalty={self.repetition_penalty}"
        )

        # TTS PAD-silence fix: substitute codec silence tokens for PAD frames when the
        # agent text channel is idle (no open turn) or the in-turn PAD tail exceeds the
        # rendering budget. Mirrors the EOS-side substitution already in vllm_tts_step.
        self._tts_force_silence_on_pad = os.environ.get(
            "S2S_INFERENCE_FORCE_SPEECH_SILENCE_ON_PAD", "true"
        ).lower() not in ("0", "false")
        self._tts_pad_tail_ratio = float(os.environ.get("S2S_TTS_PAD_TAIL_RATIO", "3"))
        # Kronik patch: conditional suppression of spontaneous agent turn-starts (the
        # silence-monologue). Ported from the nemotron-labs-voicechat branch's
        # inference_bos_boost, made conditional on recent user activity.
        self._bos_suppress = float(os.environ.get("S2S_INFERENCE_BOS_BOOST", "0"))
        self._bos_allow_steps = int(float(os.environ.get("S2S_BOS_ALLOW_SEC", "6.0")) / 0.08)
        logger.log_info(f"KRONIK PATCH ACTIVE: bos_suppress={self._bos_suppress} "
                        f"allow_steps={self._bos_allow_steps} bos_id={self.model.tokenizer.bos_id}")
        logger.log_verbose(
            f"TTS PAD-silence — force_silence_on_pad={self._tts_force_silence_on_pad}, "
            f"pad_tail_ratio={self._tts_pad_tail_ratio}"
        )

        # Function call tokens
        self.sotc_id = self._resolve_single_token_id("<SPECIAL_20>")  # start of function calling
        self.eotc_id = self._resolve_single_token_id("<SPECIAL_21>")  # end of function calling

        # Encoded-audio hidden dim; populated on first perception() call for dummy audio in fast extraction
        self._encoded_audio_dim: Optional[int] = None
        # Silence chunk returned to keep the audio stream active during fast-path skips;
        self._silence_audio_chunk: Optional[torch.Tensor] = None
        # Tracks active tool-token extraction threads keyed by seq_id
        self._extraction_threads: Dict[int, threading.Thread] = {}
        # Tracks active tool-response injection threads keyed by seq_id
        self._injection_threads: Dict[int, threading.Thread] = {}
        # Serialises run_until_complete calls on self.model.tts.event_loop from multiple threads
        self._tts_loop_lock = threading.Lock()

        logger.log_info("Model initialization done")

    def _run_async(self, coro):
        """Schedule a coroutine on the background event loop; returns a concurrent.futures.Future."""
        return asyncio.run_coroutine_threadsafe(coro, self._event_loop)

    def _run_coro(self, coro):
        """Submit a coroutine to the background event loop and block until it completes."""
        return asyncio.run_coroutine_threadsafe(coro, self._event_loop).result()

    @staticmethod
    async def _gather(*coros):
        """Await multiple coroutines concurrently.  Use instead of asyncio.gather() when
        the result must be passed to run_coroutine_threadsafe (which requires a coroutine,
        not the Future that asyncio.gather() returns directly)."""
        return await asyncio.gather(*coros)

    def _init_rnnt_display_vocab(self) -> None:
        """Resolve RNNT vocabulary and punctuation IDs for display-only transcript biasing."""
        self._rnnt_vocab = []
        self._rnnt_punct_ids = []
        self._rnnt_punct_ids_set = set()
        self._rnnt_punct_bias_increments = {}

        if self.model.rnnt_decoder is None:
            return

        joint = getattr(getattr(self.model.rnnt_decoder, "decoding", None), "joint", None)
        vocab = getattr(joint, "vocabulary", None)
        if vocab is None:
            vocab = getattr(self.model.rnnt_decoder, "vocabulary", None)
        if vocab is None:
            pb_utils.Logger.log_warning("[RNNT display] No RNNT vocabulary found; transcript display disabled")
            return

        self._rnnt_vocab = list(vocab)
        if not RNNT_PUNCT_BIAS_ENABLED:
            return

        punct_bias_increments = {".": 1.5, "?": 1.5, ",": 1.0, "!": 1.0}
        for punct in RNNT_PUNCT_BIAS_TOKENS:
            for variant in (punct, "▁" + punct):
                if variant in self._rnnt_vocab:
                    punct_id = self._rnnt_vocab.index(variant)
                    if punct_id not in self._rnnt_punct_ids_set:
                        self._rnnt_punct_ids.append(punct_id)
                        self._rnnt_punct_ids_set.add(punct_id)
                        self._rnnt_punct_bias_increments[punct_id] = punct_bias_increments.get(
                            punct, RNNT_PUNCT_BIAS_INCREMENT
                        )

        pb_utils.Logger.log_verbose(
            f"[RNNT display] vocab={len(self._rnnt_vocab)} punct_ids={self._rnnt_punct_ids} "
            f"bias_enabled={RNNT_PUNCT_BIAS_ENABLED} increments={self._rnnt_punct_bias_increments} "
            f"min_silence={RNNT_PUNCT_BIAS_MIN_SILENCE_FRAMES}"
        )

    def _get_bos_embedding(self) -> torch.Tensor:
        """
        Gets the embedding for the beginning-of-sequence token.
        """
        text_bos = torch.full((1,), fill_value=self.pad_id, device=self.model.device)
        input_embeds = self.model.embed_tokens(text_bos)
        return input_embeds.to(dtype=DTYPE)

    def _resolve_single_token_id(self, token_str: str) -> int:
        token_ids = self.model.tokenizer.text_to_ids(token_str)
        if not token_ids:
            raise ValueError(f"Tokenizer did not return an id for token: {token_str}")
        if len(token_ids) != 1:
            raise ValueError(f"Expected a single id for token {token_str}, got {token_ids}")
        return int(token_ids[0])

    def _prepare_system_prompt_embeddings(self, system_prompt: str,) -> Tuple[List[int], Optional[torch.Tensor], int]:
        """
        Prepare system prompt embeddings consistent with offline_inference.

        In offline_inference, prompt embeddings are structured as:
        - Position 0: prompt_token_emb + bos_emb + pad_func
        - Position t > 0: prompt_token_emb + pad_emb + pad_func

        Args:
            system_prompt: The system prompt text

        Returns:
            Tuple of (prompt_embedded [1, prompt_len, H], prompt_length)
            Returns (None, 0) if system_prompt is empty
        """

        if not system_prompt or not system_prompt.strip():
            return [], None, 0

        if system_prompt in self._prompt_embed_cache:
            self._prompt_embed_cache.move_to_end(system_prompt)
            token_ids, cached_embed, prompt_len = self._prompt_embed_cache[system_prompt]
            return list(token_ids), cached_embed.clone(), prompt_len

        logging.info(f"\n Preparing system prompt: {system_prompt[:100]}...")

        # Step 1: Tokenize the prompt
        # Format: [bos] + text_tokens + [eos] (consistent with collate_system_prompt)
        single_prompt_token_ids = (
            [self.model.tokenizer.bos_id]
            + self.model.tokenizer.text_to_ids(system_prompt)
            + [self.model.tokenizer.eos_id]
        )
        # Match NeMo's default two-copy system-prompt prefill.
        prompt_token_ids = single_prompt_token_ids * 2
        prompt_tokens = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.model.device).unsqueeze(
            0
        )  # [1, prompt_len]
        prompt_len = prompt_tokens.shape[1]

        logging.info(f"   Prompt length: {prompt_len} tokens")

        # Step 2: Embed the prompt tokens (this acts as the "audio channel" for prompt positions)
        prompt_embedded = self.model.embed_tokens(prompt_tokens)  # [1, prompt_len, H]
        prompt_embedded = prompt_embedded.to(dtype=DTYPE)

        # Step 3: Add pad embeddings for text (for positions t > 0)
        # In offline_inference, prompt positions use gen_text[:, t-1] = pad_id
        pad_id = self.model.tokenizer.pad_id
        pad_token = torch.full((1,), fill_value=pad_id, device=self.model.device, dtype=torch.long)
        pad_emb = self.model.embed_tokens(pad_token).to(dtype=DTYPE)  # [1, H]

        # For positions t > 0, add pad embeddings (simulating gen_text[:, t-1] = pad_id)
        if prompt_len > 1:
            prompt_embedded[:, 1:, :] += pad_emb

        # Step 4: For position 0, add BOS embeddings
        bos_emb = self._get_bos_embedding()  # [1, H]
        prompt_embedded[:, 0, :] += bos_emb.squeeze(0)

        # Step 5: Add function channel (pad) for all prompt positions
        duplex_function_channel_weight = self.model.model_cfg.model.stt.model.get(
            "duplex_function_channel_weight", 1.0
        )
        prompt_embedded += pad_emb * duplex_function_channel_weight

        logging.info(f"   System prompt embeddings prepared: shape {prompt_embedded.shape}")

        self._prompt_embed_cache[system_prompt] = (list(prompt_token_ids), prompt_embedded.clone(), prompt_len)
        if len(self._prompt_embed_cache) > self._prompt_embed_cache_max:
            self._prompt_embed_cache.popitem(last=False)
        return prompt_token_ids, prompt_embedded, prompt_len

    def _compute_tts_silence_codes(self) -> torch.Tensor:
        """Compute codec silence tokens by encoding a zero waveform, matching NeMo's get_codec_silence_frame()."""
        wav_to_token_ratio = self.model.codec.config.wav_to_token_ratio
        num_samples = wav_to_token_ratio * 10  # 10 frames of silence
        silence_audio = torch.zeros(1, 1, num_samples, device=self.model.device)
        silence_len = torch.tensor([num_samples], device=self.model.device)
        with torch.no_grad():
            sil_codes, _ = self.model.codec.encode(silence_audio, silence_len)
            sil_codes = sil_codes[0]  # [T, num_quantizers]
        if len(sil_codes) == 0:
            raise RuntimeError("Codec returned empty silence codes during initialization")
        combos = [tuple(row.tolist()) for row in sil_codes]
        most_common = Counter(combos).most_common(1)[0][0]
        return torch.tensor(most_common, dtype=torch.long, device=self.model.device)

    @_log_time("Perception time", cuda_sync=True)
    def perception(
        self, audio_to_encode: torch.Tensor, buffer_len_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the audio signal and return the source encoded and ASR embedding.
        Args:
            audio_to_encode: The audio signal to encode. Shape: (B, T)
            buffer_len_tensor: The length of the audio signal. Shape: (B,)
        Returns:
            The source encoded and ASR embedding.
            Shape: (B, T, D)
        """
        logger = pb_utils.Logger

        logger.log_verbose("***Perception****")

        encoded_audio, _, asr_emb = self.model.perception(
            input_signal=audio_to_encode, input_signal_length=buffer_len_tensor, return_encoder_emb=True,
        )

        encoded_audio = encoded_audio.to(DTYPE)  # (B, T, D)

        if self._encoded_audio_dim is None:
            self._encoded_audio_dim = encoded_audio.shape[2]

        return encoded_audio, asr_emb

    async def _get_vllm_llm_next_token(self, seq_id, prompt_token_ids: List[int], input_embeds: torch.Tensor):
        """
        seq_id: int
        input_embeds: (1, D)
        """
        logger = pb_utils.Logger
        logger.log_verbose("***VLLM LLM next token***")

        seq_state = self._seq_mgr[seq_id]

        input_dtype = self.model.llm_custom_input_specs[0].dtype
        custom_inputs = {"combined_embeds": input_embeds[0].to(dtype=getattr(torch, input_dtype)).cpu()}

        if seq_state.vllm_llm_generator is None:
            logger.log_verbose(f"Initializing vLLM LLM generator")
            inputs = {"type": "tokens", "prompt_token_ids": prompt_token_ids, "custom_inputs": custom_inputs}
            seq_state.vllm_llm_generator = self.model.llm.generate(
                inputs, self.model.llm_sampling_params, request_id=str(seq_id),
            )
            logger.log_verbose(f"gen created *******")

        if seq_state.vllm_llm_start:
            logger.log_verbose(f"custom_inputs: {custom_inputs}")
            await self.model.llm.append_request(
                request_id=str(seq_id), custom_inputs=custom_inputs,
            )
            logger.log_verbose(f"appended request *******")
        logger.log_verbose("get next token *******")
        output = await anext(seq_state.vllm_llm_generator)
        seq_state.vllm_llm_start = True
        logger.log_verbose(f"output: {output}")
        tokens = output.outputs[0].token_ids

        return {
            "token": tokens[-1],
            "custom_outputs": output.outputs[0].custom_outputs,
        }

    async def _vllm_llm_cleanup(self, seq_id):
        logger = pb_utils.Logger
        logger.log_verbose("***VLLM LLM Cleanup***")
        await self.model.llm.abort(str(seq_id))
        seq_state = self._seq_mgr[seq_id]
        seq_state.vllm_llm_generator = None

    async def _vllm_llm_step(self, seq_id, input_embeds):
        """
        input_embeds: (T, D)
        seq_id: int
        """
        logger = pb_utils.Logger
        logger.log_verbose("***VLLM LLM Step***")
        logger.log_verbose(f"input_embeds: {input_embeds.shape}************@@@@@@@@@@@@@@@")

        seq_len, _ = input_embeds.shape
        tokens = []
        function_tokens = []
        seq_state = self._seq_mgr[seq_id]

        for i in range(seq_len):
            single_embed = input_embeds[i : i + 1, :]  # (1, D)
            vllm_result = await self._get_vllm_llm_next_token(seq_id, [1], [single_embed])
            text_logits_last = (
                vllm_result["custom_outputs"]["text_logits"].to(self.model.device)[-1].unsqueeze(0)
            )  # (1, V)
            # Kronik patch: outside the post-user-speech / function-call window, bias
            # the agent's turn-start (BOS) logit down so the model cannot begin
            # unprompted monologue turns during silence. Legit answers are unaffected:
            # they start within the allow window after user speech or a tool response.
            if self._bos_suppress != 0.0:
                if getattr(seq_state, "rnnt_user_speaking", False):
                    seq_state.fn_call["kronik_last_voice"] = seq_state.decoder_global_step
                _last_voice = seq_state.fn_call.get("kronik_last_voice", 0)
                _in_fc = seq_state.fn_call.get("state", "idle") != "idle"
                _armed = _last_voice > 0 or seq_state.fn_call.get("kronik_greeted", False)
                if (_armed and not _in_fc
                        and (seq_state.decoder_global_step - _last_voice) > self._bos_allow_steps):
                    _pre = int(text_logits_last.argmax(dim=-1)[0].item())
                    if _pre == self.model.tokenizer.bos_id:
                        logger.log_info(
                            f"KRONIK: suppressing spontaneous BOS at step "
                            f"{seq_state.decoder_global_step} (last_voice={_last_voice})")
                    text_logits_last[0, self.model.tokenizer.bos_id] += self._bos_suppress
            token = sample_text_token(
                logits=text_logits_last,
                generated_tokens=seq_state.generated_text_tokens,
                current_step=seq_state.decoder_global_step,
                top_p=self.top_p,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                special_token_ids=self.special_token_ids,
            ).squeeze(0)
            if int(token.item()) == self.model.tokenizer.bos_id:
                seq_state.fn_call["kronik_greeted"] = True   # first agent turn done: arm suppression
            tokens.append(token)
            function_tokens.append(vllm_result["custom_outputs"]["function_tokens"])

        return {
            "token": tokens[-1],
            "function_token": function_tokens[-1],
        }

    @_log_time("LLM Forward Pass time")
    def _llm_step(self, encoded_audio: torch.Tensor, seq_ids_batched: List[int], step_offset: int = 0):
        """
        encoded_audio: (B, T, D)
        seq_ids_batched: List[int]
        """
        logger = pb_utils.Logger

        current_input_embeds_batched = []

        for batch_idx, seq_id in enumerate(seq_ids_batched):
            seq_state = self._seq_mgr[seq_id]

            if seq_state.decoder_global_step == 0:
                pad_id = self.model.tokenizer.pad_id
                pad_token = torch.full((1,), fill_value=pad_id, device=self.model.device, dtype=torch.long)
                last_function_emb = self.model.embed_tokens(pad_token).to(DTYPE)  # (1, D)
                if not seq_state.system_prompt:
                    last_embeds = self._get_bos_embedding()  # (1, D)
                else:
                    last_embeds = self.model.embed_tokens(pad_token).to(dtype=DTYPE)

            else:
                last_idx = max(0, seq_state.decoder_global_step - 1)
                last_text_token = seq_state.generated_text_tokens[:, last_idx]
                last_embeds = self.model.embed_tokens(last_text_token).to(DTYPE)  # (1, D)

                last_function_token = seq_state.generated_function_tokens[:, last_idx]
                last_function_emb = self.model.embed_tokens(last_function_token).to(DTYPE)  # (1, D)

            duplex_user_channel_weight = self.model.model_cfg.model.stt.model.get("duplex_user_channel_weight", 1.0)
            duplex_text_channel_weight = self.model.model_cfg.model.stt.model.get("duplex_text_channel_weight", 1.0)
            duplex_function_channel_weight = self.model.model_cfg.model.stt.model.get(
                "duplex_function_channel_weight", 1.0
            )

            max_audio_idx = self.options.max_chunks_for_inference - 1
            if seq_state.audio_chunk_idx >= max_audio_idx:
                audio_idx = max_audio_idx - (self.options.num_chunks_per_inference - 1) + step_offset
            else:
                audio_idx = seq_state.audio_chunk_idx
            current_input_embeds = (
                encoded_audio[batch_idx : batch_idx + 1, audio_idx : audio_idx + 1, :] * duplex_user_channel_weight
            )  # (1, 1, D)
            current_input_embeds += last_embeds.unsqueeze(1) * duplex_text_channel_weight  # (1, 1, D)
            current_input_embeds += last_function_emb.unsqueeze(1) * duplex_function_channel_weight  # (1, 1, D)

            seq_state.input_embeddings = current_input_embeds  # (B, 1, D)

            current_input_embeds_batched.append(seq_state.input_embeddings)

        current_input_embeds_batched = torch.cat(current_input_embeds_batched, dim=0)  # (B, 1, D)

        logger.log_verbose("LLM/Decoder Forward Pass")
        logger.log_verbose(f"current_input_embeds_batched: {current_input_embeds_batched.shape}")

        # Schedule all vLLM steps concurrently on the background event loop
        result_batched = self._run_coro(
            self._gather(
                *[
                    self._vllm_llm_step(seq_id, current_input_embeds_batched[batch_idx])
                    for batch_idx, seq_id in enumerate(seq_ids_batched)
                ]
            )
        )
        for batch_idx, seq_id in enumerate(seq_ids_batched):
            seq_state = self._seq_mgr[seq_id]
            buffer_capacity = seq_state.generated_text_tokens.shape[1]
            if seq_state.decoder_global_step >= buffer_capacity:
                if not seq_state.steps_exhausted:
                    logger.log_error(
                        f"[seq {seq_id}] _llm_step: buffer overflow at step {seq_state.decoder_global_step}, marking exhausted"
                    )
                    seq_state.steps_exhausted = True
                continue
            if seq_state.fn_call["state"] == "idle":
                seq_state.generated_text_tokens[:, seq_state.decoder_global_step] = result_batched[batch_idx]["token"]
            else:
                seq_state.generated_text_tokens[:, seq_state.decoder_global_step] = self.pad_id
            seq_state.generated_function_tokens[:, seq_state.decoder_global_step] = result_batched[batch_idx][
                "function_token"
            ]

            if seq_state.fn_call["ack_tokens"] and seq_state.fn_call["state"] == "speaking_ack":
                # Inject ack text token into text channel so the model speaks the ack message
                ack_token = seq_state.fn_call["ack_tokens"].popleft()
                seq_state.generated_text_tokens[:, seq_state.decoder_global_step] = ack_token
                if len(seq_state.fn_call["ack_tokens"]) == 0:
                    if seq_state.fn_call["response_tokens"]:
                        seq_state.fn_call["state"] = "process_response"
                        seq_state.fn_call["frames_in_state"] = 0
                        logger.log_info(
                            f"Function call state transition: speaking_ack -> process_response (seq_id: {seq_id}) (step: {seq_state.decoder_global_step})"
                        )
                    else:
                        seq_state.fn_call["state"] = "waiting_for_response"
                        seq_state.fn_call["frames_in_state"] = 0
                        logger.log_info(
                            f"Function call state transition: speaking_ack -> waiting_for_response (seq_id: {seq_id}) (step: {seq_state.decoder_global_step})"
                        )
            elif seq_state.fn_call["state"] == "process_response" and seq_state.fn_call["response_tokens"]:
                response_token = seq_state.fn_call["response_tokens"].popleft()
                seq_state.generated_function_tokens[:, seq_state.decoder_global_step] = response_token
                if len(seq_state.fn_call["response_tokens"]) == 0:
                    logger.log_info(
                        f"Function call state transition: process_response -> idle (seq_id: {seq_id}) (step: {seq_state.decoder_global_step})"
                    )
                    self._reset_fn_call_state(seq_state.fn_call, seq_state)

    @_log_time("TTS Forward Pass time")
    def vllm_tts_step(self, seq_ids_batched: List[int]):
        # Generate Audio Codes. In NeMo, this is optional. For Riva, this is mandatory always.
        logger = pb_utils.Logger

        torch_type = getattr(torch, "float32")

        tts_futures = []
        for seq_id in seq_ids_batched:
            seq_state = self._seq_mgr[seq_id]
            buffer_capacity = seq_state.generated_text_tokens.shape[1]
            if seq_state.decoder_global_step >= buffer_capacity or seq_state.steps_exhausted:
                logger.log_info(
                    f"[seq {seq_id}] vllm_tts_step: skipping exhausted sequence at step {seq_state.decoder_global_step}"
                )
                continue

            # create stream if required
            self.model.tts.create_stream(seq_id)

            done, output = self.model.tts.stream_get_output(seq_id)
            if output is None:
                logger.log_error("Error in vLLM TTS stream_get_output")
                return

            acoustic_tokens = output.outputs[0].custom_outputs["acoustic_tokens"]  # T x 31
            step_acoustic_tokens = acoustic_tokens[-1:]  # 1 x 31

            current_subword_id = torch.Tensor(
                (seq_state.generated_text_tokens[0, self._seq_mgr[seq_id].decoder_global_step])
            ).unsqueeze(0)
            _csid = int(current_subword_id.item())
            _bos_id = self.model.tokenizer.bos_id
            _eos_id = self.model.tokenizer.eos_id
            _pad_id = self.pad_id

            # (A) EOS → silence (existing): force clean silence when turn-taking injects EOS
            if _csid == _eos_id:
                logger.log_info("Found EOS token in text tokens. Inserting silence into codec input")
                step_acoustic_tokens = self._tts_silence_codes.unsqueeze(0)

            # (B) PAD → silence when agent is idle or in-turn PAD tail exceeds render budget.
            # Prevents codec hallucination during post-FC silent gaps and stuck-open turns.
            speaking_ack = seq_state.fn_call["state"] == "speaking_ack"
            if self._tts_force_silence_on_pad and _csid == _pad_id and not speaking_ack:
                _idle = seq_state.tts_agent_idle
                _c = seq_state.tts_in_turn_content
                _p = seq_state.tts_in_turn_pads
                _tail_done = self._tts_pad_tail_ratio > 0 and not _idle and _p > self._tts_pad_tail_ratio * _c
                if _idle or _tail_done:
                    step_acoustic_tokens = self._tts_silence_codes.unsqueeze(0)
                    if _tail_done:
                        logger.log_verbose(
                            f"[seq {seq_id}] TTS PAD-tail silence: "
                            f"pads={_p} > ratio={self._tts_pad_tail_ratio} * content={_c}"
                        )

            self._seq_mgr[seq_id].generated_audio_tokens[
                :, self._seq_mgr[seq_id].decoder_global_step
            ] = step_acoustic_tokens

            # (C) Update agent_idle state machine and in-turn counters.
            # Substitution (A/B) reads state from the *previous* frame; update after storing.
            if _csid == _bos_id:
                seq_state.tts_agent_idle = False
                seq_state.tts_in_turn_content = 0
                seq_state.tts_in_turn_pads = 0
            elif _csid == _eos_id:
                seq_state.tts_agent_idle = True
                seq_state.tts_in_turn_content = 0
                seq_state.tts_in_turn_pads = 0
            elif _csid == _pad_id:
                if not seq_state.tts_agent_idle:
                    seq_state.tts_in_turn_pads += 1
            else:  # content token
                seq_state.tts_in_turn_content += 1
                seq_state.tts_in_turn_pads = 0

            bos_mask = torch.full_like(current_subword_id, 1e-20, dtype=torch_type)
            new_custom_inputs = {
                "acoustic_tokens": step_acoustic_tokens.cpu(),
                "text_tokens": current_subword_id.cpu(),
                "text_mask": torch.ones_like(current_subword_id, dtype=torch_type).cpu(),
                "bos_mask": bos_mask.cpu(),
            }
            new_custom_inputs["audio_prompt_latent"] = self.model.tts._audio_prompt_latent
            tts_futures.append(self.model.tts.stream_submit_request(seq_id, new_custom_inputs))
        if tts_futures:
            with self._tts_loop_lock:
                self.model.tts.event_loop.run_until_complete(asyncio.gather(*tts_futures))

    def _new_rnnt_display_state(self) -> dict:
        return {
            "pred_out": None,
            "pred_hidden": None,
            "blank_count": 0,
            "nonblank_total": 0,
            "speech_confirmed": False,
            "y_sequence": [],
            "_punct_word_acc": [],
            "_punct_bias_val": 0.0,
        }

    def _reset_rnnt_display_buffer(self, seq_state, reset_predictor: bool = False) -> None:
        state = seq_state.rnnt_display_state or self._new_rnnt_display_state()
        state["blank_count"] = 0
        state["nonblank_total"] = 0
        state["speech_confirmed"] = False
        state["y_sequence"] = []
        state["_punct_word_acc"] = []
        state["_punct_bias_val"] = 0.0
        if reset_predictor:
            state["pred_out"] = None
            state["pred_hidden"] = None
        seq_state.rnnt_display_state = state
        seq_state.rnnt_display_emitted_len = 0
        seq_state.rnnt_display_turn_open = False

    def _rnnt_display_decode_text(self, y_sequence: list) -> str:
        if not self._rnnt_vocab or not y_sequence:
            return ""
        return (
            "".join(self._rnnt_vocab[t] for t in y_sequence if isinstance(t, int) and 0 <= t < len(self._rnnt_vocab))
            .replace("▁", " ")
            .strip()
        )

    def _append_rnnt_display_output(
        self, seq_state, agent_bos_fired: bool, force_publish: bool = False, close_turn: bool = False
    ) -> None:
        state = seq_state.rnnt_display_state
        if state is None:
            return

        full_text = self._rnnt_display_decode_text(state.get("y_sequence", []))
        has_word = any(ch.isalpha() for ch in full_text)
        # An agent BOS closes the current user turn. Publish any accumulated
        # RNNT text before resetting it, even when a short utterance did not
        # reach the normal display gate.
        if (state.get("speech_confirmed", False) or force_publish or agent_bos_fired) and full_text and has_word:
            if len(full_text) < seq_state.rnnt_display_emitted_len:
                seq_state.rnnt_display_emitted_len = 0
            delta = full_text[seq_state.rnnt_display_emitted_len :]
            if delta:
                if not seq_state.rnnt_display_turn_open:
                    seq_state.rnnt_display_pending_text += "<s>"
                    seq_state.rnnt_display_turn_open = True
                    pb_utils.Logger.log_verbose(f"[RNNT display] [seq {seq_state.seq_id}] opened user turn")
                seq_state.rnnt_display_pending_text += delta
                seq_state.rnnt_display_emitted_len = len(full_text)
                pb_utils.Logger.log_verbose(
                    f"[RNNT display] [seq {seq_state.seq_id}] queued delta={delta!r} "
                    f"full_text={full_text!r} pending={seq_state.rnnt_display_pending_text!r}"
                )

        if close_turn and seq_state.rnnt_display_turn_open:
            seq_state.rnnt_display_pending_text += "</s>"
            pb_utils.Logger.log_verbose(
                f"[RNNT display] [seq {seq_state.seq_id}] force-published turn on END "
                f"pending={seq_state.rnnt_display_pending_text!r}"
            )

        if agent_bos_fired:
            if seq_state.rnnt_display_turn_open:
                seq_state.rnnt_display_pending_text += "</s>"
                pb_utils.Logger.log_verbose(
                    f"[RNNT display] [seq {seq_state.seq_id}] queued </s> on agent BOS "
                    f"pending={seq_state.rnnt_display_pending_text!r}"
                )
            else:
                pb_utils.Logger.log_verbose(
                    f"[RNNT display] [seq {seq_state.seq_id}] agent BOS with no open display turn"
                )
            self._reset_rnnt_display_buffer(seq_state)
            return

        if state.get("blank_count", 0) >= RNNT_DISPLAY_FALLBACK_CLEAR_FRAMES and state.get("nonblank_total", 0) == 0:
            pb_utils.Logger.log_verbose(
                f"[RNNT display] [seq {seq_state.seq_id}] fallback reset after "
                f"{state.get('blank_count', 0)} blank frames with no speech"
            )
            self._reset_rnnt_display_buffer(seq_state, reset_predictor=True)

    def _rnnt_display_decode_frame(self, asr_emb: torch.Tensor, batch_idx: int, seq_id: int, frame_idx: int) -> None:
        if asr_emb is None or self.model.rnnt_decoder is None or not self._rnnt_vocab:
            return

        seq_state = self._seq_mgr[seq_id]
        state = seq_state.rnnt_display_state or self._new_rnnt_display_state()

        decoder = self.model.rnnt_decoder.decoding.decoder
        joint = self.model.rnnt_decoder.decoding.joint
        blank_id = getattr(self.model.rnnt_decoder, "blank_id", joint._num_classes - 1)
        rnnt_dtype = next(joint.parameters()).dtype

        with torch.inference_mode():
            f = asr_emb[batch_idx : batch_idx + 1, frame_idx : frame_idx + 1, :].to(rnnt_dtype)
            pred_out = state["pred_out"]
            pred_hidden = state["pred_hidden"]
            if pred_out is None:
                pred_out, pred_hidden = decoder.predict(y=None, state=None, add_sos=True, batch_size=1)
            if pred_out.dim() == 3 and pred_out.shape[1] > 1:
                pred_out = pred_out[:, -1:, :]

            logits = joint.joint(f, pred_out)
            scores = logits.squeeze(1).squeeze(1)
            tokens = scores.argmax(-1)
            is_blank = int(tokens[0].item()) == blank_id

            emitted = []
            cur_pred_out, cur_pred_hidden = pred_out, pred_hidden

            if is_blank and self._rnnt_punct_ids:
                punct_bias = float(state.get("_punct_bias_val", 0.0))
                if punct_bias > 0.0:
                    biased_scores = scores.clone()
                    for punct_id in self._rnnt_punct_ids:
                        biased_scores[0, punct_id] += punct_bias * self._rnnt_punct_bias_increments[punct_id]
                    punct_token = int(biased_scores.argmax(-1)[0].item())
                    if punct_token in self._rnnt_punct_ids_set:
                        emitted.append(punct_token)
                        y_punct = torch.tensor([[punct_token]], dtype=torch.long, device=asr_emb.device)
                        try:
                            cur_pred_out, cur_pred_hidden = decoder.predict(
                                y=y_punct, state=cur_pred_hidden, add_sos=False, batch_size=1
                            )
                        except Exception as exc:
                            pb_utils.Logger.log_warning(f"[RNNT display] punct predictor step failed: {exc}")

            loop_tokens = tokens
            loop_is_blank = tokens == blank_id
            symbols = 0
            while not loop_is_blank.all().item() and symbols < RNNT_DISPLAY_MAX_SYMBOLS:
                token = int(loop_tokens[0].item())
                emitted.append(token)
                y = loop_tokens.unsqueeze(1)
                try:
                    cur_pred_out, cur_pred_hidden = decoder.predict(
                        y=y, state=cur_pred_hidden, add_sos=False, batch_size=1
                    )
                except Exception as exc:
                    pb_utils.Logger.log_warning(f"[RNNT display] label-loop predictor step failed: {exc}")
                    break
                symbols += 1
                loop_logits = joint.joint(f, cur_pred_out)
                loop_tokens = loop_logits.squeeze(1).squeeze(1).argmax(-1)
                loop_is_blank = loop_tokens == blank_id

        if is_blank:
            state["blank_count"] += 1
        else:
            state["blank_count"] = 0
            state["nonblank_total"] += 1
            was_confirmed = state.get("speech_confirmed", False)
            if state["nonblank_total"] >= RNNT_DISPLAY_GATE_FRAMES:
                state["speech_confirmed"] = True
            if emitted:
                emitted_text = self._rnnt_display_decode_text(emitted)
                pb_utils.Logger.log_verbose(
                    f"[RNNT display] [seq {seq_id}] frame={frame_idx} emitted={emitted} "
                    f"text={emitted_text!r} nonblank_total={state['nonblank_total']} "
                    f"blank_count={state['blank_count']} confirmed={state['speech_confirmed']}"
                )
            if not was_confirmed and state.get("speech_confirmed", False):
                pb_utils.Logger.log_verbose(
                    f"[RNNT display] [seq {seq_id}] speech gate opened at "
                    f"{state['nonblank_total']} nonblank frames"
                )

        word_acc = list(state.get("_punct_word_acc", []))
        punct_bias_val = float(state.get("_punct_bias_val", 0.0))
        if emitted:
            emitted_punct = [t for t in emitted if t in self._rnnt_punct_ids_set]
            emitted_nonpunct = [t for t in emitted if t not in self._rnnt_punct_ids_set]
            if emitted_punct:
                word_acc = []
                punct_bias_val = 0.0
            elif emitted_nonpunct:
                word_acc.extend(emitted_nonpunct)
                punct_bias_val = 0.0
        elif word_acc and is_blank and state["blank_count"] >= RNNT_PUNCT_BIAS_MIN_SILENCE_FRAMES:
            punct_bias_val += 1.0

        state["pred_out"] = cur_pred_out
        state["pred_hidden"] = cur_pred_hidden
        state["y_sequence"] = state.get("y_sequence", []) + emitted
        state["_punct_word_acc"] = word_acc
        state["_punct_bias_val"] = punct_bias_val
        seq_state.rnnt_display_state = state

    def _rnnt_step_transcript_display(
        self, asr_emb: torch.Tensor, seq_ids_batched: List[int], max_audio_idx: int, step_offset: int,
    ) -> None:
        if asr_emb is None or self.model.rnnt_decoder is None:
            return

        bos_id = self.model.tokenizer.bos_id
        for batch_idx, seq_id in enumerate(seq_ids_batched):
            seq_state = self._seq_mgr[seq_id]
            if seq_state.audio_chunk_idx >= max_audio_idx:
                frame_idx = max_audio_idx - (self.options.num_chunks_per_inference - 1) + step_offset
            else:
                frame_idx = seq_state.audio_chunk_idx
            frame_idx = max(0, min(frame_idx, max_audio_idx))
            self._rnnt_display_decode_frame(asr_emb, batch_idx, seq_id, frame_idx)

            t = seq_state.decoder_global_step
            agent_bos_fired = (
                t < seq_state.generated_text_tokens.shape[1]
                and int(seq_state.generated_text_tokens[0, t].item()) == bos_id
            )
            self._append_rnnt_display_output(seq_state, agent_bos_fired)

    def _rnnt_eou_decode_frame(
        self, asr_emb: torch.Tensor, seq_ids_batched: List[int], frame_indices: List[int]
    ) -> None:
        """Batched RNNT decode + raw counter update for one encoder frame per sequence.

        Runs one joint step on each row's selected encoder frame, updates
        rolling_density, rnnt_consecutive_speech_frames, rnnt_silent_frames, and
        rnnt_nonblank_total per sequence, then writes predictor state back.
        BOU/EOU decisions are made in _apply_turn_taking_from_rnnt_states.
        """
        if asr_emb is None or self.model.rnnt_decoder is None:
            return

        B = len(seq_ids_batched)
        decoder = self.model.rnnt_decoder.decoding.decoder
        joint = self.model.rnnt_decoder.decoding.joint
        blank_id = getattr(self.model.rnnt_decoder, "blank_id", joint._num_classes - 1)
        rnnt_dtype = next(joint.parameters()).dtype

        pred_outs = [None] * B
        pred_hiddens = [None] * B
        last_tokens = [None] * B
        punct_word_accs = [[] for _ in range(B)]
        punct_bias_vals = [0.0] * B
        for i, seq_id in enumerate(seq_ids_batched):
            state = self._seq_mgr[seq_id].rnnt_eou_state
            if state is not None:
                pred_outs[i] = state["pred_out"]
                pred_hiddens[i] = state["pred_hidden"]
                last_tokens[i] = state.get("last_token")
                punct_word_accs[i] = list(state.get("_punct_word_acc", []))
                punct_bias_vals[i] = float(state.get("_punct_bias_val", 0.0))

        with torch.inference_mode():
            if any(p is None for p in pred_outs):
                init_out, init_hidden = decoder.predict(y=None, state=None, add_sos=True, batch_size=B)
                for i in range(B):
                    if pred_outs[i] is None:
                        pred_outs[i] = init_out[i : i + 1]
                        pred_hiddens[i] = _rnnt_slice_hidden(init_hidden, i)

            pred_outs = [p[:, -1:, :] if p.dim() == 3 and p.shape[1] > 1 else p for p in pred_outs]
            pred_out = torch.cat(pred_outs, dim=0)  # (B, 1, D_pred)
            pred_hidden = _rnnt_stack_hidden(pred_hiddens)  # (num_layers, B, H) or tuple

            row_indices = torch.arange(B, device=asr_emb.device)
            frame_indices_t = torch.tensor(frame_indices, dtype=torch.long, device=asr_emb.device)
            f = asr_emb[row_indices, frame_indices_t, :].unsqueeze(1).to(rnnt_dtype)  # (B, 1, D_enc)
            logits = joint.joint(f, pred_out)  # (B, 1, 1, V)
            tokens = logits.squeeze(1).squeeze(1).argmax(-1)  # (B,)
            is_blank = tokens == blank_id  # (B,) bool

            # NeMo uses one RNNT predictor for both transcript punctuation and
            # turn evidence. Preserve the first prediction above as this
            # frame's blank/non-blank evidence, but advance the EOU predictor
            # through any punctuation injected on a blank frame so subsequent
            # predictions follow the same state trajectory.
            emitted = [[] for _ in range(B)]
            if self._rnnt_punct_ids:
                scores = logits.squeeze(1).squeeze(1)
                punct_tokens = torch.full_like(tokens, blank_id)
                no_punct = torch.ones_like(is_blank)
                for i in range(B):
                    if not is_blank[i].item() or punct_bias_vals[i] <= 0.0:
                        continue
                    biased_scores = scores[i : i + 1].clone()
                    for punct_id in self._rnnt_punct_ids:
                        biased_scores[0, punct_id] += punct_bias_vals[i] * self._rnnt_punct_bias_increments[punct_id]
                    punct_token = int(biased_scores.argmax(-1)[0].item())
                    if punct_token not in self._rnnt_punct_ids_set:
                        continue
                    emitted[i].append(punct_token)
                    punct_tokens[i] = punct_token
                    no_punct[i] = False
                if not no_punct.all().item():
                    punct_pred_out, punct_pred_hidden = decoder.predict(
                        y=punct_tokens.unsqueeze(1), state=pred_hidden, add_sos=False, batch_size=B,
                    )
                    mask = no_punct.view(B, 1, 1).expand_as(pred_out)
                    pred_out = torch.where(mask, pred_out, punct_pred_out)
                    pred_hidden = _rnnt_masked_update_hidden(no_punct, pred_hidden, punct_pred_hidden)

            # Keep the first prediction's blank/non-blank result for turn-taking,
            # but drain all labels for this encoder frame before moving on. This
            # keeps the predictor state aligned with NeMo's greedy RNNT decoder.
            loop_tokens = tokens
            loop_is_blank = is_blank
            symbols = 0
            while not loop_is_blank.all() and symbols < RNNT_MAX_SYMBOLS:
                y = torch.where(loop_is_blank, torch.full_like(loop_tokens, blank_id), loop_tokens).unsqueeze(1)
                new_pred_out, new_pred_hidden = decoder.predict(y=y, state=pred_hidden, add_sos=False, batch_size=B,)
                mask = loop_is_blank.view(B, 1, 1).expand_as(pred_out)
                pred_out = torch.where(mask, pred_out, new_pred_out)
                pred_hidden = _rnnt_masked_update_hidden(loop_is_blank, pred_hidden, new_pred_hidden)
                for i in range(B):
                    if not loop_is_blank[i].item():
                        last_tokens[i] = loop_tokens[i].item()
                symbols += 1
                loop_logits = joint.joint(f, pred_out)
                loop_tokens = loop_logits.squeeze(1).squeeze(1).argmax(-1)
                loop_is_blank = loop_tokens == blank_id

            for i, seq_id in enumerate(seq_ids_batched):
                seq_state = self._seq_mgr[seq_id]
                _blank = is_blank[i].item()
                if not seq_state.rnnt_agent_speaking:
                    seq_state.rnnt_rolling_density = (
                        RNNT_DENSITY_ALPHA * (0.0 if _blank else 1.0)
                        + (1.0 - RNNT_DENSITY_ALPHA) * seq_state.rnnt_rolling_density
                    )
                if not _blank:
                    seq_state.rnnt_silent_frames = 0
                    seq_state.rnnt_consecutive_speech_frames += 1
                    seq_state.fn_call["kronik_last_voice"] = seq_state.decoder_global_step
                    seq_state.rnnt_nonblank_total += 1
                    if seq_state.rnnt_consecutive_speech_frames >= RNNT_FC_INTERRUPT_FRAMES:
                        fn_call = seq_state.fn_call
                        if fn_call.get("extracting_tool", False) and not fn_call.get("extract_cancelled", False):
                            fn_call["extract_cancelled"] = True
                            pb_utils.Logger.log_info(
                                f"[RNNT] [seq {seq_id}] FC extraction cancelled after "
                                f"{seq_state.rnnt_consecutive_speech_frames} speech frames"
                                f" (step {seq_state.decoder_global_step})"
                            )
                        if fn_call.get("injecting_response", False) and not fn_call.get("inject_cancelled", False):
                            fn_call["inject_cancelled"] = True
                            pb_utils.Logger.log_info(
                                f"[RNNT] [seq {seq_id}] FC injection cancelled after "
                                f"{seq_state.rnnt_consecutive_speech_frames} speech frames"
                                f" (step {seq_state.decoder_global_step})"
                            )
                else:
                    seq_state.rnnt_consecutive_speech_frames = 0
                    seq_state.rnnt_silent_frames += 1

                emitted_punct = [t for t in emitted[i] if t in self._rnnt_punct_ids_set]
                emitted_nonpunct = [t for t in emitted[i] if t not in self._rnnt_punct_ids_set]
                if emitted_punct:
                    punct_word_accs[i] = []
                    punct_bias_vals[i] = 0.0
                elif emitted_nonpunct:
                    punct_word_accs[i].extend(emitted_nonpunct)
                    punct_bias_vals[i] = 0.0
                elif (
                    punct_word_accs[i]
                    and _blank
                    and seq_state.rnnt_silent_frames >= RNNT_PUNCT_BIAS_MIN_SILENCE_FRAMES
                ):
                    punct_bias_vals[i] += 1.0

        for i, seq_id in enumerate(seq_ids_batched):
            self._seq_mgr[seq_id].rnnt_eou_state = {
                "pred_out": pred_out[i : i + 1],
                "pred_hidden": _rnnt_slice_hidden(pred_hidden, i),
                "last_token": last_tokens[i],
                "_punct_word_acc": punct_word_accs[i],
                "_punct_bias_val": punct_bias_vals[i],
            }

    def _rnnt_step_eou(
        self, asr_emb: torch.Tensor, seq_ids_batched: List[int], max_audio_idx: int, step_offset: int,
    ) -> None:
        """NeMo-style low-level RNNT path — EOU detection only, no transcription.

        Processes exactly one encoder frame per sequence, using each sequence's
        own audio_chunk_idx. Batches all sequences together: one joint.joint() +
        one conditional decoder.predict() call.
        """
        frame_indices = []
        for seq_id in seq_ids_batched:
            seq_state = self._seq_mgr[seq_id]
            if seq_state.audio_chunk_idx >= max_audio_idx:
                frame_idx = max_audio_idx - (self.options.num_chunks_per_inference - 1) + step_offset
            else:
                frame_idx = seq_state.audio_chunk_idx
            frame_indices.append(max(0, min(frame_idx, max_audio_idx)))
        self._rnnt_eou_decode_frame(asr_emb, seq_ids_batched, frame_indices)

    def _apply_turn_taking_from_rnnt_states(self, seq_id: int, seq_state) -> None:
        t = seq_state.decoder_global_step
        bos_id = self.model.tokenizer.bos_id
        eos_id = self.model.tokenizer.eos_id

        # ── window / threshold ────────────────────────────── NeMo L2934-2956
        lookback_start = max(0, t - FORCE_TURN_TAKING_THRESHOLD)
        window = seq_state.generated_text_tokens[0, lookback_start:t]
        current_tok = seq_state.generated_text_tokens[0, t].item()

        density = seq_state.rnnt_rolling_density
        if seq_state.rnnt_first_turn:
            effective_min = RNNT_BOU_MIN_FRAMES_FIRST_TURN
        elif 0.0 < density < RNNT_DENSITY_THRESHOLD:
            effective_min = RNNT_DENSITY_LOW_MIN
        else:
            effective_min = RNNT_BOU_MIN_FRAMES

        # A second BOS before EOS would create a nested agent turn. Close the
        # existing turn instead so the next decoder step receives a valid EOS
        # boundary and can open a new turn cleanly.
        if current_tok == bos_id and seq_state.rnnt_agent_speaking:
            seq_state.generated_text_tokens[0, t] = eos_id
            current_tok = eos_id
            pb_utils.Logger.log_verbose(f"[RNNT] [seq {seq_id}] duplicate agent BOS -> EOS at step {t}")

        # ── agent_speaking sync ───────────────────────────── NeMo L2983-3003
        if current_tok == bos_id:
            seq_state.rnnt_agent_speaking = True
            seq_state.rnnt_first_turn = False
            seq_state.rnnt_user_speaking = False
            seq_state.rnnt_nonblank_total = 0
            seq_state.rnnt_turn_text_tokens = 0

        # EOS is an AGENT -> IDLE transition.  The decoder can emit EOS while
        # Riva is still listening; treating that idle EOS as a completed agent
        # turn discards already-confirmed user speech and permits a later
        # model-native BOS to start a duplicate/partial response.
        if seq_state.rnnt_agent_speaking and current_tok == eos_id:
            seq_state.rnnt_agent_speaking = False
            seq_state.rnnt_user_speaking = False
            seq_state.rnnt_nonblank_total = 0
            seq_state.rnnt_agent_talking_frames = 0

        # ── noise reset ───────────────────────────────────── NeMo L3006-3008
        if (
            seq_state.rnnt_silent_frames >= RNNT_NOISE_RESET_FRAMES
            and not seq_state.rnnt_user_speaking
            and not seq_state.rnnt_agent_speaking
        ):
            seq_state.rnnt_nonblank_total = 0

        # ── BOU ──────────────────────────────────────────── NeMo L3010-3012
        if (
            seq_state.rnnt_consecutive_speech_frames >= effective_min or seq_state.rnnt_nonblank_total >= effective_min
        ) and not seq_state.rnnt_agent_speaking:
            if not seq_state.rnnt_user_speaking:
                pb_utils.Logger.log_verbose(
                    f"[RNNT] [seq {seq_id}] BOU at {seq_state.rnnt_consecutive_speech_frames} consec"
                    f" / {seq_state.rnnt_nonblank_total} total"
                    f" (density={density:.3f}, min={effective_min}, step={t})"
                )
            seq_state.rnnt_user_speaking = True
            seq_state.fn_call["kronik_last_voice"] = seq_state.decoder_global_step

        # ── EOU ──────────────────────────────────────────── NeMo L3025-3038
        if (
            seq_state.rnnt_silent_frames >= RNNT_EOS_SILENCE_FRAMES
            and seq_state.rnnt_user_speaking
            and not seq_state.rnnt_agent_speaking
        ):
            if not (window == bos_id).any() and current_tok != bos_id:
                seq_state.generated_text_tokens[0, t] = bos_id
                seq_state.rnnt_user_speaking = False
                seq_state.rnnt_nonblank_total = 0
                seq_state.rnnt_agent_speaking = True
                seq_state.rnnt_first_turn = False
                seq_state.rnnt_turn_text_tokens = 0
                pb_utils.Logger.log_verbose(
                    f"[RNNT] [seq {seq_id}] EOU -> agent BOS at step {t}"
                    f" (blank_cnt={seq_state.rnnt_silent_frames}, density={density:.3f})"
                )
            return

        # ── talking_frames / text_token counters ─────────── NeMo L3073-3097, L3109-3114
        if seq_state.rnnt_agent_speaking and current_tok != eos_id:
            seq_state.rnnt_agent_talking_frames += 1
            if current_tok not in (bos_id, eos_id, self.pad_id):
                seq_state.rnnt_turn_text_tokens += 1
        elif not seq_state.rnnt_agent_speaking:
            seq_state.rnnt_agent_talking_frames = 0

        # ── MAX RESPONSE ─────────────────────────────────── NeMo L3073-3093
        if (
            RNNT_MAX_AGENT_RESPONSE_FRAMES > 0
            and seq_state.rnnt_agent_speaking
            and current_tok != eos_id
            and seq_state.rnnt_agent_talking_frames >= RNNT_MAX_AGENT_RESPONSE_FRAMES
        ):
            seq_state.generated_text_tokens[0, t] = eos_id
            seq_state.rnnt_agent_speaking = False
            _frames = seq_state.rnnt_agent_talking_frames
            seq_state.rnnt_agent_talking_frames = 0
            pb_utils.Logger.log_info(
                f"[RNNT] [seq {seq_id}] MAX RESPONSE: agent EOS at step {t}"
                f" (talking_frames={_frames} >= {RNNT_MAX_AGENT_RESPONSE_FRAMES})"
            )
            return

        # ── TTS CAP ──────────────────────────────────────── NeMo L3099-3127
        if (
            RNNT_TTS_RATIO_CAP > 0
            and seq_state.rnnt_agent_speaking
            and current_tok != eos_id
            and seq_state.rnnt_turn_text_tokens >= RNNT_TTS_MIN_TOKENS
            and seq_state.rnnt_agent_talking_frames >= RNNT_TTS_RATIO_CAP * seq_state.rnnt_turn_text_tokens
        ):
            seq_state.generated_text_tokens[0, t] = eos_id
            seq_state.rnnt_agent_speaking = False
            _frames = seq_state.rnnt_agent_talking_frames
            _tokens = seq_state.rnnt_turn_text_tokens
            seq_state.rnnt_agent_talking_frames = 0
            seq_state.rnnt_turn_text_tokens = 0
            pb_utils.Logger.log_info(
                f"[RNNT] [seq {seq_id}] TTS CAP: agent EOS at step {t}"
                f" ({_frames} frames >= {RNNT_TTS_RATIO_CAP} x {_tokens} text tokens)"
            )
            return

        # ── BARGE-IN ─────────────────────────────────────── NeMo L3129-3140
        if (
            seq_state.rnnt_consecutive_speech_frames >= RNNT_BARGE_IN_FRAMES
            and seq_state.rnnt_agent_speaking
            and current_tok != eos_id
        ):
            seq_state.generated_text_tokens[0, t] = eos_id
            seq_state.rnnt_agent_speaking = False
            _consec = seq_state.rnnt_consecutive_speech_frames
            seq_state.rnnt_consecutive_speech_frames = 0
            seq_state.rnnt_nonblank_total = 0
            pb_utils.Logger.log_info(
                f"[RNNT] [seq {seq_id}] BARGE-IN: agent EOS at step {t}"
                f" (consec={_consec}, threshold={RNNT_BARGE_IN_FRAMES})"
            )

    @_log_time("Generate tokens")
    def generate_tokens(
        self, encoded_audio: torch.Tensor, seq_ids_batched: List[int], asr_emb: Optional[torch.Tensor] = None
    ):
        """
        source_encoded: (B, T, D)
        seq_ids_batched: List[int]
        asr_emb: raw encoder output (B, T, D) for RNNT decoding, or None
        """
        logger = pb_utils.Logger

        logger.log_verbose("*** Generate Tokens ***")

        for step_offset in range(self.options.num_chunks_per_inference):
            with _time_block(f"Step {step_offset}"):
                self._llm_step(encoded_audio, seq_ids_batched, step_offset)
                if asr_emb is not None and self.model.rnnt_decoder is not None:
                    self._rnnt_step_eou(
                        asr_emb, seq_ids_batched, self.options.max_chunks_for_inference - 1, step_offset
                    )
                for seq_id in seq_ids_batched:
                    seq_state = self._seq_mgr[seq_id]
                    self._apply_turn_taking_from_rnnt_states(seq_id, seq_state)
                if asr_emb is not None and self.model.rnnt_decoder is not None:
                    self._rnnt_step_transcript_display(
                        asr_emb, seq_ids_batched, self.options.max_chunks_for_inference - 1, step_offset
                    )
                self.vllm_tts_step(seq_ids_batched)

                for seq_id in seq_ids_batched:
                    seq_state = self._seq_mgr[seq_id]
                    if seq_state.audio_chunk_idx < self.options.max_chunks_for_inference - 1:
                        seq_state.audio_chunk_idx += 1
                    else:
                        seq_state.audio_chunk_idx = self.options.max_chunks_for_inference - 1
                    seq_state.decoder_global_step += 1

    @_log_time("Audio decode time", cuda_sync=True)
    def decode_audio_tokens(self, codes, seq_ids_batched: List[int], flush: bool = False):
        """
        Converts a batch of audio codec tokens into a waveform.
        Uses GatheredCausalConv1dCache so each batch row uses its own sequence's codec cache.
        """
        logger = pb_utils.Logger
        if len(codes) == 0:
            return torch.tensor([[]], device=self.model.device), None

        per_item_caches = {seq_id: self._seq_mgr[seq_id].codec_cache for seq_id in seq_ids_batched}
        cache = GatheredCausalConv1dCache(seq_ids_batched, per_item_caches)
        code_lens = torch.full((codes.shape[0],), codes.shape[1], device=self.model.device, dtype=torch.long)

        with fp32_precision(), torch.no_grad():
            codes = codes.to(torch.long)
            logger.log_verbose(f"codes: {codes.shape}")
            wav, wav_len = self.model.codec.decode(codes, code_lens, cache=cache, flush=flush,)
            logger.log_verbose(f"wav: {wav.shape}")

        return wav, wav_len

    def token_to_text_with_bos_eos(self, token_ids):
        token_ids = token_ids[token_ids != self.pad_id]
        text = ""
        for token in token_ids:
            if token == self.model.tokenizer.bos or token == self.model.tokenizer.eos:
                text += self.model.tokenizer.ids_to_tokens([token])[0]
            else:
                text += self.model.tokenizer.ids_to_text([token])
        return text

    def _reset_fn_call_state(self, fn_call: Dict[str, Any], seq_state=None) -> None:
        fn_call["state"] = "idle"
        fn_call["frames_in_state"] = 0
        fn_call["request"] = ""
        fn_call["response"] = ""
        fn_call["response_tokens"] = deque()
        fn_call["ack_tokens"] = deque()
        fn_call["extracting_tool"] = False
        fn_call["extract_cancelled"] = False
        fn_call["pending_function_text"] = ""
        # Mark agent idle so that post-FC PAD frames are immediately silenced
        # (no open turn, LLM hasn't emitted BOS for its verbal response yet).
        if seq_state is not None:
            seq_state.tts_agent_idle = True
            seq_state.tts_in_turn_content = 0
            seq_state.tts_in_turn_pads = 0
            # Kronik patch: FC teardown re-opens the turn-start window — the spoken
            # answer's BOS comes right after, but fast-path injection has raced the
            # step counter far past the voice-based window by then.
            fn_call["kronik_last_voice"] = getattr(seq_state, "decoder_global_step", 0)

    def _get_silence_audio_chunk(self) -> torch.Tensor:
        if self._silence_audio_chunk is None:
            samples_per_step = int(MODEL_OUTPUT_SAMPLE_RATE * 0.08)  # 80ms per step
            n_samples = self.options.num_chunks_per_inference * samples_per_step
            self._silence_audio_chunk = torch.zeros(1, 1, n_samples, dtype=torch.float32, device=self.model.device)
        return self._silence_audio_chunk

    def _maybe_start_fast_path_threads(self, seq_ids_batched: List[int]) -> None:
        """
        After each decode pass, check whether any sequence just entered a state that
        should be driven as fast as possible (not gated by realtime audio).  Starts a
        background daemon thread for each such sequence if one is not already running.
        """
        if self._encoded_audio_dim is None:
            return

        logger = pb_utils.Logger
        for seq_id in seq_ids_batched:
            seq_state = self._seq_mgr[seq_id]
            fn_call = seq_state.fn_call

            # Tool-token extraction: sotc_id was just seen → spin until eotc_id.
            if (
                fn_call["state"] == "waiting_for_request"
                and not fn_call.get("extracting_tool", False)
                and seq_id not in self._extraction_threads
            ):
                fn_call["extracting_tool"] = True
                fn_call["extract_cancelled"] = False
                t = threading.Thread(
                    target=self._fast_extract_tool_tokens, args=(seq_id,), daemon=True, name=f"tool_extract_{seq_id}",
                )
                self._extraction_threads[seq_id] = t
                t.start()
                logger.log_info(
                    f"[seq {seq_id}] Triggered fast tool-token extraction thread "
                    f"(step: {seq_state.decoder_global_step})"
                )

            # Response injection: tool response received → spin until all tokens consumed.
            elif (
                fn_call["state"] == "process_response"
                and fn_call.get("response_tokens")
                and not fn_call.get("injecting_response", False)
                and seq_id not in self._injection_threads
            ):
                fn_call["injecting_response"] = True
                t = threading.Thread(
                    target=self._fast_inject_response_tokens,
                    args=(seq_id,),
                    daemon=True,
                    name=f"tool_inject_{seq_id}",
                )
                self._injection_threads[seq_id] = t
                t.start()
                logger.log_info(
                    f"[seq {seq_id}] Triggered fast response-token injection thread "
                    f"(step: {seq_state.decoder_global_step})"
                )

    @_log_time("Fast tool-token extraction", debug_only=False)
    def _fast_extract_tool_tokens(self, seq_id: int) -> None:
        """
        Spin _llm_step with dummy zero audio until eotc_id is produced, extracting all
        tool-call tokens as fast as possible (not gated by realtime audio arrival).

        Runs in a background daemon thread.  While this is running, infer() skips the
        sequence and returns empty responses so batch processing is not blocked.
        """
        logger = pb_utils.Logger
        logger.log_info(f"[seq {seq_id}] Fast tool-token extraction started")

        seq_state = self._seq_mgr[seq_id]
        fn_call = seq_state.fn_call

        with torch.no_grad(), torch.inference_mode():
            dummy_audio = torch.zeros(
                1,
                self.options.max_chunks_for_inference,
                self._encoded_audio_dim,
                device=self.model.device,
                dtype=DTYPE,
            )

            max_tool_tokens = MAX_TOOL_TOKENS
            for _ in range(max_tool_tokens):
                if fn_call.get("extract_cancelled", False):
                    logger.log_info(f"[seq {seq_id}] Fast extraction cancelled")
                    break
                if not self._seq_mgr.is_started(seq_id):
                    logger.log_info(f"[seq {seq_id}] Sequence removed during extraction, stopping")
                    break
                buffer_capacity = seq_state.generated_text_tokens.shape[1]
                if seq_state.decoder_global_step >= buffer_capacity:
                    logger.log_error(
                        f"[seq {seq_id}] Max decoder steps ({buffer_capacity}) reached during fast extraction, stopping"
                    )
                    seq_state.steps_exhausted = True
                    self._reset_fn_call_state(fn_call, seq_state)
                    break

                self._llm_step(dummy_audio, [seq_id])

                step_idx = seq_state.decoder_global_step
                fn_token = seq_state.generated_function_tokens[0, step_idx].item()

                seq_state.audio_chunk_idx = min(
                    seq_state.audio_chunk_idx + 1, self.options.max_chunks_for_inference - 1
                )
                seq_state.decoder_global_step += 1

                if fn_token == self.eotc_id:
                    sanitized = self.sanitize_function_text(fn_call["request"])
                    if sanitized:
                        fn_call["request"] = sanitized
                        fn_call["pending_function_text"] = sanitized
                        # Mirror the ack/waiting_for_response transition from parse_function_call_tokens
                        ack_message = self._select_tool_ack_message(seq_state, sanitized)
                        if ack_message:
                            fn_call["ack_tokens"] = self._build_ack_tokens(ack_message)
                            fn_call["state"] = "speaking_ack"
                            fn_call["frames_in_state"] = 0
                            logger.log_info(
                                f"[seq {seq_id}] fast extract -> speaking_ack, ack: {ack_message!r} "
                                f"(step: {seq_state.decoder_global_step})"
                            )
                        else:
                            fn_call["state"] = "waiting_for_response"
                            fn_call["frames_in_state"] = 0
                            logger.log_info(
                                f"[seq {seq_id}] fast extract -> waiting_for_response "
                                f"(step: {seq_state.decoder_global_step})"
                            )
                    else:
                        logger.log_error(f"[seq {seq_id}] Fast extract: failed to sanitize {fn_call['request']!r}")
                        self._reset_fn_call_state(fn_call, seq_state)
                    break
                elif fn_token != self.pad_id:
                    fn_call["request"] += self.model.tokenizer.ids_to_text([fn_token])
                    # logger.log_info(f"[seq {seq_id}] fast extract accumulated: {fn_call['request']!r}")
            else:
                logger.log_error(
                    f"[seq {seq_id}] Fast extract: exceeded {max_tool_tokens} steps without eotc_id, "
                    f"accumulated: {fn_call['request']!r}, resetting"
                )
                self._reset_fn_call_state(fn_call, seq_state)

        self._sync_audio_buffer_to_step(seq_id)
        fn_call["extracting_tool"] = False
        self._extraction_threads.pop(seq_id, None)
        logger.log_info(f"[seq {seq_id}] Fast tool-token extraction done, state: {fn_call['state']}")

    @_log_time("Fast response-token injection", debug_only=False)
    def _fast_inject_response_tokens(self, seq_id: int) -> None:
        """
        Spin _llm_step with dummy zero audio, injecting tool-response tokens through the
        function channel as fast as possible (not gated by realtime audio arrival).

        Runs in a background daemon thread.  infer() skips the sequence while this runs
        and resumes normal processing once all response_tokens are consumed (state → idle).
        """
        logger = pb_utils.Logger
        logger.log_info(f"[seq {seq_id}] Fast response-token injection started")

        seq_state = self._seq_mgr[seq_id]
        fn_call = seq_state.fn_call

        with torch.no_grad(), torch.inference_mode():
            dummy_audio = torch.zeros(
                1,
                self.options.max_chunks_for_inference,
                self._encoded_audio_dim,
                device=self.model.device,
                dtype=DTYPE,
            )

            while fn_call["state"] == "process_response":
                if fn_call.get("extract_cancelled", False):
                    logger.log_info(f"[seq {seq_id}] Response injection cancelled")
                    break
                if not self._seq_mgr.is_started(seq_id):
                    logger.log_info(f"[seq {seq_id}] Sequence removed during response injection, stopping")
                    break
                buffer_capacity = seq_state.generated_text_tokens.shape[1]
                if seq_state.decoder_global_step >= buffer_capacity:
                    logger.log_error(
                        f"[seq {seq_id}] Max decoder steps ({buffer_capacity}) reached during response injection, stopping"
                    )
                    seq_state.steps_exhausted = True
                    self._reset_fn_call_state(fn_call, seq_state)
                    break

                self._llm_step(dummy_audio, [seq_id])
                # _llm_step pops one response_token and writes it into generated_function_tokens;
                # when the deque empties it calls _reset_fn_call_state → state becomes "idle".

                seq_state.audio_chunk_idx = min(
                    seq_state.audio_chunk_idx + 1, self.options.max_chunks_for_inference - 1
                )
                seq_state.decoder_global_step += 1

        self._sync_audio_buffer_to_step(seq_id)
        fn_call["injecting_response"] = False
        self._injection_threads.pop(seq_id, None)
        logger.log_info(f"[seq {seq_id}] Fast response-token injection done, state: {fn_call['state']}")

    def _sync_audio_buffer_to_step(self, seq_id: int) -> None:
        """Pad audio_buffer with zeros so it satisfies the min_samples check in infer().

        Fast path threads advance decoder_global_step using dummy zero audio without
        touching audio_buffer.  On return to normal inference, infer() requires:
            audio_buffer.shape[1] >= (decoder_global_step + num_chunks_per_inference)
                                      * num_samples_per_chunk
        Padding with zeros is correct because the fast path used zero audio anyway.
        """
        if not self._seq_mgr.is_started(seq_id):
            return
        seq_state = self._seq_mgr[seq_id]
        min_samples = (
            seq_state.decoder_global_step + self.options.num_chunks_per_inference
        ) * self.options.num_samples_per_chunk
        current = seq_state.audio_buffer.shape[1]
        if current < min_samples:
            padding = torch.zeros(1, min_samples - current, device=self.model.device)
            seq_state.audio_buffer = torch.cat([seq_state.audio_buffer, padding], dim=1)
            pb_utils.Logger.log_info(
                f"[seq {seq_id}] Padded audio_buffer by {min_samples - current} samples "
                f"({current} -> {min_samples}) after fast path"
            )

    def sanitize_function_text(self, function_text):
        """
        Sanitize tool calling tokens.
        """
        logger = pb_utils.Logger
        function_text = function_text.strip()
        # Check if the function_text contains <TOOLCALL> and </TOOLCALL> and a valid JSON
        if function_text and "<TOOLCALL>" in function_text and "</TOOLCALL>" in function_text:
            function_text = function_text.split("<TOOLCALL>")[1].split("</TOOLCALL>")[0]
            try:
                json.loads(function_text)
                return function_text
            except json.JSONDecodeError:
                logger.log_error(f"Failed to parse function text as JSON: {function_text}")
                return ""
        else:
            logger.log_error(f"Function call request does not contain <TOOLCALL> and </TOOLCALL>: {function_text}")
            return ""

    def parse_function_call_tokens(self, seq_id, token_ids):
        """
        Parse function call tokens and advance the function calling state machine.
        """
        logger = pb_utils.Logger
        token_ids = token_ids[token_ids != self.pad_id]

        seq_state = self._seq_mgr[seq_id]
        fn_call = seq_state.fn_call

        if fn_call["state"] == "waiting_for_request":
            fn_call["frames_in_state"] += 1
            if fn_call["frames_in_state"] >= seq_state.fn_call_request_timeout:
                logger.log_error(
                    f"Function call request timed out after {fn_call['frames_in_state']} frames, resetting to idle"
                )
                self._reset_fn_call_state(fn_call, seq_state)
                return ""
        elif fn_call["state"] == "waiting_for_response":
            fn_call["frames_in_state"] += 1
            if fn_call["frames_in_state"] >= seq_state.fn_call_response_timeout:
                logger.log_error(
                    f"Function call response timed out after {fn_call['frames_in_state']} frames, resetting to idle"
                )
                self._reset_fn_call_state(fn_call, seq_state)
                return ""

        sanitized_function_text = ""
        for token in token_ids:
            if token == self.sotc_id:
                if fn_call["state"] != "idle":
                    logger.log_error(f"Already in state {fn_call['state']}, skipping SOTC token")
                    continue
                self._reset_fn_call_state(fn_call)
                fn_call["state"] = "waiting_for_request"
                logger.log_info(
                    f"Function call state transition: idle -> waiting_for_request (seq_id: {seq_id}) (step: {seq_state.decoder_global_step})"
                )

            elif token == self.eotc_id:
                if fn_call["state"] != "waiting_for_request":
                    logger.log_error("Not in waiting for request state, skipping EOTC token")
                    continue
                sanitized_function_text = self.sanitize_function_text(fn_call["request"])
                if sanitized_function_text:
                    fn_call["request"] = sanitized_function_text
                    fn_call["state"] = "request_received"
                    logger.log_info(
                        f"Function call state transition: waiting_for_request -> request_received (seq_id: {seq_id}) (step: {seq_state.decoder_global_step})"
                    )
                else:
                    logger.log_error(f"Failed to sanitize function calling tokens {fn_call['request']}")
                    self._reset_fn_call_state(fn_call, seq_state)
            else:
                if fn_call["state"] == "waiting_for_request":
                    fn_call["request"] += self.model.tokenizer.ids_to_text([token])
                    logger.log_info(f"Function request text: {fn_call['request']}")

        # Transition out of request_received: speak ack immediately if available, else wait for response
        if sanitized_function_text:
            ack_message = self._select_tool_ack_message(seq_state, sanitized_function_text)

            if ack_message:
                fn_call["ack_tokens"] = self._build_ack_tokens(ack_message)
                fn_call["state"] = "speaking_ack"
                fn_call["frames_in_state"] = 0
                logger.log_info(
                    f"Function call state transition: request_received -> speaking_ack (seq_id: {seq_id}) (step: {seq_state.decoder_global_step}) ack: {ack_message!r}"
                )
            else:
                fn_call["state"] = "waiting_for_response"
                fn_call["frames_in_state"] = 0
                logger.log_info(
                    f"Function call state transition: request_received -> waiting_for_response (seq_id: {seq_id}) (step: {seq_state.decoder_global_step})"
                )

        return sanitized_function_text

    @_log_time("Decode output from tokens")
    def decode_output_from_tokens(self, seq_ids_batched):
        # Decode text
        logger = pb_utils.Logger

        logger.log_verbose("*** Decode Output From Tokens ***")

        audio_tokens_to_decode_batched = []
        generated_text_batched = []
        generated_function_text_batched = []
        generated_asr_text_batched = []
        generated_audio_batched = []
        for seq_id in seq_ids_batched:
            seq_state = self._seq_mgr[seq_id]
            start_idx = max(0, self._seq_mgr[seq_id].decoder_global_step - self.options.num_chunks_per_inference,)
            end_idx = start_idx + self.options.num_chunks_per_inference

            token_ids = seq_state.generated_text_tokens[:, start_idx:end_idx].cpu()
            generated_text = self.token_to_text_with_bos_eos(token_ids)

            token_ids = seq_state.generated_function_tokens[:, start_idx:end_idx].cpu()
            generated_function_text = self.parse_function_call_tokens(seq_id, token_ids)

            generated_asr_text = ""
            if seq_state.rnnt_display_pending_text:
                generated_asr_text = seq_state.rnnt_display_pending_text
                seq_state.rnnt_display_pending_text = ""
                logger.log_verbose(f"[RNNT display] [seq {seq_id}] flushing output_asr_text={generated_asr_text!r}")

            # If fast extraction completed while this sequence was skipped, deliver the
            # tool-call JSON now (exactly once) so the client receives it.
            if not generated_function_text and seq_state.fn_call.get("pending_function_text"):
                generated_function_text = seq_state.fn_call["pending_function_text"]
                seq_state.fn_call["pending_function_text"] = ""
                logger.log_info(
                    f"[seq {seq_id}] delivering pending function text from fast extraction: {generated_function_text!r}"
                )

            generated_text_batched.append(generated_text)
            generated_function_text_batched.append(generated_function_text)
            generated_asr_text_batched.append(generated_asr_text)

            history_start_idx = max(0, end_idx - self.options.num_chunks_per_inference)

            audio_tokens_to_decode = seq_state.generated_audio_tokens[:, history_start_idx:end_idx, :]
            audio_tokens_to_decode_batched.append(audio_tokens_to_decode)
        audio_tokens_to_decode_batched = torch.cat(audio_tokens_to_decode_batched, dim=0)

        decoded_audio_wav, wav_len = self.decode_audio_tokens(audio_tokens_to_decode_batched, seq_ids_batched)

        for idx, seq_id in enumerate(seq_ids_batched):
            seq_state = self._seq_mgr[seq_id]
            start_idx = max(0, self._seq_mgr[seq_id].decoder_global_step - self.options.num_chunks_per_inference,)
            logger.log_verbose(f"decoded_audio_wav: {decoded_audio_wav.shape}")

            # 0th item in decoded_audio_wav is the output of the prompt input. Note that the
            # prompt input doesn't include any audio/text tokens to generate a valid output token.
            # Hence, this should be skipped. Otherwise, it will result in an initial static noise.
            if start_idx == 0:
                second_token_idx = self.model.model_cfg.model.speech_generation.model.codec_config.wav_to_token_ratio
                audio_output_wav = decoded_audio_wav[idx, :, second_token_idx:].unsqueeze(0)
            else:
                audio_output_wav = decoded_audio_wav[idx, :, :].unsqueeze(0)
            logger.log_verbose(f"audio_output_wav: {audio_output_wav.shape}")
            generated_audio_batched.append(audio_output_wav)

        return (
            generated_text_batched,
            generated_function_text_batched,
            generated_asr_text_batched,
            generated_audio_batched,
        )

    @_log_time("### Time to infer chunk")
    def infer(
        self, audio_signal: torch.Tensor, function_response_batched: List[str], seq_info: SeqInfo
    ) -> List[InferenceResult]:
        """
        Infer the audio signal and return the generated text and audio.

        Args:
            audio_signal: The audio signal to infer.
                Shape: (B, T)
            function_response_batched: The function response for each sequence.
                Shape: (B,)
            seq_info: The sequence information.
        Returns:
            A list of InferenceResult objects.
        """

        logger = pb_utils.Logger
        logger.log_verbose("**** Called self.infer ****")

        audio_to_encode = []
        buffer_len_tensor = []

        seq_ids_batched = []

        results_batched = [
            InferenceResult(
                generated_text="",
                generated_asr_text="",
                generated_function_text="",
                generated_audio=torch.tensor([[]], device=self.model.device),
            )
            for _ in range(audio_signal.shape[0])
        ]

        logger.log_verbose("Collating inputs")
        max_samples_in_batch = 0
        for batch_idx, seq_id in enumerate(seq_info.sequence_ids):
            seq_id = seq_id.item()

            # Sequence-start requests carry only placeholder audio (system prompt
            # was already processed in execute); skip buffering to keep it clean.
            if seq_info.start_flags[batch_idx].item():
                results_batched[batch_idx].skipped = True
                continue

            # Stop processing sequences that have exhausted the pre-allocated step buffer.
            seq_state_check = self._seq_mgr[seq_id]
            buffer_capacity = seq_state_check.generated_text_tokens.shape[1]
            if seq_state_check.decoder_global_step >= buffer_capacity or seq_state_check.steps_exhausted:
                if not seq_state_check.steps_exhausted:
                    logger.log_error(
                        f"[seq {seq_id}] Max decoder steps ({buffer_capacity}) reached, stopping sequence"
                    )
                    seq_state_check.steps_exhausted = True
                results_batched[batch_idx].error = f"Sequence {seq_id} exceeded max decoder steps ({buffer_capacity})"
                continue

            # While background tool-token extraction or response injection is running,
            # return silence to keep the audio stream active.
            if self._seq_mgr[seq_id].fn_call.get("extracting_tool", False) or self._seq_mgr[seq_id].fn_call.get(
                "injecting_response", False
            ):
                results_batched[batch_idx].generated_audio = self._get_silence_audio_chunk()
                continue

            # Append input audio to buffer
            self._seq_mgr[seq_id].audio_buffer = torch.cat(
                (self._seq_mgr[seq_id].audio_buffer, audio_signal[batch_idx].unsqueeze(0)), dim=1
            )  # (1, T)

            fn_state = self._seq_mgr[seq_id].fn_call["state"]
            fn_response = function_response_batched[batch_idx]

            if fn_response:

                fn_response = format_tool_response_for_nemo(fn_response)
                self._seq_mgr[seq_id].fn_call["response"] = fn_response
                self._seq_mgr[seq_id].fn_call["response_tokens"] = deque(self.model.tokenizer.text_to_ids(fn_response))
                # Kronik patch: a tool response solicits the follow-up answer
                self._seq_mgr[seq_id].fn_call["kronik_last_voice"] = self._seq_mgr[seq_id].decoder_global_step
                logger.log_info(
                    f"seq_id: {seq_id} response: {fn_response!r} "
                    f"Token length:{len(self._seq_mgr[seq_id].fn_call['response_tokens'])}"
                )

                if fn_state == "waiting_for_response":
                    self._seq_mgr[seq_id].fn_call["frames_in_state"] = 0
                    self._seq_mgr[seq_id].fn_call["state"] = "process_response"
                    logger.log_info(
                        f"Function call state transition: waiting_for_response -> process_response (seq_id: {seq_id}) (step: {self._seq_mgr[seq_id].decoder_global_step})"
                    )

                elif fn_state == "speaking_ack":
                    # Tool response arrived while ack is still playing — store it for after ack finishes. Do not change state.
                    logger.log_info(
                        f"Tool response buffered during speaking_ack (seq_id: {seq_id}) (step: {self._seq_mgr[seq_id].decoder_global_step})"
                    )

            min_samples = (
                self._seq_mgr[seq_id].decoder_global_step + self.options.num_chunks_per_inference
            ) * self.options.num_samples_per_chunk

            if self._seq_mgr[seq_id].audio_buffer.shape[1] < min_samples:
                logger.log_verbose(
                    f"Need atleast {min_samples} samples. Have only {self._seq_mgr[seq_id].audio_buffer.shape[1]} samples for seq_id: {seq_id}"
                )
                results_batched[batch_idx].generated_text = ""
                results_batched[batch_idx].generated_function_text = ""
                results_batched[batch_idx].generated_audio = torch.tensor([[]], device=self.model.device)
                results_batched[batch_idx].skipped = True
                continue

            num_samples = min(min_samples, self.options.max_samples_for_inference)
            max_samples_in_batch = max(max_samples_in_batch, num_samples)
            seq_ids_batched.append(seq_id)
            audio_to_encode.append(self._seq_mgr[seq_id].audio_buffer[:, -num_samples:])
            buffer_len_tensor.append(torch.tensor([num_samples], device=self.model.device))

        if len(audio_to_encode) == 0:
            logger.log_verbose("No audio to encode, returning results_batched")
            return results_batched

        for i in range(len(audio_to_encode)):
            if audio_to_encode[i].shape[1] != max_samples_in_batch:
                # Right-pad with zeros to maintain continuity of audio
                audio_to_encode[i] = torch.cat(
                    [
                        audio_to_encode[i],
                        torch.zeros(
                            audio_to_encode[i].shape[0],
                            max_samples_in_batch - audio_to_encode[i].shape[1],
                            device=self.model.device,
                        ),
                    ],
                    dim=1,
                )

        audio_to_encode = torch.cat(audio_to_encode, dim=0)  # (B, T)
        buffer_len_tensor = torch.cat(buffer_len_tensor, dim=0)  # (B,)

        logger.log_verbose("Inputs collated")

        encoded_audio, asr_emb = self.perception(audio_to_encode, buffer_len_tensor)  # (B, T, D)

        self.generate_tokens(encoded_audio, seq_ids_batched, asr_emb=asr_emb)

        generated_text, generated_function_text, generated_asr_text, generated_audio = self.decode_output_from_tokens(
            seq_ids_batched
        )

        self._maybe_start_fast_path_threads(seq_ids_batched)

        # Critical: use seq_to_batch_idx to map results back to original request positions.
        # seq_ids_batched is a subset (skipped sequences excluded), so enumerate indices
        # don't match results_batched positions. Without this, outputs cross wires between streams.
        for gen_idx, seq_id in enumerate(seq_ids_batched):
            original_idx = seq_info.seq_to_batch_idx[seq_id]
            (
                results_batched[original_idx].generated_text,
                results_batched[original_idx].generated_function_text,
                results_batched[original_idx].generated_asr_text,
                results_batched[original_idx].generated_audio,
            ) = (
                generated_text[gen_idx],
                generated_function_text[gen_idx],
                generated_asr_text[gen_idx],
                generated_audio[gen_idx],
            )
        return results_batched

    @_log_time("Execute")
    def execute(self, requests):
        logger = pb_utils.Logger
        responses = []

        audio_signal_batched = []
        function_response_batched = []
        start_flag_batched = []
        end_flag_batched = []
        sequence_id_batched = []
        seq_to_batch_idx = {}

        with torch.no_grad(), torch.inference_mode():
            logger.log_verbose(f"=========Received {requests} requests==========")

            system_prompt_coros = []
            for idx, request in enumerate(requests):
                sequence_id = request.correlation_id()

                start_flag = from_dlpack(pb_utils.get_input_tensor_by_name(request, "START"))  # (1,)
                end_flag = from_dlpack(pb_utils.get_input_tensor_by_name(request, "END"))  # (1,)

                logger.log_verbose(
                    f"Unpacking inputs for request {idx} with sequence_id {sequence_id}. start_flag: {start_flag}, end_flag: {end_flag}"
                )

                if start_flag[0].item():
                    if self._seq_mgr.is_started(sequence_id):
                        logger.log_error(f"Sequence {sequence_id} already started")
                        self._seq_mgr[sequence_id].start = False
                    else:
                        self._seq_mgr.mark_start(sequence_id)
                        arr = pb_utils.get_input_tensor_by_name(request, "system_prompt").as_numpy()
                        system_prompt = arr.ravel()[0].decode("utf-8").strip()
                        self._seq_mgr[sequence_id].system_prompt = system_prompt
                        if system_prompt:
                            system_prompt_coros.append(
                                self._send_system_prompt(sequence_id, self._seq_mgr[sequence_id].system_prompt)
                            )
                        else:
                            logger.log_info(f"No system prompt to send to sequence {sequence_id}")

                elif end_flag:
                    self._seq_mgr.mark_end(sequence_id)
                elif self._seq_mgr.is_started(sequence_id):
                    self._seq_mgr[sequence_id].start = False

                audio_signal = from_dlpack(
                    pb_utils.get_input_tensor_by_name(request, "audio_signal")
                )  # (1, T) - 1 since this is a single batch item at this point

                # Accept single-chunk (80ms) or multi-chunk inputs (e.g. 160ms when NUM_CHUNKS_PER_INFERENCE=2).
                # Warmup and finalize still send single 80ms chunks; normal inference sends the full batch.
                expected_multi = self.options.num_samples_per_chunk * self.options.num_chunks_per_inference
                if audio_signal.shape[1] not in (self.options.num_samples_per_chunk, expected_multi):
                    raise ValueError(
                        f"audio_signal shape: {audio_signal.shape}, expected "
                        f"{self.options.num_samples_per_chunk} or {expected_multi} samples"
                    )

                # Get the function response from the request
                function_response = pb_utils.get_input_tensor_by_name(request, "function_response").as_numpy()
                function_response = function_response.ravel()[0].decode("utf-8").strip()

                device = self.model.device
                audio_signal = audio_signal.to(device)

                audio_signal_batched.append(audio_signal)
                function_response_batched.append(function_response)
                start_flag_batched.append(start_flag)
                end_flag_batched.append(end_flag)
                sequence_id_batched.append(sequence_id)
                seq_to_batch_idx[sequence_id] = idx

            if system_prompt_coros:
                self._run_coro(self._gather(*system_prompt_coros))

            audio_signal_batched = torch.cat(audio_signal_batched, dim=0)  # (B, T)
            start_flag_batched = torch.cat(start_flag_batched, dim=0).to(device=device)  # (B,)
            end_flag_batched = torch.cat(end_flag_batched, dim=0).to(device=device)  # (B,)
            sequence_id_batched = torch.tensor(sequence_id_batched).to(device=device)  # (B,)

            seq_info = SeqInfo(
                start_flags=start_flag_batched,
                end_flags=end_flag_batched,
                sequence_ids=sequence_id_batched,
                seq_to_batch_idx=seq_to_batch_idx,
            )

            results = self.infer(audio_signal_batched, function_response_batched, seq_info)

            for idx, result in enumerate(results):
                seq_id = seq_info.sequence_ids[idx].item()

                logger.log_verbose(f"generated_text: {result.generated_text}")
                logger.log_verbose(f"generated_function_text: {result.generated_function_text}")

                self._seq_mgr[seq_id].generated_text.append(result.generated_text)
                self._seq_mgr[seq_id].generated_function_text.append(result.generated_function_text)
                self._seq_mgr[seq_id].generated_audio.append(result.generated_audio)

                output_text = "".join(result.generated_text)
                output_function_text = "".join(result.generated_function_text)
                output_asr_text = result.generated_asr_text
                if end_flag_batched[idx]:
                    seq_state = self._seq_mgr[seq_id]
                    self._append_rnnt_display_output(
                        seq_state, agent_bos_fired=False, force_publish=True, close_turn=True
                    )
                    if seq_state.rnnt_display_pending_text:
                        output_asr_text += seq_state.rnnt_display_pending_text
                        seq_state.rnnt_display_pending_text = ""
                if output_asr_text:
                    logger.log_verbose(
                        f"[RNNT display] [seq {seq_id}] Triton response output_asr_text={output_asr_text!r}"
                    )
                output_text_arr = np.array([output_text.encode("utf-8")], dtype=np.object_)
                output_function_text_arr = np.array([output_function_text.encode("utf-8")], dtype=np.object_)
                output_asr_text_arr = np.array([output_asr_text.encode("utf-8")], dtype=np.object_)

                if end_flag_batched[idx]:
                    logger.log_info(f"Sequence {seq_id} ended")
                    # Cancel any in-flight tool extraction or response injection
                    self._seq_mgr[seq_id].fn_call["extract_cancelled"] = True
                    for threads in (self._extraction_threads, self._injection_threads):
                        t = threads.pop(seq_id, None)
                        if t and t.is_alive():
                            t.join(timeout=1.0)
                    self._run_coro(self._vllm_llm_cleanup(seq_id))
                    self.model.tts.stream_abort_request(seq_id)
                    self._seq_mgr.remove(seq_id)

                if result.error:
                    responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(result.error)))
                else:
                    responses.append(
                        pb_utils.InferenceResponse(
                            output_tensors=[
                                pb_utils.Tensor("output_text", output_text_arr),
                                pb_utils.Tensor("output_asr_text", output_asr_text_arr),
                                pb_utils.Tensor("output_function_text", output_function_text_arr),
                                pb_utils.Tensor(
                                    "output_audio", result.generated_audio.to(torch.float32).cpu().numpy(),
                                ),
                            ]
                        )
                    )

        return responses

    def _select_tool_ack_message(self, seq_state, sanitized_function_text: str) -> str:
        try:
            calls = json.loads(sanitized_function_text)
            if not isinstance(calls, list) or not calls:
                return ""
            tool_name = calls[0].get("name", "")
            ack_messages = seq_state.tool_ack_messages.get(tool_name, [])
            if not ack_messages:
                return ""
            return random.choice(ack_messages)
        except (json.JSONDecodeError, AttributeError, KeyError, IndexError):
            return ""

    def _build_ack_tokens(self, ack_message: str) -> deque:
        """Frame an acknowledgement as a complete TTS turn."""
        ack_tokens = list(self.model.tokenizer.text_to_ids(ack_message))
        trailing_pad_count = max(17, math.ceil(0.5 * len(ack_message)))
        return deque(
            [self.model.tokenizer.bos_id]
            + ack_tokens
            + [self.pad_id] * trailing_pad_count
            + [self.model.tokenizer.eos_id]
        )

    def _extract_tool_ack_messages(self, system_prompt: str) -> Dict[str, List[str]]:
        """Parse <TOOL_ACK_MESSAGES> from the system prompt and return a tool_name -> ack_messages map."""
        match = _re_tool_ack_messages_pattern.search(system_prompt)
        if not match:
            return {}
        try:
            ack_list = json.loads(match.group(1).strip())
            if not isinstance(ack_list, list):
                return {}
            ack_messages = {}
            for t in ack_list:
                if not isinstance(t, dict) or not t.get("name"):
                    continue
                messages = t.get("ack_messages")
                if not isinstance(messages, list):
                    continue
                messages = [message for message in messages if isinstance(message, str) and message]
                if messages:
                    ack_messages[t["name"]] = messages
            return ack_messages
        except (json.JSONDecodeError, KeyError):
            return {}

    async def _send_system_prompt(self, seq_id, system_prompt):
        """Send the system prompt to the sequence."""
        logger = pb_utils.Logger
        self._seq_mgr[seq_id].tool_ack_messages = self._extract_tool_ack_messages(system_prompt)
        if self._seq_mgr[seq_id].tool_ack_messages:
            logger.log_info(f"Tool ack messages for seq {seq_id}: {self._seq_mgr[seq_id].tool_ack_messages}")
        llm_prompt = _re_tool_ack_messages_pattern.sub("", system_prompt).strip()
        prompt_token_ids, prompt_embedded, prompt_len = self._prepare_system_prompt_embeddings(llm_prompt)
        if prompt_embedded is None:
            logger.log_info(f"Skipping empty system prompt for sequence {seq_id}")
            return None
        logger.log_info(f"Sending system prompt to sequence {seq_id}: {llm_prompt}")
        result = await self._get_vllm_llm_next_token(seq_id, prompt_token_ids, prompt_embedded)
        logger.log_info(f"System prompt sent to sequence {seq_id}")
        return result

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        logger = pb_utils.Logger
        self.model.llm.shutdown()
        self.model.tts.shutdown()

        logger.log_verbose("Cleaning up...")
