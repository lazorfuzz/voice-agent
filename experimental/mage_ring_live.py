#!/usr/bin/env python
"""Experimental Ring WebRTC -> Mage-VL running-commentary probe.

This is intentionally separate from the production voice worker. It negotiates a
server-side Ring live view with aiortc, samples a small number of decoded frames
per segment, and feeds each rolling segment to Mage-VL. Results are printed as
JSON lines so the probe can later be wrapped by a service or consumed by the
voice agent without moving the model into the latency-sensitive agent process.

The default path uses frame-sampled video. Mage's released proactive gate was
trained on codec features, so this proves the end-to-end live path but is not a
calibrated evaluation of the gate. See experimental/MAGE_RING.md for the codec
follow-up and setup commands.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agent"
sys.path.insert(0, str(AGENT_DIR))

# A stale user-level token should not make a public checkpoint look private.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(AGENT_DIR / ".env.local")

import ring_tools  # noqa: E402
from aiortc import (  # noqa: E402
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCRtpReceiver,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError  # noqa: E402
from aiortc.sdp import candidate_from_sdp  # noqa: E402


LOG = logging.getLogger("mage_ring")
MODEL_ID = "microsoft/Mage-VL"
# Pin the custom code reviewed for this experiment. Override with --revision.
MODEL_REVISION = "5c78cab61938e73859b63724d9bf5cb88c477eaa"
COMMENTARY_PROMPT = (
    "You are giving natural live commentary on a home security camera. Report only "
    "a meaningful new development, such as an arrival, departure, new gesture, fall, "
    "or object being moved. Give one short, spoken-ready sentence, refer to the same "
    "person with a pronoun when it is clear, and do not inventory furniture. Never "
    "comment on whether the camera itself is moving. "
)


def emit(kind: str, **fields: Any) -> None:
    print(json.dumps({"type": kind, **fields}, ensure_ascii=False), flush=True)


@dataclass
class SegmentResult:
    probability: float | None
    text: str | None
    inference_seconds: float


class MageEngine:
    """Mage model and rolling gate state; call all methods from one worker thread."""

    def __init__(
        self,
        model_id: str,
        revision: str | None,
        device: str,
        max_new_tokens: int,
        history_segments: int,
        max_pixels: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        started = time.monotonic()
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if revision:
            kwargs["revision"] = revision
        self.torch = torch
        self.device = device
        self.dtype = torch.bfloat16
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        # The checkpoint's generic frame processor defaults to four million
        # pixels per frame, which is far too expensive for live commentary.
        # 150k matches Microsoft's codec-streaming example.
        self.processor.video_processor.max_pixels = max_pixels
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=self.dtype,
            attn_implementation="eager" if device == "mps" else "sdpa",
            **kwargs,
        ).to(device).eval()
        self.history: deque[dict[str, Any]] = deque(maxlen=history_segments)
        self.previous_commentary: str | None = None
        self.load_seconds = time.monotonic() - started

    def _resize_frames(self, frames: list[Any]) -> list[Any]:
        """Apply the pixel budget that Mage skips for already-decoded PIL frames."""
        import math
        from PIL import Image

        factor = 32  # 16px patch * 2x2 spatial merge
        resized = []
        for frame in frames:
            width, height = frame.size
            if width * height <= self.max_pixels:
                resized.append(frame)
                continue
            scale = math.sqrt(self.max_pixels / float(width * height))
            target_w = max(factor, int(width * scale) // factor * factor)
            target_h = max(factor, int(height * scale) // factor * factor)
            resized.append(frame.resize((target_w, target_h), Image.Resampling.BILINEAR))
        return resized

    def _to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        moved = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                moved[key] = value
            elif key in ("pixel_values", "pixel_values_videos"):
                moved[key] = value.to(device=self.device, dtype=self.dtype)
            else:
                moved[key] = value.to(self.device)
        return moved

    def _prepare(self, frames: list[Any]) -> dict[str, Any]:
        frames = self._resize_frames(frames)
        if self.previous_commentary:
            context = (
                "Previous spoken update: "
                + json.dumps(self.previous_commentary, ensure_ascii=False)
                + ". Do not repeat or rephrase that action merely because it continues. "
                "If there is no meaningful new development, return exactly SILENT."
            )
        else:
            context = (
                "This is the first update. Say whether anyone is present and what they are "
                "doing. If nobody is present, say the area is empty and quiet. Do not return SILENT."
            )
        messages = [{
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": COMMENTARY_PROMPT + context},
            ],
        }]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt],
            videos=[frames],
            return_tensors="pt",
            padding=False,
        )
        return self._to_device(dict(inputs))

    @staticmethod
    def _gate_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: inputs[key]
            for key in ("pixel_values", "image_grid_thw", "patch_positions")
            if key in inputs
        }

    def analyze(
        self,
        frames: list[Any],
        gate_threshold: float,
        skip_gate: bool,
    ) -> SegmentResult:
        started = time.monotonic()
        inputs = self._prepare(frames)
        required = {"pixel_values", "image_grid_thw"}
        if not required.issubset(inputs):
            raise RuntimeError(
                f"Mage processor did not return gate inputs; got {sorted(inputs)}"
            )
        probability = None
        should_speak = True
        if not skip_gate:
            self.history.append(inputs)
            visual_segments = [self._gate_inputs(item) for item in self.history]
            with self.torch.inference_mode():
                logits = self.model.streammind_gate_forward_segments(visual_segments)[0]
            last_index = sum(
                int(item["image_grid_thw"][:, 0].sum())
                for item in visual_segments
            ) - 1
            probability = float(
                self.torch.softmax(logits[last_index].float(), dim=-1)[1].item()
            )
            should_speak = probability >= gate_threshold

        text = None
        if should_speak:
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            new_tokens = output[0, inputs["input_ids"].shape[1]:]
            text = self.processor.tokenizer.decode(
                new_tokens, skip_special_tokens=True,
            ).strip()
            if text.strip(" <>[].").casefold() == "silent":
                text = None
            elif text:
                self.previous_commentary = text
        return SegmentResult(
            probability=probability,
            text=text or None,
            inference_seconds=time.monotonic() - started,
        )


class RingReceiver:
    """Negotiate Ring WebRTC locally and expose its decoded video track."""

    def __init__(self, camera_name: str) -> None:
        self.camera_name = camera_name
        self.ring = None
        self.camera = None
        self.camera_label = camera_name
        self.pc: RTCPeerConnection | None = None
        self.session_id: str | None = None
        self.keepalive_task: asyncio.Task | None = None
        self.callback_tasks: set[asyncio.Task] = set()

    async def connect(self):
        ring = await ring_tools._session()
        self.ring = ring
        camera = await ring_tools.resolve(self.camera_name)
        if camera is None:
            names = await ring_tools.camera_names()
            raise RuntimeError(
                f"Ring camera {self.camera_name!r} not found; available: {names}"
            )
        self.camera = camera
        self.camera_label = camera.name

        ice_servers = [RTCIceServer(urls=camera.get_ice_servers())]
        pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
        self.pc = pc
        video = pc.addTransceiver("video", direction="recvonly")
        h264 = [
            codec for codec in RTCRtpReceiver.getCapabilities("video").codecs
            if codec.mimeType.lower() == "video/h264"
        ]
        if h264:
            video.setCodecPreferences(h264)

        loop = asyncio.get_running_loop()
        track_ready = loop.create_future()
        answer_ready = loop.create_future()

        @pc.on("track")
        def on_track(track):
            if track.kind == "video" and not track_ready.done():
                track_ready.set_result(track)

        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            LOG.info("Ring peer connection: %s", pc.connectionState)
            if pc.connectionState == "failed" and not track_ready.done():
                track_ready.set_exception(RuntimeError("Ring WebRTC connection failed"))

        async def apply_ring_message(message) -> None:
            try:
                if message.answer is not None:
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=message.answer, type="answer")
                    )
                    if not answer_ready.done():
                        answer_ready.set_result(True)
                if message.candidate is not None:
                    candidate = candidate_from_sdp(message.candidate)
                    candidate.sdpMLineIndex = int(message.sdp_m_line_index or 0)
                    await pc.addIceCandidate(candidate)
                if message.error_message and not track_ready.done():
                    track_ready.set_exception(RuntimeError(message.error_message))
            except Exception as error:
                if not track_ready.done():
                    track_ready.set_exception(error)

        def on_ring_message(message) -> None:
            task = asyncio.create_task(apply_ring_message(message))
            self.callback_tasks.add(task)
            task.add_done_callback(self.callback_tasks.discard)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        if pc.localDescription is None:
            raise RuntimeError("aiortc did not produce a local offer")

        # ring-doorbell keys the session independently of the SDP origin value.
        import uuid
        self.session_id = uuid.uuid4().hex
        await camera.generate_async_webrtc_stream(
            pc.localDescription.sdp,
            self.session_id,
            on_ring_message,
            keep_alive_timeout=60 * 5,
        )
        await asyncio.wait_for(answer_ready, timeout=20)
        track = await asyncio.wait_for(track_ready, timeout=30)
        self.keepalive_task = asyncio.create_task(self._keepalive())
        return track

    async def _keepalive(self) -> None:
        assert self.camera is not None and self.session_id is not None
        while True:
            await asyncio.sleep(20)
            await self.camera.keep_alive_webrtc_stream(self.session_id)

    async def close(self, *, close_auth: bool = True) -> None:
        if self.keepalive_task:
            self.keepalive_task.cancel()
        if self.camera is not None and self.session_id is not None:
            try:
                await self.camera.close_webrtc_stream(self.session_id)
            except Exception:
                LOG.debug("Ring stream close failed", exc_info=True)
        if self.pc is not None:
            await self.pc.close()
        for task in list(self.callback_tasks):
            task.cancel()
        if close_auth and self.ring is not None:
            await self.ring.auth.async_close()


async def capture_segment(track, seconds: float, sample_fps: float) -> list[Any]:
    """Receive continuously but retain only the requested number of PIL frames."""
    interval = 1.0 / sample_fps
    frames = []
    first = await asyncio.wait_for(track.recv(), timeout=20)
    frames.append(first.to_image())
    started = time.monotonic()
    next_sample = started + interval
    while time.monotonic() - started < seconds:
        frame = await asyncio.wait_for(track.recv(), timeout=10)
        now = time.monotonic()
        if now >= next_sample:
            frames.append(frame.to_image())
            while next_sample <= now:
                next_sample += interval
    return frames


async def timed_capture_segment(track, seconds: float, sample_fps: float):
    """Capture a segment and retain the wall-clock time at which it began."""
    captured_at = time.time()
    frames = await capture_segment(track, seconds, sample_fps)
    return captured_at, frames


async def run(args: argparse.Namespace) -> None:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mage-vl")
    loop = asyncio.get_running_loop()
    engine = None
    capture_task: asyncio.Task | None = None
    receiver = RingReceiver(args.camera)
    try:
        if not args.capture_only:
            engine = await loop.run_in_executor(
                executor,
                lambda: MageEngine(
                    model_id=args.model,
                    revision=args.revision or None,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                    history_segments=args.history_segments,
                    max_pixels=args.max_pixels,
                ),
            )
            emit("model_ready", model=args.model, device=args.device,
                 load_seconds=round(engine.load_seconds, 3))

        track = await receiver.connect()
        emit("camera_ready", camera=receiver.camera_label)

        capture_task = asyncio.create_task(
            timed_capture_segment(track, args.segment_seconds, args.sample_fps)
        )
        active_deadline = (
            time.monotonic() + args.run_seconds if args.run_seconds else None
        )
        index = 0
        reconnects = 0
        while index < args.max_segments and (
            active_deadline is None or time.monotonic() < active_deadline
        ):
            while True:
                try:
                    captured, frames = await capture_task
                    break
                except (TimeoutError, MediaStreamError) as error:
                    if reconnects >= args.max_reconnects:
                        raise
                    reconnects += 1
                    emit(
                        "camera_stalled",
                        camera=receiver.camera_label,
                        reconnect=reconnects,
                        error=type(error).__name__,
                    )
                    await receiver.close(close_auth=False)
                    receiver = RingReceiver(args.camera)
                    track = await receiver.connect()
                    emit(
                        "camera_ready",
                        camera=receiver.camera_label,
                        reconnected=True,
                    )
                    capture_task = asyncio.create_task(
                        timed_capture_segment(
                            track, args.segment_seconds, args.sample_fps,
                        )
                    )
            # Keep draining/decoding Ring while Mage works. Without this overlap,
            # decoded frames queue behind inference and later commentary drifts away
            # from the live edge.
            if index + 1 < args.max_segments:
                capture_task = asyncio.create_task(
                    timed_capture_segment(track, args.segment_seconds, args.sample_fps)
                )
            if args.capture_only:
                emit("segment", index=index, frames=len(frames),
                     captured_at=captured, decision="captured")
                index += 1
                continue
            assert engine is not None
            result = await loop.run_in_executor(
                executor,
                lambda frames=frames: engine.analyze(
                    frames,
                    gate_threshold=args.gate_threshold,
                    skip_gate=args.skip_gate,
                ),
            )
            emit(
                "commentary",
                index=index,
                camera=receiver.camera_label,
                frames=len(frames),
                captured_at=captured,
                gate_probability=(
                    round(result.probability, 4)
                    if result.probability is not None else None
                ),
                decision="response" if result.text else "silence",
                inference_seconds=round(result.inference_seconds, 3),
                text=result.text,
            )
            index += 1
    finally:
        if capture_task is not None:
            if not capture_task.done():
                capture_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await capture_task
        await receiver.close()
        executor.shutdown(wait=False, cancel_futures=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="bedroom")
    parser.add_argument("--segment-seconds", type=float, default=8.0)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument("--run-seconds", type=float, default=0.0,
                        help="Active camera time after WebRTC is ready; 0 uses max-segments only.")
    parser.add_argument("--max-reconnects", type=int, default=2,
                        help="Reconnect Ring after a frame-delivery stall.")
    parser.add_argument("--history-segments", type=int, default=4)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=150_000,
                        help="Per-frame resize budget; Microsoft's streaming default is 150k.")
    parser.add_argument("--skip-gate", action="store_true",
                        help="Generate on every segment; useful to isolate gate issues.")
    parser.add_argument("--capture-only", action="store_true",
                        help="Verify Ring WebRTC/frame sampling without loading Mage.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.segment_seconds <= 0 or args.sample_fps <= 0 or args.max_pixels <= 0:
        parser.error("segment seconds and sample FPS must be positive")
    if args.max_segments <= 0 or args.history_segments <= 0:
        parser.error("segment counts must be positive")
    if args.run_seconds < 0 or args.max_reconnects < 0:
        parser.error("run seconds and max reconnects cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # ring-doorbell logs SDP tickets and session tokens at DEBUG; never enable it here.
    logging.getLogger("ring_doorbell.webrtcstream").setLevel(logging.INFO)
    # ICE INFO logs enumerate private/public candidate addresses; keep probe output clean.
    logging.getLogger("aioice").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        emit("stopped", reason="keyboard_interrupt")
    except Exception as error:
        LOG.exception("Mage Ring probe failed")
        emit("error", error=f"{type(error).__name__}: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
