"""Streaming SVS renderer — RUNS IN THE SOULX-SINGER VENV.

Same generation as SoulX's cli.inference (model load, per-segment loop, score control)
but writes each segment's wav the moment it is rendered, so the agent can start
playback after segment 1 (~9s gen for ~13s audio = faster than realtime) while later
segments render behind it. Files in save_dir:

    segment_00.wav, segment_01.wav, ...   (appear as rendered; atomic via tmp+rename)
    generated.wav                          (full merged song, written last — cache file)
    ALL_DONE                               (marker: success)

Usage mirrors cli.inference:
    python sing_render_stream.py --prompt_wav_path .. --prompt_metadata_path ..
        --target_metadata_path .. --save_dir .. [--model_path ..] [--config ..]
        [--phoneset_path ..]
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

SING_DIR = os.path.expanduser(os.environ.get("SING_DIR", "~/soulx-singer"))
sys.path.insert(0, SING_DIR)

from cli.inference import build_model, load_config          # noqa: E402
from soulxsinger.utils.data_processor import DataProcessor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=os.path.join(SING_DIR, "pretrained_models/SoulX-Singer/model.pt"))
    ap.add_argument("--config", default=os.path.join(SING_DIR, "soulxsinger/config/soulxsinger.yaml"))
    ap.add_argument("--phoneset_path", default=os.path.join(SING_DIR, "soulxsinger/utils/phoneme/phone_set.json"))
    ap.add_argument("--prompt_wav_path", required=True)
    ap.add_argument("--prompt_metadata_path", required=True)
    ap.add_argument("--target_metadata_path", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    config = load_config(args.config)
    model = build_model(model_path=args.model_path, config=config,
                        device=args.device, use_fp16=False)
    dp = DataProcessor(hop_size=config.audio.hop_size, sample_rate=config.audio.sample_rate,
                       phoneset_path=args.phoneset_path, device=args.device)

    prompt_meta = json.load(open(args.prompt_metadata_path))[0]
    targets = json.load(open(args.target_metadata_path))
    prompt_data = dp.process(prompt_meta, args.prompt_wav_path)

    sr = config.audio.sample_rate
    total_len = int(targets[-1]["time"][1] / 1000 * sr)
    merged = np.zeros(total_len, dtype=np.float32)

    for idx, target_meta in enumerate(targets):
        start = int(target_meta["time"][0] / 1000 * sr)
        target_data = dp.process(target_meta, None)
        with torch.no_grad():
            audio = model.infer({"prompt": prompt_data, "target": target_data},
                                auto_shift=True, pitch_shift=0,
                                n_steps=config.infer.n_steps, cfg=config.infer.cfg,
                                control="score", use_fp16=False)
        audio = audio.squeeze().cpu().numpy().astype(np.float32)
        n = min(len(audio), total_len - start)
        merged[start:start + n] = audio[:n]
        seg_path = os.path.join(args.save_dir, f"segment_{idx:02d}.wav")
        sf.write(seg_path + ".tmp.wav", audio, sr)
        os.replace(seg_path + ".tmp.wav", seg_path)      # atomic: readers never see partials
        print(f"SEGMENT {idx} ready ({len(audio)/sr:.1f}s)", flush=True)

    sf.write(os.path.join(args.save_dir, "generated.wav"), merged, sr)
    open(os.path.join(args.save_dir, "ALL_DONE"), "w").write("ok")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
