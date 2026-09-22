#!/usr/bin/env python3
"""
YuE2-3B inference entrypoint.

Usage:
  python src/train/infer_yue2.py \
      --prompts_file /data/t2music/eval/prompts.txt \
      --out_dir /data/t2music/outputs/yue2 \
      --weights /data/t2music/pretrained/yue2_3b \
      --num_steps 50 --guidance_scale 4.5 --seed 42 \
      --num_samples 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    if not args.prompts_file.exists():
        print(f"[ERROR] prompts file not found: {args.prompts_file}")
        return 1

    prompts = [l.strip() for l in args.prompts_file.read_text().splitlines() if l.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[infer-yue2] weights: : {args.prompts_file}")
    print(f"[infer-yue2] out_dir: : {args.out_dir}")
    print(f"[infer-yue2] prompts: {len(prompts)}")
    print(f"[infer-yue2] seed: : {args.seed}")

    # Placeholder: load m-a-p/YuE reference and batch-generate
    # In real run, call m-a-p wrapper with torch.no_grad()
    print("[infer-yue2] ⚠  Skeleton — plug in m-a-p/YuE reference implementation.")
    manifest = []
    for i, prompt in enumerate(prompts[:5]):
        out_path = args.out_dir / f"{i:04d}.wav"
        manifest.append({"prompt": prompt, "audio": str(out_path), "status": "skipped_skeleton"})

    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[infer-yue2] manifest saved: {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
