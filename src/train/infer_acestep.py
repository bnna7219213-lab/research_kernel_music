#!/usr/bin/env python3
"""
ACE-Step v1 inference entrypoint.

Usage:
  python src/train/infer_acestep.py \
      --prompts_file /data/t2music/eval/prompts.txt \
      --out_dir /data/t2music/outputs/acestep \
      --weights /data/t2music/pretrained/acestep_v1 \
      --num_steps 50 --guidance_scale 3.5 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.prompts_file.exists():
        print(f"[ERROR] prompts not found: : {args.prompts_file}")
        return 1

    prompts = [l for l in args.prompts_file.read_text().splitlines() if l.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[infer-acestep] weights: : {args.weights}")
    print(f"[infer-acestep] out_dir: : {args.out_dir}")
    print(f"[infer-acestep] prompts: : {len(prompts)}")

    print("[infer-acestep] ⚠  Skeleton — plug in ACE-Step reference implementation.")
    manifest = []
    for i, prompt in enumerate(prompts[:5]):
        out_path = args.out_dir / f"{i:04d}.wav"
        manifest.append({"prompt": prompt, "audio": str(out_path), "status": "skipped_skeleton"})

    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
