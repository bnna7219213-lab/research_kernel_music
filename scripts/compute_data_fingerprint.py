#!/usr/bin/env python3
"""
T2Music — Dataset fingerprint tool.

Produces a reproducible SHA-256 hash over:
  - sorted list of audio files (path + size + mtime)
  - caption / label CSV contents (if present)

This fingerprint is used as a training cache key — if the dataset changes,
the cached features / tokenized outputs become stale.

Usage:
  python compute_data_fingerprint.py \
      --audio-dir /data/t2music/raw/musiccaps \
      --caption-csv /data/t2music/raw/musiccaps/musiccaps.csv \
      --out /data/t2music/_spec/musiccaps.fingerprint.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scan_audio(root: Path) -> list[dict]:
    entries = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        st = p.stat()
        entries.append(
            {
                "path": str(p.relative_to(root)),
                "size": st.st_size,
                "mtime": int(st.st_mtime),
            }
        )
    return entries


def hash_csv(csv_path: Optional[Path]) -> str:
    if csv_path is None or not csv_path.exists():
        return sha256("")
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute dataset fingerprint")
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--caption-csv", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--name", default="t2music-dataset")
    args = parser.parse_args()

    if not args.audio_dir.exists():
        print(f"[ERROR] audio_dir not found: {args.audio_dir}", file=sys.stderr)
        return 1

    entries = scan_audio(args.audio_dir)
    manifest_str = json.dumps(entries, sort_keys=True).encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_str).hexdigest()

    csv_hash = hash_csv(args.caption_csv)

    combined = sha256(manifest_hash + csv_hash)

    fingerprint = {
        "name": args.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audio_dir": str(args.audio_dir.resolve()),
        "caption_csv": str(args.caption_csv.resolve()) if args.caption_csv else None,
        "num_files": len(entries),
        "total_gb": round(sum(e["size"] for e in entries) / (1024**3), 3),
        "manifest_sha256": manifest_hash,
        "csv_sha256": csv_hash,
        "dataset_fingerprint": combined,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(fingerprint, f, indent=2, ensure_ascii=False)

    print(json.dumps(fingerprint, indent=2, ensure_ascii=False))
    print(f"\n[FINGERPRINT] {combined}")
    print(f"[OUT] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
