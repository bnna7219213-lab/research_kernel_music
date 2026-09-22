"""Phase-A smoke tests — runs on a 开发机 (CPU-only welcome).

Covers:
  - config YAML parsing
  - data fingerprint CLI
  - dataset fingerprint file integrity
  - blind-listening playlist generation
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def test_configs_parseable():
    cfg_dir = ROOT / "configs"
    for path in cfg_dir.glob("*.yaml"):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert "name" in cfg
        assert "model" in cfg
        assert "data" in cfg
        assert "training" in cfg
        print(f"[OK] config parseable: {path.name}")


def test_configs_have_required_fields():
    cfg_dir = ROOT / "configs"
    required = ["name", "model.name", "data.train", "training.max_steps", "distributed", "logging"]
    for path in cfg_dir.glob("*.yaml"):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        for dotted in required:
            parts = dotted.split(".")
            node = cfg
            for p in parts:
                assert p in node, f"missing field {dotted} in {path.name}"
                node = node[p]
        print(f"[OK] required fields present: {path.name}")


def test_fingerprint_runs():
    tmp = Path(tempfile.mkdtemp())
    audio_dir = tmp / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        (audio_dir / f"sample_{i}.wav").write_bytes(b"RIFF" + b"\x00" * 100)

    out = tmp / "fp.json"
    r = subprocess.run(
        [sys.executable, "scripts/compute_data_fingerprint.py",
         "--audio-dir", str(audio_dir), "--out", str(out), "--name", "smoke"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert r.returncode == 0, r.stderr
    fp = json.loads(out.read_text())
    assert "dataset_fingerprint" in fp
    assert fp["num_files"] == 5
    print("[OK] fingerprint CLI works")


def test_blind_listening_playlist():
    tmp = Path(tempfile.mkdtemp())
    sys_a = tmp / "sysA"
    sys_b = tmp / "sysB"
    sys_a.mkdir()
    sys_b.mkdir()
    for i in range(3):
        for d in [sys_a, sys_b]:
            (d / f"sample_{i}.wav").write_bytes(b"RIFF" + b"\x00" * 64)

    out = tmp / "playlist.csv"
    r = subprocess.run(
        [sys.executable, "src/eval/blind_listening/generate_playlist.py",
         "--systems", json.dumps({"A": str(sys_a), "B": str(sys_b)}),
         "--out", str(out), "--num_pairs", "2", "--seed", "0"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert r.returncode == 0, r.stderr
    lines = out.read_text().splitlines()
    assert len(lines) >= 3, f"expected header + 2 pairs, got {len(lines)}: {out.read_text()}"
    print(f"[OK] blind-listening playlist ({len(lines)-1} pairs)")


if __name__ == "__main__":
    test_configs_parseable()
    test_configs_have_required_fields()
    test_fingerprint_runs()
    test_blind_listening_playlist()
    print("\n[ALL SMOKE TESTS PASSED]")
