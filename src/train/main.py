#!/usr/bin/env python3
"""
T2Music — unified training entrypoint.

Dispatches to the selected model's training loop based on `model.name`.
Phase A supports: yue2_3b, acestep_v1.

Usage:
  torchrun --nproc_per_node=8 src/train/main.py --config configs/yue2_3b.yaml
  torchrun --nproc_per_node=8 src/train/main.py --config configs/acestep_v1.yaml
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf, DictConfig

# local
from src.utils.logger import setup_logger
from src.utils.distributed import setup_distributed, is_main_process


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T2Music training")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML config",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf overrides, e.g. training.max_steps=5000",
    )
    return parser.parse_args()


def load_config(path: Path, overrides: list[str]) -> DictConfig:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg.merge_with_dotlist(overrides)
    return cfg


def prepare_dataset_fingerprint(cfg: DictConfig) -> dict:
    """Verify dataset fingerprint against cached lock (reproducibility)."""
    from src.train.data import check_fingerprint
    try:
        return check_fingerprint(cfg.data.dataset_fingerprint)
    except FileNotFoundError as e:
        print(f"[WARN] {e}", file=sys.stderr)
        return {"checked": False}


def train(cfg: DictConfig) -> None:
    model_name = cfg.model.name

    if model_name == "yue2_3b":
        from src.train.train_yue2 import train_yue2 as train_fn
    elif model_name == "acestep_v1":
        from src.train.train_acestep import train_acestep as train_fn
    else:
        raise ValueError(f"Unknown model: {model_name}")

    train_fn(cfg)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    # distributed init
    dist_cfg = setup_distributed(cfg.distributed)
    setup_logger(cfg, dist_cfg)

    # reproducibility seed
    from src.utils.seed import set_seed
    set_seed(cfg.seed, cfg.reproducibility.deterministic_algorithms)

    if is_main_process():
        print(OmegaConf.to_yaml(cfg))

    # dataset fingerprint
    fingerprint = prepare_dataset_fingerprint(cfg)
    if is_main_process():
        print(f"[DATA] fingerprint check: {fingerprint}")

    # train
    train(cfg)

    # cleanup
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
