#!/usr/bin/env python3
"""
ACE-Step v1 fine-tuning loop (Phase A).

Adapter over  ACE-Step  reference implementation (Apache 2.0):
  - Stage A: DCAE codec training (mel reconstruction + adversarial + feature matching)
  - Stage B: DiT + flow matching rendering with REPA semantic alignment

Phase A focuses on Stage A (codec) + Stage B (renderer) with 30-60s short clips.

Usage:
  torchrun --nproc_per_node=8 src/train/main.py --config configs/acestep_v1.yaml
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.distributed import setup_distributed, wrap_fsdp, DistConfig, is_main_process
from src.utils.logger import setup_logger, log_metrics, log_audio, finish as log_finish
from src.train.data import build_dataloaders

log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: DictConfig, dist_cfg: DistConfig):
        self.cfg = cfg
        self.dist_cfg = dist_cfg
        self.global_step = 0
        self.stage = getattr(cfg.training, "stage", "A")

        self._init_dataset()
        self._init_models()
        self._init_opt()

    def _init_dataset(self) -> None:
        self.train_loader = build_dataloaders(self.cfg.data, self.dist_cfg, split="train")
        if hasattr(self.cfg.data, "val"):
            try:
                self.val_loader = build_dataloaders(self.cfg.data, self.dist_cfg, split="val")
            except Exception:
                self.val_loader = None

    def _init_models(self) -> None:
        if self.stage == "A":
            self._init_codec()
        else:
            self._init_renderer()

    def _init_codec(self) -> None:
        """Load DCAE encoder + HiFiGAN vocoder (ACE-Step v1 design)."""
        pretrained = self.cfg.model.pretrained
        log.info(f"[ACE-Step] loading codec from: : {pretrained}")
        # Reference: from acestep.models import DCAE, HiFiGAN
        self.codec = None   # plug in ACE-Step implementation
        self.vocoder = None
        log.info("[ACE-Step] codec initialized (skeleton).")

    def _init_renderer(self) -> None:
        """Load DiT renderer + text encoder."""
        pretrained = self.cfg.model.pretrained
        log.info(f"[ACE-Step] loading DiT renderer from: : {pretrained}")
        # Reference: from acestep.models import MusicDiT, TextEncoder
        self.dit = None
        self.text_encoder = None
        log.info("[ACE-Step] renderer initialized (skeleton).")

    def _init_opt(self) -> None:
        trainable = []
        if self.stage == "A":
            trainable = self._collect_trainable([self.codec, self.vocoder])
        else:
            trainable = self._collect_trainable([self.dit])
        opt_cfg = self.cfg.training.adam
        self.optimizer = torch.optim.AdamW(
            trainable, lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas), eps=opt_cfg.eps, weight_decay=opt_cfg.weight_decay,
        )
        max_steps = int(self.cfg.training.max_steps.get(self.stage, 10000))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=opt_cfg.lr * 0.01,
        )

    def _collect_trainable(self, modules):
        out = []
        for m in modules:
            if m is not None:
                out.extend([p for p in m.parameters() if p.requires_grad])
        return out

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        t0 = time.time()
        audios = batch["audio"].cuda(non_blocking=True)
        loss = torch.tensor(0.0, device=audios.device)
        # Placeholder — replace with DCAE mel-loss or DiT flow-matching loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in self._collect_trainable([]) if False] or [torch.zeros(1, requires_grad=True)], self.cfg.training.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return {
            "loss": loss.item(), "grad_norm": 0.0,
            "lr": self.scheduler.get_last_lr()[0],
            "throughput": audios.shape[0] / (time.time() - t0 + 1e-6),
            "step_time": time.time() - t0,
        }

    @torch.no_grad()
    def evaluate(self, step: int) -> Dict[str, float]:
        return {"eval_step": step}

    def run(self) -> None:
        max_steps = int(self.cfg.training.max_steps.get(self.stage, 10000))
        log.info(f"[ACE-Step] training loop starting — stage={self.stage} max_steps={max_steps}")
        ckpt_every = int(self.cfg.training.checkpoint_every)
        log_every = int(self.cfg.training.log_every)
        step = self.global_step
        while step < max_steps:
            step += 1
            self.global_step = step
            for batch in self.train_loader:
                metrics = self.train_step(batch)
                if step % log_every == 0 and is_main_process():
                    log_metrics(metrics, step=step)
                if step >= max_steps:
                    break
            if step % ckpt_every == 0:
                self.save_checkpoint(step)
        self.save_checkpoint(self.global_step)
        log.info("[ACE-Step] training finished.")
        log_finish()

    def save_checkpoint(self, step: int) -> None:
        if not is_main_process():
            return
        out = Path(self.cfg.output_dir) / f"{self.stage}_step-{step}"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out.with_suffix(".config.json"), "w") as f:
            json.dump(OmegaConf.to_container(self.cfg, resolve=True), f, indent=2, default=str)
        log.info(f"[ACE-Step] checkpoint config saved: {out.with_suffix('.config.json')}")


def train_acestep(cfg: DictConfig) -> None:
    dist_cfg = setup_distributed(cfg.distributed)
    setup_logger(cfg, dist_cfg)
    trainer = Trainer(cfg, dist_cfg)
    trainer.run()
