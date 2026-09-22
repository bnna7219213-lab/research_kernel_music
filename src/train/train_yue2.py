#!/usr/bin/env python3
"""
YuE2-3B fine-tuning loop (Phase A).

Adapter over the  m-a-p / YuE reference implementation.
Since YuE2 is AR+LFM  with dual-track decoupling, Phase A focuses on
short-clip, single-track text-to-music fine-tuning on MusicCaps.

Typical 8×A100-80GB run (frozen backbone + LoRA-free for memory safety):
  batch_size_per_gpu=4 × grad_accum=2 × 8 gpus = 64 effective
  Sequence: 10s @ 12.5Hz codec = ~125 tokens — very comfortable

Usage (inside docker):
  torchrun --nproc_per_node=8 src/train/main.py --config configs/yue2_3b.yaml
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
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None

        self._init_dataset()
        self._init_model()
        self._init_opt()

    # ---------------- dataset ----------------
    def _init_dataset(self) -> None:
        self.train_loader = build_dataloaders(
            self.cfg.data, self.dist_cfg, split="train"
        )
        if hasattr(self.cfg.data, "val") and self.cfg.data.val:
            self.val_loader = build_dataloaders(
                self.cfg.data, self.dist_cfg, split="val"
            )

    # ---------------- model ----------------
    def _init_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        pretrained = self.cfg.model.pretrained
        log.info(f"[YuE2] loading model from: : {pretrained}")

        # Phase A: we load the reference LLM backbond (LLaMA2 7B-based)
        # and add a small music head. For skeleton we use transformers
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        # Phase A options:
        #   freeze_backbone=true  → only finetune the music head
        if self.cfg.model.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            log.info("[YuE2] backbone frozen, only music head trainable.")

        # DCAE codec for encoding audio -> tokens
        # (in practice this is from m-a-p's inference/sampling code)
        self.codec = self._load_codec(pretrained)
        self.codec.eval()

        # FSDP wrap the trainable part(s)
        self.backbone = wrap_fsdp(self.backbone, self.dist_cfg)

    def _load_codec(self, pretrained: str):
        """Reference codec: m-a-p codec from YuE repo."""
        try:
            from yue.codec import Codec  # m-a-p repo
            return Codec.from_pretrained(pretrained).cuda()
        except Exception as e:
            log.warning(f"[YuE2] could not load m-a-p codec: : {e}")
            return None

    # ---------------- optimizer ----------------
    def _init_opt(self) -> None:
        trainable = [p for p in self.backbone.parameters() if p.requires_grad]
        opt_cfg = self.cfg.training.adam
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(self.cfg.training.max_steps),
            eta_min=opt_cfg.lr * 0.01,
        )

    # ---------------- step ----------------
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.backbone.train()
        captions = batch["captions"]
        audios = batch["audio"].cuda(non_blocking=True)

        t0 = time.time()

        # Encode audio with codec
        if self.codec is not None:
            with torch.no_grad():
                audio_tokens = self.codec.encode(audios)
        else:
            # Fallback: passthrough
            audio_tokens = audios

        # LLM forward: tokens -> next-token / audio logits
        # Simplified — implement m-a-p's loss in real run
        outputs = self.backbone(
            input_ids=audio_tokens.long() if audio_tokens.dtype == torch.long else None,
        )
        loss = outputs.loss if hasattr(outputs, "loss") else outputs

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.backbone.parameters(),
            self.cfg.training.max_grad_norm,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        dt = time.time() - t0
        return {
            "loss": loss.detach().item(),
            "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            "lr": self.scheduler.get_last_lr()[0],
            "step_time": dt,
            "throughput": audios.shape[0] / dt,
        }

    # ---------------- eval ----------------
    @torch.no_grad()
    def evaluate(self, step: int) -> Dict[str, float]:
        self.backbone.eval()
        # Build a small set of demo generations for logging
        return {"eval_step": step, "placeholder": 0.0}

    # ---------------- loop ----------------
    def run(self) -> None:
        max_steps = int(self.cfg.training.max_steps)
        ckpt_every = int(self.cfg.training.checkpoint_every)
        log_every = int(self.cfg.training.log_every)
        eval_every = int(self.cfg.training.eval_every)

        log.info(f"[YuE2] training loop starting — max_steps={max_steps}")
        for step in range(self.global_step + 1, max_steps + 1):
            self.global_step = step
            batch = next(iter(self.train_loader))

            metrics = self.train_step(batch)

            if step % log_every == 0 and is_main_process():
                log_metrics(metrics, step=step)
                log.info(f"step={step} loss={metrics['loss']:.4f} grad_norm={metrics['grad_norm']:.3f}")

            if step % eval_every == 0 and is_main_process():
                eval_metrics = self.evaluate(step)
                log_metrics(eval_metrics, step=step)

            if step % ckpt_every == 0:
                self.save_checkpoint(step)

        self.save_checkpoint(self.global_step)
        log.info("[YuE2] training loop completed.")
        log_finish()

    def save_checkpoint(self, step: int) -> None:
        out = Path(self.cfg.output_dir) / f"step-{step}"
        out.mkdir(parents=True, exist_ok=True)
        if is_main_process():
            # Save only the delta of trainable params to save disk
            state_dict = {
                k: v for k, v in self.backbone.state_dict().items() if v.requires_grad
            }
            torch.save(state_dict, out / "model.pt")
            # Save config for reproducibility
            with open(out / "config.json", "w") as f:
                json.dump(OmegaConf.to_container(self.cfg, resolve=True), f, indent=2, default=str)
            log.info(f"[YuE2] checkpoint saved: {out}")


def train_yue2(cfg: DictConfig) -> None:
    dist_cfg = setup_distributed(cfg.distributed)
    setup_logger(cfg, dist_cfg)
    trainer = Trainer(cfg, dist_cfg)
    trainer.run()
