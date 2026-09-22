"""
Distributed training utilities (FSDP-2 on 8×A100).

Wraps torch.distributed + FSDP-2 setup so model code stays rank-agnostic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class DistConfig:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "nccl"
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"
    use_fsdp: bool = True
    sharding: str = "full_shard"  # full_shard | shard_grad_op | no_shard
    cpu_offload: bool = False
    mixed_precision: str = "bf16"  # bf16 | fp16 | fp32
    activation_checkpoint: bool = True
    compile_model: bool = True


def setup_distributed(cfg) -> DistConfig:
    """Initialize process group + return resolved DistConfig."""

    dist_cfg = DistConfig(
        backend=getattr(cfg, "backend", "nccl"),
        use_fsdp=getattr(cfg, "fsdp", True),
        sharding=getattr(cfg, "fsdp_sharding", "full_shard"),
        cpu_offload=getattr(cfg, "fsdp_cpu_offload", False),
        mixed_precision=getattr(cfg, "fsdp_mixed_precision", "bf16"),
        activation_checkpoint=getattr(cfg, "activation_checkpointing", True),
        compile_model=getattr(cfg, "compile", True),
    )

    if "WORLD_SIZE" in os.environ:
        # torchrun / torch.distributed.launch sets these
        dist_cfg.rank = int(os.environ.get("RANK", 0))
        dist_cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist_cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist_cfg.master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        dist_cfg.master_port = os.environ.get("MASTER_PORT", "29500")
    else:
        dist_cfg.rank = 0
        dist_cfg.world_size = 1

    if dist_cfg.world_size > 1:
        dist.init_process_group(
            backend=dist_cfg.backend,
            init_method=f"tcp://{dist_cfg.master_addr}:{dist_cfg.master_port}",
            world_size=dist_cfg.world_size,
            rank=dist_cfg.rank,
        )
        torch.cuda.set_device(dist_cfg.local_rank)

    return dist_cfg


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def wrap_fsdp(model: torch.nn.Module, dist_cfg: DistConfig,
              auto_wrap_min_params: int = 1e6) -> torch.nn.Module:
    """Wrap model in FSDP-2 with appropriate mixed precision strategy."""

    mp_policy = None
    if dist_cfg.mixed_precision == "bf16":
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif dist_cfg.mixed_precision == "fp16":
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    sharding_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }

    mesh = None
    if hasattr(dist, "DeviceMesh"):  # torch >= 2.2
        try:
            mesh = init_device_mesh("cuda", (dist_cfg.world_size,))
        except Exception:
            mesh = None

    wrapped = FSDP(
        model,
        sharding_strategy=sharding_map.get(dist_cfg.sharding, ShardingStrategy.FULL_SHARD),
        mixed_precision=mp_policy,
        cpu_offload=True if dist_cfg.cpu_offload else None,
        auto_wrap_policy=(
            size_based_auto_wrap_policy(min_num_params=int(auto_wrap_min_params))
            if auto_wrap_min_params > 0
            else None
        ),
        device_id=torch.cuda.current_device(),
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
        limit_all_gathers=True,
    )

    if dist_cfg.compile_model and hasattr(torch, "compile"):
        wrapped = torch.compile(wrapped, mode="reduce-overhead", fullgraph=False)

    return wrapped
