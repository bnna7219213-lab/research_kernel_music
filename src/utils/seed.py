"""Reproducibility seed controls."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # torch.use_deterministic_algorithms(True)  # can throw on unsupported ops
    else:
        torch.backends.cudnn.benchmark = True
