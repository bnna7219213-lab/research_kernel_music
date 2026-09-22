"""
Unified logger wrapper: W&B (primary) + MLflow (fallback/offline).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.distributed import DistConfig, is_main_process


_LAZY = {}


def setup_logger(cfg, dist_cfg: DistConfig) -> None:
    """Initialize logging backends (only on main process for W&B)."""
    log_dir = Path(cfg.logging.log_dir) if "log_dir" in cfg.logging else Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    #  Python stdlib logger
    logging.basicConfig(
        level=logging.INFO,
        format=f"[rank {dist_cfg.rank}] %(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f"rank{dist_cfg.rank}.log"),
        ],
    )

    if not is_main_process():
        return

    # W&B
    backend = getattr(cfg.logging, "backend", "wandb")
    if backend == "wandb":
        try:
            import wandb
            wandb.init(
                project=cfg.logging.project,
                entity=getattr(cfg.logging, "entity", None),
                name=cfg.name,
                config=_flatten_omegaconf(cfg),
                dir=str(log_dir),
                save_code=getattr(cfg.logging, "save_code", True),
            )
            _LAZY["wandb"] = wandb
            logging.info(f"[W&B] initialized: {cfg.logging.project}/{cfg.name}")
        except Exception as e:
            logging.warning(f"[W&B] init failed, falling back to MLflow: : {e}")
            _setup_mlflow(cfg, log_dir)
    else:
        _setup_mlflow(cfg, log_dir)


def _setup_mlflow(cfg, log_dir: Path) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file:{log_dir}/mlruns")
        mlflow.set_experiment(cfg.name)
        mlflow.start_run(run_name=cfg.name)
        mlflow.log_params(_flatten_omegaconf(cfg))
        _LAZY["mlflow"] = mlflow
        logging.info("[MLflow] initialized (offline).")
    except Exception as e:
        logging.warning(f"[MLflow] init failed: : {e}")


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    if "wandb" in _LAZY:
        _LAZY["wandb"].log(metrics, step=step)
    if "mlflow" in _LAZY:
        import mlflow
        mlflow.log_metrics(metrics, step=step)


def log_audio(name: str, audio_path: Path, sample_rate: int = 48000) -> None:
    if "wandb" in _LAZY:
        import wandb
        wandb.log({name: wandb.Audio(str(audio_path), sample_rate=sample_rate)})


def finish() -> None:
    if "wandb" in _LAZY:
        _LAZY["wandb"].finish()
    if "mlflow" in _LAZY:
        import mlflow
        mlflow.end_run()


def _flatten_omegaconf(cfg) -> dict:
    """Flatten OmegaConf to a flat dict for logging."""
    try:
        from omegaconf import OmegaConf
        return _flatten_dot_notation(OmegaConf.to_container(cfg, resolve=True))
    except Exception:
        return {}


def _flatten_dot_notation(d: dict, parent_key: str = "") -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten_dot_notation(v, new_key))
        else:
            items[new_key] = v
    return items
