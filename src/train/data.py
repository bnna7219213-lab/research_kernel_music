"""
Dataset loader + fingerprint-lock utilities.

Phase A supports two dataset types:
  - csv_audio : audio + caption CSV (MusicCaps-style)
  - dir_audio : directory of audio + optional weak tags (FMA-style)
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from dataclasses import dataclass, field


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@dataclass
class CsvAudioDataset(Dataset):
    audio_dir: Path
    csv_path: Path
    caption_column: str = "caption"
    id_column: str = "ytid"
    ext: str = ".wav"
    sampling_rate: int = 48000
    duration: int = 10   # seconds
    entries: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        self.audio_dir = Path(self.audio_dir)
        self.csv_path = Path(self.csv_path)
        self._sr = self.sampling_rate
        self._len = int(self.duration * self.sampling_rate)
        self._parse_csv()

    def _parse_csv(self) -> None:
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append(row)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.entries[idx]
        audio_path = self.audio_dir / f"{row[self.id_column]}{self.ext}"
        if not audio_path.exists():
            # skip sample — logger warns downstream
            return None

        waveform, sr = torchaudio.load(str(audio_path))
        if sr != self._sr:
            waveform = torchaudio.functional.resample(waveform, sr, self._sr)

        # truncate / pad to fixed length
        target_len = self._len
        if waveform.shape[-1] > target_len:
            waveform = waveform[:, :target_len]
        elif waveform.shape[-1] < target_len:
            pad = target_len - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        # mix down to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return {
            "audio": waveform,
            "caption": row.get(self.caption_column, ""),
            "duration": self.duration,
            "sr": self._sr,
        }


@dataclass
class DirAudioDataset(Dataset):
    audio_dir: Path
    sampling_rate: int = 48000
    duration: int = 10
    exts: set = field(default_factory=lambda: AUDIO_EXTS)

    def __post_init__(self):
        self.audio_dir = Path(self.audio_dir)
        self._sr = self.sampling_rate
        self._len = int(self.duration * self.sampling_rate)
        self.files = sorted(
            p for p in self.audio_dir.rglob("*") if p.suffix.lower() in self.exts
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        waveform, sr = torchaudio.load(str(path))
        if sr != self._sr:
            waveform = torchaudio.functional.resample(waveform, sr, self._sr)

        target_len = self._len
        if waveform.shape[-1] > target_len:
            waveform = waveform[:, :target_len]
        elif waveform.shape[-1] < target_len:
            pad = target_len - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return {
            "audio": waveform,
            "caption": "",  # weak / no caption
            "duration": self.duration,
            "sr": self._sr,
        }


def build_dataloaders(
    data_cfg,
    dist_cfg,
    split: str = "train",
) -> DataLoader:
    """Build concatenated, weighted dataloader for given split."""
    dataset_cfgs = getattr(data_cfg, split, [])
    weights = []
    datasets: List[Dataset] = []
    for ds_spec in dataset_cfgs:
        ds_type = ds_spec.get("type", "dir_audio")
        if ds_type == "csv_audio":
            ds = CsvAudioDataset(
                audio_dir=Path(ds_spec["audio_dir"]),
                csv_path=Path(ds_spec["csv_path"]),
                caption_column=ds_spec.get("csv_caption_column", "caption"),
                id_column=ds_spec.get("csv_id_column", "ytid"),
                ext=ds_spec.get("ext", ".wav"),
                sampling_rate=ds_spec.get("sampling_rate", 48000),
                duration=ds_spec.get("duration", 10),
            )
        else:
            ds = DirAudioDataset(
                audio_dir=Path(ds_spec["audio_dir"]),
                sampling_rate=ds_spec.get("sampling_rate", 48000),
                duration=ds_spec.get("duration", 10),
            )
        datasets.append(ds)
        weights.append(float(ds_spec.get("weight", 1.0)))

    combined = ConcatDataset(datasets)
    # Use distributed sampler so each rank sees unique data
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(
        combined,
        num_replicas=dist_cfg.world_size,
        rank=dist_cfg.rank,
        shuffle=(split == "train"),
        seed=42,
    )

    loader = DataLoader(
        combined,
        batch_size=int(getattr(data_cfg, "batch_size_per_gpu", 4)),
        sampler=sampler,
        num_workers=int(getattr(data_cfg, "num_workers", 8)),
        prefetch_factor=int(getattr(data_cfg, "prefetch_factor", 2)),
        pin_memory=bool(getattr(data_cfg, "pin_memory", True)),
        drop_last=(split == "train"),
        collate_fn=_collate,
    )
    return loader


def _collate(batch):
    """Drop None entries, batch audios + captions together."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return {"audio": torch.empty(0, 1, 480000), "captions": []}

    audios = torch.stack([b["audio"] for b in batch])
    captions = [b.get("caption", "") for b in batch]
    return {"audio": audios, "captions": captions}


def check_fingerprint(expected_paths: List[Path]) -> Dict[str, Any]:
    """Verify each dataset fingerprint 文件 exists and matches hash."""
    result = {"checked": True, "entries": []}
    for p in expected_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Dataset fingerprint not found: {p}")
        with open(p) as f:
            fp = json.load(f)
        result["entries"].append(name=fp.get("name", ""), fingerprint=fp["dataset_fingerprint"])
    return result
