"""Shared training helpers: seeding, cosine LR schedule, checkpoint I/O (§5.3).

Checkpoints are self-describing (model config + train config + step) so a
Colab reconnect resumes exactly where it left off from the Drive mirror.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from model import ModelConfig, ADATransformer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, *, lr: float, min_lr: float,
              warmup_steps: int, max_steps: int) -> float:
    """Linear warmup then cosine decay to min_lr (nanoGPT convention)."""
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def pick_device_dtype(pref_dtype: str):
    if torch.cuda.is_available():
        device = "cuda"
        if pref_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif pref_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32
    return device, dtype


def save_checkpoint(path, *, model, optimizer, model_cfg: ModelConfig,
                    train_cfg, step: int, best_val: float, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "model_cfg": model_cfg.to_dict(),
        "train_cfg": train_cfg.to_dict() if hasattr(train_cfg, "to_dict") else dict(train_cfg),
        "step": step,
        "best_val": best_val,
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)             # atomic — survives a mid-write disconnect
    return path


def load_checkpoint(path, map_location="cpu"):
    return torch.load(str(path), map_location=map_location, weights_only=False)


def build_model_from_ckpt(ckpt: dict, map_location="cpu") -> ADATransformer:
    cfg = ModelConfig.from_dict(ckpt["model_cfg"])
    model = ADATransformer(cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(map_location)


def latest_checkpoint(ckpt_dir, prefix: str) -> Path | None:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    cands = sorted(ckpt_dir.glob(f"{prefix}_step*.pt"))
    return cands[-1] if cands else None
