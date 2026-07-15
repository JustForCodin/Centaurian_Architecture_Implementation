"""Shared training helpers: seeding, cosine LR schedule, checkpoint I/O (§5.3).

Checkpoints are self-describing (model config + train config + step) so a
Colab reconnect resumes exactly where it left off from the Drive mirror.
"""

from __future__ import annotations

import math
import os
import random
import shutil
import tempfile
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
    # IMPORTANT: os.replace()/rename on the Google Drive FUSE mount does NOT
    # overwrite — Drive keys files by internal ID, not path, so a rename onto an
    # existing name creates a *duplicate* (two files, same name) and leaves the
    # old one, silently filling storage. Write the tmp on LOCAL disk (atomic
    # there), then copy it *in place* over the destination: copyfile opens the
    # dest with 'wb', truncating the existing file (same ID) and rewriting its
    # contents — no rename, so Drive overwrites cleanly with no duplicate.
    fd, tmp = tempfile.mkstemp(suffix=".pt", dir="/tmp")
    os.close(fd)
    try:
        torch.save(ckpt, tmp)
        shutil.copyfile(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def load_checkpoint(path, map_location="cpu"):
    return torch.load(str(path), map_location=map_location, weights_only=False)


def build_model_from_ckpt(ckpt: dict, map_location="cpu") -> ADATransformer:
    cfg = ModelConfig.from_dict(ckpt["model_cfg"])
    model = ADATransformer(cfg)
    load_backbone(model, ckpt["model"])
    return model.to(map_location)


def load_backbone(model: ADATransformer, state: dict) -> None:
    """Load a checkpoint into `model`, tolerating the read heads being (a) absent
    from the checkpoint (an older backbone predates them) or (b) a different
    SHAPE than the model (e.g. the answerability head grew span-confidence
    inputs, or the checkpoint carries an untrained old-shape head). In both cases
    the head is left at the model's fresh init. Any mismatch on a *non-head*
    (trunk) tensor is a real config error and raises."""
    fresh_prefixes = ("span_head", "answerable_head")
    model_sd = model.state_dict()
    # keep only checkpoint tensors that exist in the model with a matching shape;
    # everything else (shape-changed heads, extra keys) is skipped
    filtered, skipped = {}, []
    for k, v in state.items():
        if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape):
            filtered[k] = v
        else:
            skipped.append(k)
    incompat = model.load_state_dict(filtered, strict=False)
    bad = ([k for k in incompat.missing_keys if not k.startswith(fresh_prefixes)]
           + [k for k in skipped if not k.startswith(fresh_prefixes)])
    if bad:
        raise RuntimeError(f"checkpoint/model mismatch on non-head tensors: {bad}")
    reinit = sorted({k.split(".")[0] for k in list(incompat.missing_keys) + skipped
                     if k.startswith(fresh_prefixes)})
    if reinit:
        print(f"  [ckpt] heads (re)initialised fresh: {'/'.join(reinit)}")


def latest_checkpoint(ckpt_dir, prefix: str) -> Path | None:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    cands = sorted(ckpt_dir.glob(f"{prefix}_step*.pt"))
    return cands[-1] if cands else None
