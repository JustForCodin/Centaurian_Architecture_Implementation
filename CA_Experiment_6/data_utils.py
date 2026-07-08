"""Data loading for Stage-A pretraining and Stage-B SFT (plan §5.2).

- PretrainBinDataset: memmap over concatenated uint16 token .bin files (nanoGPT
  layout). Random contiguous windows of length seq_len for next-token LM.
- SFTDataset: JSONL of §4.3 records → tokenized (chat template) with an
  assistant-span loss mask, packed/padded to seq_len.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ca_assets import format_chat, ASSISTANT_TOKEN, EOT_TOKEN
from tokenizer_util import ADATokenizer


# ── Stage-A: pretraining LM data (memmapped uint16 bins) ─────────────────

class PretrainBinDataset:
    """Random windows over a memmapped token stream. Not a torch Dataset —
    exposes get_batch() to match the nanoGPT training-loop convention."""

    def __init__(self, bin_path: str | Path, seq_len: int, device: str = "cpu",
                 seed: int = 1337):
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.device = device
        self.rng = np.random.default_rng(seed)
        if len(self.data) <= seq_len + 1:
            raise ValueError(f"{bin_path} has too few tokens ({len(self.data)})")

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(self, batch_size: int, generator: np.random.Generator | None = None):
        rng = generator or self.rng
        ix = rng.integers(0, len(self.data) - self.seq_len - 1, size=batch_size)
        x = np.stack([self.data[i:i + self.seq_len].astype(np.int64) for i in ix])
        y = np.stack([self.data[i + 1:i + 1 + self.seq_len].astype(np.int64) for i in ix])
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        if self.device.startswith("cuda"):
            xt = xt.pin_memory().to(self.device, non_blocking=True)
            yt = yt.pin_memory().to(self.device, non_blocking=True)
        else:
            xt, yt = xt.to(self.device), yt.to(self.device)
        return xt, yt


# ── Stage-B: SFT data with assistant-span masking ────────────────────────

def tokenize_sft_record(rec: dict, tok: ADATokenizer, max_len: int) -> dict | None:
    """Tokenize one §4.3 record → {input_ids, loss_mask}. Loss is on assistant
    spans only (tokens strictly after <|assistant|> up to & including <|endofturn|>).
    Over-length records are trimmed inside the context (then system) span so the
    assistant answer is always preserved. Returns None if no answer survives."""
    text = format_chat(rec["messages"], context=rec.get("context"), sci=rec.get("sci"),
                       persona_state=rec.get("persona_state"))
    ids = tok.encode(text)
    ids = _trim_to_fit(ids, tok, max_len)
    mask = _assistant_span_mask(ids, tok)
    if sum(mask) == 0:
        return None
    has_persona = tok.persona_id is not None and tok.persona_id in ids
    return {"input_ids": ids, "loss_mask": mask, "has_persona": has_persona}


def _trim_to_fit(ids: list[int], tok: ADATokenizer, max_len: int) -> list[int]:
    """Trim over-length token lists by shrinking the context span first, then the
    system span, preserving the trailing assistant answer. Head-truncates only as
    a last resort (keeps the tail = the answer)."""
    if len(ids) <= max_len:
        return ids
    ids = list(ids)
    overflow = len(ids) - max_len
    # (span-open token, tokens that close the span). Order: trim the context span
    # first, then the system span. The <|persona|> QPM channel is never trimmed —
    # the system span is closed by persona_id (when present) so it is preserved.
    persona_id = getattr(tok, "persona_id", None)
    sys_close = tuple(t for t in (persona_id, tok.context_id, tok.user_id) if t is not None)
    for open_id, close_ids in ((tok.context_id, (tok.user_id,)),
                               (tok.system_id, sys_close)):
        if overflow <= 0 or open_id not in ids:
            continue
        start = ids.index(open_id) + 1
        ends = [ids.index(c) for c in close_ids if c in ids and ids.index(c) > start]
        end = min(ends) if ends else len(ids)
        span = end - start
        cut = min(span - 1, overflow)          # keep at least 1 token of the span
        if cut > 0:
            del ids[end - cut:end]
            overflow -= cut
    if len(ids) > max_len:                      # last resort: keep the tail
        ids = ids[len(ids) - max_len:]
    return ids


def _assistant_span_mask(ids: list[int], tok: ADATokenizer) -> list[int]:
    """1 on tokens that belong to an assistant answer span, else 0.
    A span opens right after an <|assistant|> token and closes at (and includes)
    the following <|endofturn|> token — so the model is supervised to emit EOT."""
    mask = [0] * len(ids)
    in_span = False
    for i, t in enumerate(ids):
        if t == tok.assistant_id:
            in_span = True
            continue                      # the marker itself is not supervised
        if in_span:
            mask[i] = 1                    # supervise content (and the EOT below)
            if t == tok.eot_id:
                in_span = False
    return mask


class SFTDataset(torch.utils.data.Dataset):
    """JSONL of §4.3 records → padded (input_ids, targets, loss_mask) tensors."""

    def __init__(self, jsonl_path: str | Path, tok: ADATokenizer, seq_len: int,
                 sources: set[str] | None = None):
        self.tok = tok
        self.seq_len = seq_len
        self.examples: list[dict] = []
        skipped = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if sources is not None and rec.get("source") not in sources:
                    continue
                ex = tokenize_sft_record(rec, tok, seq_len + 1)
                if ex is None:
                    skipped += 1
                    continue
                self.examples.append(ex)
        if not self.examples:
            raise ValueError(f"no usable SFT examples in {jsonl_path}")
        print(f"SFTDataset: {len(self.examples)} examples ({skipped} skipped) from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        ids = ex["input_ids"]
        m = ex["loss_mask"]
        # next-token: x = ids[:-1], y = ids[1:], mask aligns with y
        x = ids[:-1]
        y = ids[1:]
        my = m[1:]
        # prefix_len = index of the first answer token (everything before it —
        # system/persona/context/question/assistant-marker — is the prefix-LM prefix)
        prefix_len = m.index(1) if 1 in m else len(x)
        return x, y, my, prefix_len

    def collate(self, batch):
        """Pad to the longest in-batch; masked/pad target positions → ignore.
        Returns (X, Y, M, prefix_lens) — prefix_lens is used only by prefix-LM SFT."""
        maxlen = max(len(x) for x, _, _, _ in batch)
        pad = self.tok.pad_id
        X, Y, M, P = [], [], [], []
        for x, y, my, pl in batch:
            n = maxlen - len(x)
            X.append(x + [pad] * n)
            Y.append(y + [-100] * n)             # -100 ignored by cross_entropy
            M.append(my + [0] * n)
            P.append(pl)
        return (torch.tensor(X, dtype=torch.long),
                torch.tensor(Y, dtype=torch.long),
                torch.tensor(M, dtype=torch.float),
                torch.tensor(P, dtype=torch.long))
