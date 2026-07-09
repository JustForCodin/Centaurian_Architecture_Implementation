#!/usr/bin/env python3
"""Phase-1 extractive span-head training (plan: "reach 0.70 from scratch").

Fine-tunes a discriminative start/end span head on top of an existing pretrained
backbone. Reading is bidirectional (encoder-style — the whole question+context is
the prefix), so the context representations are question-aware. This isolates the
*objective* lever (span extraction vs generation) without re-pretraining: if a
span head on the current 8B-token backbone already lifts F1 well above the
generative baseline, scaling the pretrain (Phase 2) should reach 0.70.

Only the answer's start/end token positions are supervised; unanswerable examples
are labelled at the null anchor (position 0 = <|bos|>). The backbone stays
trainable (the span signal has to reshape representations, not just the head).

Usage (Colab):
    python train_span.py --init checkpoints/pretrain_80m_best.pt \
        --span data/qa_span.jsonl \
        --ckpt-dir /content/drive/MyDrive/CA_Experiment_6/checkpoints --max-steps 4000
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import ADATransformer, ModelConfig, TrainConfig
from tokenizer_util import ADATokenizer
from span_utils import build_span_input
from train_common import (
    set_seed, cosine_lr, pick_device_dtype, save_checkpoint,
    load_checkpoint, latest_checkpoint, load_backbone,
)


class SpanDataset(Dataset):
    """JSONL span records → (ids, start_pos, end_pos, ctx_lo, n_ctx) tensors."""

    def __init__(self, jsonl_path, tok: ADATokenizer, max_len: int):
        self.tok = tok
        self.max_len = max_len
        self.rows: list[dict] = []
        skipped = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                ex = self._build(r)
                if ex is None:
                    skipped += 1
                    continue
                self.rows.append(ex)
        if not self.rows:
            raise ValueError(f"no usable span examples in {jsonl_path}")
        n_ans = sum(1 for r in self.rows if r["answerable"])
        print(f"SpanDataset: {len(self.rows)} examples ({n_ans} answerable, "
              f"{len(self.rows)-n_ans} null, {skipped} skipped) from {jsonl_path}")

    def _build(self, r: dict):
        answerable = bool(r["answerable"])
        answer_char = None
        if answerable:
            s = int(r["answer_start"])
            answer_char = (s, s + len(r["answer"]))
        out = build_span_input(self.tok, r["question"], r["context"],
                               self.max_len, answer_char=answer_char)
        if out is None:                                   # unalignable answer
            return None
        ids = out["ids"]
        if answerable:
            start_pos, end_pos = out["label"]
        else:
            start_pos = end_pos = 0                       # null anchor (<|bos|>)
        n_ctx = len(ids) - out["ctx_lo"]
        return {"ids": ids, "start": start_pos, "end": end_pos,
                "ctx_lo": out["ctx_lo"], "n_ctx": n_ctx, "answerable": answerable}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

    def collate(self, batch):
        pad = self.tok.pad_id
        maxlen = max(len(b["ids"]) for b in batch)
        X, valid, starts, ends, plens, ans = [], [], [], [], [], []
        for b in batch:
            ids = b["ids"]
            n = maxlen - len(ids)
            X.append(ids + [pad] * n)
            # Valid answer positions: the null anchor (0) + this example's context
            # span. Everything else (question, pads) is masked out of the softmax.
            v = [0.0] * maxlen
            v[0] = 1.0
            for p in range(b["ctx_lo"], b["ctx_lo"] + b["n_ctx"]):
                v[p] = 1.0
            valid.append(v)
            starts.append(b["start"])
            ends.append(b["end"])
            plens.append(len(ids))                        # bidirectional over real tokens
            ans.append(1.0 if b["answerable"] else 0.0)
        return (torch.tensor(X, dtype=torch.long),
                torch.tensor(valid, dtype=torch.float),
                torch.tensor(starts, dtype=torch.long),
                torch.tensor(ends, dtype=torch.long),
                torch.tensor(plens, dtype=torch.long),
                torch.tensor(ans, dtype=torch.float))


def _masked_ce(logits, valid, target):
    """Cross-entropy over valid positions only (invalid → -inf before softmax)."""
    neg = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(valid < 0.5, neg)
    return F.cross_entropy(logits, target)


def cycle(loader):
    while True:
        for b in loader:
            yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="pretrained backbone checkpoint")
    ap.add_argument("--span", default="data/qa_span.jsonl")
    ap.add_argument("--tokenizer", default="tokenizer/ada_bpe.json")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--ans-weight", type=float, default=1.0,
                    help="weight on the answerability BCE loss term")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    ck = load_checkpoint(args.init, map_location="cpu")
    mcfg = ModelConfig.from_dict(ck["model_cfg"])
    seq_len = min(args.seq_len, mcfg.max_seq_len)
    tcfg = TrainConfig(stage="sft", max_steps=args.max_steps, batch_size=args.batch_size,
                       grad_accum_steps=args.grad_accum, lr=args.lr, seq_len=seq_len,
                       warmup_steps=100)

    set_seed(tcfg.seed)
    device, dtype = pick_device_dtype(tcfg.dtype)
    ctx = (nullcontext() if device == "cpu"
           else torch.amp.autocast(device_type="cuda", dtype=dtype))

    tok = ADATokenizer.load(args.tokenizer)
    assert tok.vocab_size == mcfg.vocab_size

    ds = SpanDataset(args.span, tok, seq_len)
    loader = DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True,
                        collate_fn=ds.collate, drop_last=True)
    it = cycle(loader)

    model = ADATransformer(mcfg)
    load_backbone(model, ck["model"])
    model = model.to(device)
    opt = model.configure_optimizers(tcfg.weight_decay, tcfg.lr,
                                     (tcfg.beta1, tcfg.beta2), device)
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    start_step = 0
    ckpt_dir = Path(args.ckpt_dir)
    if args.resume:
        rolling = ckpt_dir / "span_last.pt"
        last = rolling if rolling.exists() else latest_checkpoint(ckpt_dir, "span")
        if last:
            rck = load_checkpoint(last, map_location=device)
            load_backbone(model, rck["model"])
            if rck.get("optimizer"):
                opt.load_state_dict(rck["optimizer"])
            start_step = rck["step"] + 1
            print(f"resumed span from {last} @ step {start_step}")

    print("=" * 68, flush=True)
    print(f"=== Phase-1 span head — {len(ds)} examples | seq_len {seq_len} | "
          f"attn=bidirectional | params {model.num_params()/1e6:.1f}M ===", flush=True)
    print(f"    init {args.init} | max_steps {tcfg.max_steps} | "
          f"batch {tcfg.batch_size}x{tcfg.grad_accum_steps} | lr {tcfg.lr:.1e} | "
          f"device {device}", flush=True)
    print("=" * 68, flush=True)

    model.train()
    running, run_ans, n_log, run_t0, t0 = 0.0, 0.0, 0, time.time(), time.time()
    for step in range(start_step, tcfg.max_steps):
        lr = cosine_lr(step, lr=tcfg.lr, min_lr=tcfg.lr * 0.1,
                       warmup_steps=tcfg.warmup_steps, max_steps=tcfg.max_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(tcfg.grad_accum_steps):
            x, valid, s_lab, e_lab, plens, a_lab = next(it)
            x, valid = x.to(device), valid.to(device)
            s_lab, e_lab, plens = s_lab.to(device), e_lab.to(device), plens.to(device)
            a_lab = a_lab.to(device)
            with ctx:
                s_logits, e_logits, a_logit = model.read_heads(x, prefix_lens=plens)
                span_loss = 0.5 * (_masked_ce(s_logits, valid, s_lab)
                                   + _masked_ce(e_logits, valid, e_lab))
                ans_loss = F.binary_cross_entropy_with_logits(a_logit, a_lab)
                loss = span_loss + args.ans_weight * ans_loss
            loss = loss / tcfg.grad_accum_steps
            scaler.scale(loss).backward()
            running += loss.item()
            run_ans += ans_loss.item() / tcfg.grad_accum_steps
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        n_log += 1

        if step % tcfg.log_interval == 0:
            dt = time.time() - t0
            eta_m = (tcfg.max_steps - step) * (dt / max(n_log, 1)) / 60
            print(f"step {step:5d}/{tcfg.max_steps} | loss {running/n_log:.4f} | "
                  f"ans {run_ans/n_log:.4f} | lr {lr:.2e} | gnorm {float(gnorm):.2f} | "
                  f"{dt/max(n_log,1):.2f}s/step | ETA {eta_m:.1f}m", flush=True)
            running, run_ans, n_log, t0 = 0.0, 0.0, 0, time.time()

        if step > 0 and step % tcfg.ckpt_interval == 0:
            p = save_checkpoint(ckpt_dir / "span_last.pt", model=model, optimizer=opt,
                                model_cfg=mcfg, train_cfg=tcfg, step=step, best_val=0.0,
                                extra={"span_head": True, "answerable_head": True})
            print(f"     ⤓ checkpoint → {p.name}", flush=True)

    save_checkpoint(ckpt_dir / "span_final.pt", model=model, optimizer=opt,
                    model_cfg=mcfg, train_cfg=tcfg, step=tcfg.max_steps - 1, best_val=0.0,
                    extra={"span_head": True, "answerable_head": True})
    print(f"span done → span_final.pt ({time.time()-run_t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
