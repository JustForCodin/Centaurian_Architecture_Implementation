#!/usr/bin/env python3
"""Stage-A pretraining — next-token LM on the free corpus (plan §5.2, §5.3).

Checkpoint-resilient: mirrors to --ckpt-dir (point it at Drive on Colab) every
ckpt_interval steps and resumes from the latest on restart. Cosine LR + warmup,
AdamW, bf16 autocast, gradient accumulation. Stop on val-loss plateau.

Usage (Colab A100):
    python train_pretrain.py --train-bin data/pretrain/train.bin \
        --val-bin data/pretrain/val.bin --ckpt-dir /content/drive/MyDrive/CA_Experiment_6/checkpoints \
        --max-steps 40000

Pilot (§5.4, tiny model, free T4):
    python train_pretrain.py --config pilot --max-steps 2000 --train-bin data/pretrain/train.bin \
        --val-bin data/pretrain/val.bin
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from model import ADATransformer, ModelConfig, TrainConfig, ADA_80M, ADA_160M, ADA_300M, ADA_PILOT, ADA_SMOKE
from data_utils import PretrainBinDataset
from train_common import (
    set_seed, cosine_lr, pick_device_dtype, save_checkpoint,
    load_checkpoint, latest_checkpoint, load_backbone,
)

CONFIGS = {"80m": ADA_80M, "160m": ADA_160M, "300m": ADA_300M, "pilot": ADA_PILOT, "smoke": ADA_SMOKE}


def prefix_lm_masks(x, seq_len, lo, hi, device):
    """Phase-2 prefix-LM objective: per sequence pick a random split b — the
    prefix [0,b) attends bidirectionally, the suffix [b,T) stays causal, and the
    LM loss is computed on the suffix only (prefix positions can see their own
    next token under bidirectional attention, so their prediction is trivial).
    Returns (prefix_lens, loss_mask). Teaches the backbone to build the
    bidirectional representations the span/answerability heads read from."""
    B = x.shape[0]
    b_lo = max(1, int(seq_len * lo))
    b_hi = max(b_lo + 1, int(seq_len * hi))
    bs = torch.randint(b_lo, b_hi, (B,), device=device)
    pos = torch.arange(seq_len, device=device).view(1, seq_len)
    mask = (pos >= bs.view(-1, 1)).float()             # loss on suffix only
    return bs, mask


@torch.no_grad()
def estimate_loss(model, data: PretrainBinDataset, batch_size: int, iters: int, ctx,
                  prefix_lm=False, seq_len=0, frac=(0.25, 0.75)):
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = data.get_batch(batch_size)
        with ctx:
            if prefix_lm:
                bs, m = prefix_lm_masks(x, seq_len, frac[0], frac[1], x.device)
                _, loss = model(x, y, loss_mask=m, prefix_lens=bs)
            else:
                _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS, default="80m")
    ap.add_argument("--train-bin", required=True)
    ap.add_argument("--val-bin", required=True)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="override model vocab (must match the trained tokenizer)")
    ap.add_argument("--ckpt-interval", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="resume from latest ckpt")
    ap.add_argument("--plateau-patience", type=int, default=5,
                    help="stop after this many evals without val improvement")
    ap.add_argument("--init", default=None,
                    help="Phase-2: continue from this backbone checkpoint (fresh "
                         "optimizer/step) instead of training from random init")
    ap.add_argument("--prefix-lm", action="store_true",
                    help="Phase-2 prefix-LM objective (bidirectional prefix, causal "
                         "suffix, loss on suffix) — improves reading/abstention reps")
    ap.add_argument("--prefix-frac", type=float, nargs=2, default=(0.25, 0.75),
                    metavar=("LO", "HI"), help="prefix-length fraction range")
    ap.add_argument("--run-name", default=None,
                    help="checkpoint name prefix (default pretrain_<config>); set a "
                         "distinct name for Phase-2 so it doesn't clobber Stage-A")
    args = ap.parse_args()
    run_name = args.run_name or f"pretrain_{args.config}"

    import dataclasses
    mcfg: ModelConfig = CONFIGS[args.config]
    if args.vocab_size:
        mcfg = dataclasses.replace(mcfg, vocab_size=args.vocab_size)
    tcfg = TrainConfig(stage="pretrain")
    if args.max_steps:  tcfg.max_steps = args.max_steps
    if args.batch_size: tcfg.batch_size = args.batch_size
    if args.grad_accum: tcfg.grad_accum_steps = args.grad_accum
    if args.lr:         tcfg.lr = args.lr
    if args.ckpt_interval: tcfg.ckpt_interval = args.ckpt_interval
    tcfg.seq_len = args.seq_len or min(mcfg.max_seq_len, tcfg.seq_len)

    set_seed(tcfg.seed)
    device, dtype = pick_device_dtype(tcfg.dtype)
    ctx = (nullcontext() if device == "cpu"
           else torch.amp.autocast(device_type="cuda", dtype=dtype))
    print(f"config={args.config} device={device} dtype={dtype} seq_len={tcfg.seq_len}")

    train_data = PretrainBinDataset(args.train_bin, tcfg.seq_len, device=device)
    val_data = PretrainBinDataset(args.val_bin, tcfg.seq_len, device=device)

    model = ADATransformer(mcfg)
    if args.init:                                      # Phase-2: continue from backbone
        ick = load_checkpoint(args.init, map_location="cpu")
        load_backbone(model, ick["model"])
        print(f"Phase-2 init from {args.init} (fresh optimizer/step)")
    model = model.to(device)
    tok_per_step = tcfg.batch_size * tcfg.grad_accum_steps * tcfg.seq_len
    obj = f"prefix-LM (prefix {args.prefix_frac[0]:.2f}-{args.prefix_frac[1]:.2f})" \
          if args.prefix_lm else "causal-LM"
    print("=" * 68, flush=True)
    print(f"=== {'Phase-2 continued' if args.init else 'Stage-A'} pretrain — "
          f"config={args.config} | obj={obj} | run={run_name} ===", flush=True)
    print(f"    params {model.num_params()/1e6:.1f}M | d_model {mcfg.d_model} "
          f"layers {mcfg.n_layers} heads {mcfg.n_heads} vocab {mcfg.vocab_size}", flush=True)
    print(f"    device {device} | dtype {dtype} | max_steps {tcfg.max_steps} | "
          f"tok/step {tok_per_step:,} | target tokens {tcfg.max_steps*tok_per_step/1e9:.2f}B", flush=True)
    print("=" * 68, flush=True)
    opt = model.configure_optimizers(tcfg.weight_decay, tcfg.lr,
                                     (tcfg.beta1, tcfg.beta2), device)
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    start_step, best_val = 0, float("inf")
    ckpt_dir = Path(args.ckpt_dir)
    if args.resume:
        # Prefer the rolling checkpoint; fall back to any legacy step*.pt (first
        # resume after switching to rolling saves), then to best.pt.
        rolling = ckpt_dir / f"{run_name}_last.pt"
        last = rolling if rolling.exists() else latest_checkpoint(ckpt_dir, run_name)
        if last is None:
            best = ckpt_dir / f"{run_name}_best.pt"
            last = best if best.exists() else None
        if last:
            ck = load_checkpoint(last, map_location=device)
            model.load_state_dict(ck["model"])
            if ck.get("optimizer"):
                opt.load_state_dict(ck["optimizer"])
            start_step = ck["step"] + 1
            best_val = ck.get("best_val", float("inf"))
            print(f"resumed from {last} @ step {start_step}")

    if tcfg.compile and device == "cuda":
        model = torch.compile(model)

    model.train()
    t0 = time.time()
    no_improve = 0
    running, n_log = 0.0, 0
    for step in range(start_step, tcfg.max_steps):
        lr = cosine_lr(step, lr=tcfg.lr, min_lr=tcfg.min_lr,
                       warmup_steps=tcfg.warmup_steps, max_steps=tcfg.max_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        for micro in range(tcfg.grad_accum_steps):
            x, y = train_data.get_batch(tcfg.batch_size)
            with ctx:
                if args.prefix_lm:
                    bs, m = prefix_lm_masks(x, tcfg.seq_len, args.prefix_frac[0],
                                            args.prefix_frac[1], x.device)
                    _, loss = model(x, y, loss_mask=m, prefix_lens=bs)
                else:
                    _, loss = model(x, y)
                loss = loss / tcfg.grad_accum_steps
            scaler.scale(loss).backward()
            running += loss.item()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        n_log += 1

        if step % tcfg.log_interval == 0:
            dt = time.time() - t0
            tok_per_step = tcfg.batch_size * tcfg.grad_accum_steps * tcfg.seq_len
            tps = tok_per_step * n_log / dt if dt > 0 else 0
            toks_seen = (step + 1) * tok_per_step
            eta_h = (tcfg.max_steps - step) * (dt / max(n_log, 1)) / 3600
            print(f"step {step:6d}/{tcfg.max_steps} | loss {running/n_log:.4f} | "
                  f"lr {lr:.2e} | gnorm {float(gnorm):.2f} | {tps/1e3:.1f}k tok/s | "
                  f"seen {toks_seen/1e6:.0f}M | ETA {eta_h:.1f}h", flush=True)
            running, n_log = 0.0, 0
            t0 = time.time()

        if step > 0 and step % tcfg.eval_interval == 0:
            val = estimate_loss(model, val_data, tcfg.batch_size, tcfg.eval_iters, ctx,
                                prefix_lm=args.prefix_lm, seq_len=tcfg.seq_len,
                                frac=args.prefix_frac)
            print(f"  >> eval step {step}: val_loss {val:.4f} (best {best_val:.4f})", flush=True)
            improved = val < best_val - 1e-3
            if improved:
                best_val = val
                no_improve = 0
                save_checkpoint(ckpt_dir / f"{run_name}_best.pt",
                                model=model, optimizer=opt, model_cfg=mcfg,
                                train_cfg=tcfg, step=step, best_val=best_val,
                                extra={"prefix_lm_pretrain": args.prefix_lm})
                print(f"     ✓ new best val_loss {best_val:.4f} → "
                      f"{run_name}_best.pt", flush=True)
            else:
                no_improve += 1
                print(f"     · no improvement ({no_improve}/{args.plateau_patience})", flush=True)
            if no_improve >= args.plateau_patience:
                print(f"val plateau ({no_improve} evals) — stopping at step {step}", flush=True)
                break

        if step > 0 and step % tcfg.ckpt_interval == 0:
            # Rolling checkpoint: overwrite one file (resume point) instead of
            # accumulating a new ~0.8 GB step file every interval on Drive.
            p = save_checkpoint(ckpt_dir / f"{run_name}_last.pt",
                                model=model, optimizer=opt, model_cfg=mcfg,
                                train_cfg=tcfg, step=step, best_val=best_val,
                                extra={"prefix_lm_pretrain": args.prefix_lm})
            print(f"     ⤓ rolling checkpoint → {p.name} (mirrored to Drive)", flush=True)

    save_checkpoint(ckpt_dir / f"{run_name}_final.pt",
                    model=model, optimizer=opt, model_cfg=mcfg,
                    train_cfg=tcfg, step=tcfg.max_steps - 1, best_val=best_val,
                    extra={"prefix_lm_pretrain": args.prefix_lm})
    print(f"done. best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
