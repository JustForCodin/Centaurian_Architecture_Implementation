#!/usr/bin/env python3
"""Stage-B SFT — grounded-QA + ADA persona/style/refusal (plan §5.2).

Loss is masked to the assistant span (answer | refusal). A fraction of Stage-A
LM data is mixed in each step to prevent fluency forgetting (sft_replay_frac).
Initialises from the Stage-A pretrained checkpoint.

Usage (Colab):
    python train_sft.py --init checkpoints/pretrain_80m_best.pt \
        --sft data/qa_sft.jsonl --pretrain-bin data/pretrain/train.bin \
        --ckpt-dir /content/drive/MyDrive/CA_Experiment_6/checkpoints --max-steps 6000
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import ADATransformer, ModelConfig, TrainConfig
from data_utils import SFTDataset, PretrainBinDataset
from tokenizer_util import ADATokenizer
from train_common import (
    set_seed, cosine_lr, pick_device_dtype, save_checkpoint,
    load_checkpoint, latest_checkpoint, load_backbone,
)


def cycle(loader):
    while True:
        for b in loader:
            yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="Stage-A checkpoint to fine-tune")
    ap.add_argument("--sft", default="data/qa_sft.jsonl")
    ap.add_argument("--pretrain-bin", default=None, help="Stage-A bin for replay mix")
    ap.add_argument("--tokenizer", default="tokenizer/ada_bpe.json")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--replay-frac", type=float, default=0.1)
    ap.add_argument("--sources", nargs="*", default=None,
                    help="restrict SFT to these record sources")
    ap.add_argument("--reading-cap", type=int, default=None,
                    help="cap non-persona (reading) records — reading is the span "
                         "head's job now; a huge SQuAD set drowns the persona")
    ap.add_argument("--persona-oversample", type=int, default=1,
                    help="replicate sonnet_* persona records N× to counter the "
                         "reading imbalance so the LM head learns ADA's voice")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--prefix-lm", action="store_true",
                    help="bidirectional attention over the prefix (system+persona+context+"
                         "question), causal over the answer — lets context attend to the question")
    args = ap.parse_args()

    ck = load_checkpoint(args.init, map_location="cpu")
    mcfg = ModelConfig.from_dict(ck["model_cfg"])
    seq_len = min(args.seq_len, mcfg.max_seq_len)
    if seq_len < args.seq_len:
        print(f"note: clamping SFT seq_len {args.seq_len} → model max {seq_len}")
    tcfg = TrainConfig(stage="sft", max_steps=args.max_steps, batch_size=args.batch_size,
                       grad_accum_steps=args.grad_accum, lr=args.lr, seq_len=seq_len,
                       warmup_steps=100, sft_replay_frac=args.replay_frac)

    set_seed(tcfg.seed)
    device, dtype = pick_device_dtype(tcfg.dtype)
    ctx = (nullcontext() if device == "cpu"
           else torch.amp.autocast(device_type="cuda", dtype=dtype))

    tok = ADATokenizer.load(args.tokenizer)
    assert tok.vocab_size == mcfg.vocab_size, \
        f"tokenizer vocab {tok.vocab_size} != model vocab {mcfg.vocab_size}"

    sources = set(args.sources) if args.sources else None
    ds = SFTDataset(args.sft, tok, tcfg.seq_len, sources=sources,
                    reading_cap=args.reading_cap,
                    persona_oversample=args.persona_oversample)
    loader = DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True,
                        collate_fn=ds.collate, drop_last=True)
    sft_iter = cycle(loader)

    replay = None
    if args.pretrain_bin and tcfg.sft_replay_frac > 0:
        replay = PretrainBinDataset(args.pretrain_bin, tcfg.seq_len, device=device)

    model = ADATransformer(mcfg)
    load_backbone(model, ck["model"])
    model = model.to(device)
    print(f"SFT init from {args.init} | params {model.num_params()/1e6:.1f}M | device {device}")
    opt = model.configure_optimizers(tcfg.weight_decay, tcfg.lr,
                                     (tcfg.beta1, tcfg.beta2), device)
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    start_step = 0
    ckpt_dir = Path(args.ckpt_dir)
    if args.resume:
        rolling = ckpt_dir / "sft_last.pt"
        last = rolling if rolling.exists() else latest_checkpoint(ckpt_dir, "sft")
        if last:
            rck = load_checkpoint(last, map_location=device)
            load_backbone(model, rck["model"])
            if rck.get("optimizer"):
                opt.load_state_dict(rck["optimizer"])
            start_step = rck["step"] + 1
            print(f"resumed SFT from {last} @ step {start_step}")

    n_persona = sum(1 for ex in ds.examples if ex.get("has_persona"))
    print("=" * 68, flush=True)
    print(f"=== Stage-B SFT — {len(ds)} examples ({n_persona} with QPM <|persona|> "
          f"channel) | seq_len {tcfg.seq_len} | replay_frac {tcfg.sft_replay_frac} | "
          f"attn={'prefix-LM' if args.prefix_lm else 'causal'} ===", flush=True)
    print(f"    max_steps {tcfg.max_steps} | batch {tcfg.batch_size}x{tcfg.grad_accum_steps} | "
          f"lr {tcfg.lr:.1e} | device {device}", flush=True)
    print("=" * 68, flush=True)
    model.train()
    running, n_log, n_replay, t0, run_t0 = 0.0, 0, 0, time.time(), time.time()
    for step in range(start_step, tcfg.max_steps):
        lr = cosine_lr(step, lr=tcfg.lr, min_lr=tcfg.lr * 0.1,
                       warmup_steps=tcfg.warmup_steps, max_steps=tcfg.max_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        for micro in range(tcfg.grad_accum_steps):
            use_replay = replay is not None and (torch.rand(1).item() < tcfg.sft_replay_frac)
            if use_replay:
                x, y = replay.get_batch(tcfg.batch_size)
                with ctx:
                    _, loss = model(x, y)                       # plain LM loss
            else:
                x, y, m, plens = next(sft_iter)
                x, y, m = x.to(device), y.to(device), m.to(device)
                # prefix-LM: bidirectional over the prefix (system+persona+context+
                # question), causal over the answer, so context can attend to the
                # question while reading. Replay (plain-LM) batches stay causal.
                plens = plens.to(device) if args.prefix_lm else None
                with ctx:
                    _, loss = model(x, y, loss_mask=m, prefix_lens=plens)
            loss = loss / tcfg.grad_accum_steps
            scaler.scale(loss).backward()
            running += loss.item()
            n_replay += int(use_replay)
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        n_log += 1

        if step % tcfg.log_interval == 0:
            dt = time.time() - t0
            eta_m = (tcfg.max_steps - step) * (dt / max(n_log, 1)) / 60
            print(f"step {step:5d}/{tcfg.max_steps} | loss {running/n_log:.4f} | "
                  f"lr {lr:.2e} | gnorm {float(gnorm):.2f} | replay {n_replay} | "
                  f"{dt/max(n_log,1):.2f}s/step | ETA {eta_m:.1f}m", flush=True)
            running, n_log, t0 = 0.0, 0, time.time()

        if step > 0 and step % tcfg.ckpt_interval == 0:
            p = save_checkpoint(ckpt_dir / "sft_last.pt", model=model,
                                optimizer=opt, model_cfg=mcfg, train_cfg=tcfg,
                                step=step, best_val=0.0, extra={"prefix_lm": args.prefix_lm})
            print(f"     ⤓ checkpoint → {p.name} (mirrored to Drive)", flush=True)

    save_checkpoint(ckpt_dir / "sft_final.pt", model=model, optimizer=opt,
                    model_cfg=mcfg, train_cfg=tcfg, step=tcfg.max_steps - 1, best_val=0.0,
                    extra={"prefix_lm": args.prefix_lm})
    print(f"SFT done → sft_final.pt ({time.time()-run_t0:.0f}s, {n_replay} replay steps)", flush=True)


if __name__ == "__main__":
    main()
