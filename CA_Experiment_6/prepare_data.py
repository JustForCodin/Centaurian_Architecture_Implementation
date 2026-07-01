#!/usr/bin/env python3
"""Prepare Stage-A pretrain shards and Stage-B free reading-QA data (plan §4.2).

Subcommands
-----------
pretrain    Tokenize a free corpus (FineWeb-Edu slice / Wikipedia text) into
            uint16 .bin shards (train.bin + val.bin) for Stage-A.
qa-sft      Reformat SQuAD 2.0 (backbone; its *unanswerable* questions supervise
            H2 abstention) + optional NQ/MS MARCO/TriviaQA/HotpotQA into §4.3
            records → data/qa_sft.jsonl.
eval        Build data/eval_answerable.jsonl + data/eval_unanswerable.jsonl from
            the SQuAD 2.0 validation split (held out; EM/F1 comparable).

`datasets` is imported lazily so the module loads without it (e.g. in the local
.venv smoke test). Use --dry-run on qa-sft/eval to emit a tiny synthetic sample
with no network (pipeline sanity only).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from ca_assets import (
    make_record, validate_record, ABSTENTION_CANONICAL, ADA_SCI,
)


# ── Stage-A: corpus → uint16 bins ────────────────────────────────────────

def cmd_pretrain(args):
    from tokenizer_util import ADATokenizer
    tok = ADATokenizer.load(args.tokenizer)

    def text_iter():
        if args.hf_dataset:
            from datasets import load_dataset
            ds = load_dataset(args.hf_dataset, name=args.hf_config, split=args.split,
                              streaming=True)
            for i, ex in enumerate(ds):
                if args.max_docs and i >= args.max_docs:
                    break
                yield ex[args.text_field]
        else:
            for p in sorted(Path(args.input_dir).glob("*.txt")):
                yield p.read_text(encoding="utf-8", errors="ignore")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stream_path = out_dir / "_stream.bin"
    if stream_path.exists():
        stream_path.unlink()                  # fresh run (avoid appending to a stale stream)
    print(f"pretrain: tokenizing → {out_dir}  (target {args.max_tokens or 'all'} tokens)", flush=True)
    all_ids: list[int] = []
    n_tokens = n_docs = 0
    t0 = time.time()
    for doc in text_iter():
        ids = tok.encode(doc, add_eos=True)   # EOS delimits documents
        all_ids.extend(ids)
        n_tokens += len(ids)
        n_docs += 1
        if n_docs % 20_000 == 0:
            rate = n_tokens / max(time.time() - t0, 1e-9)
            print(f"  {n_docs:,} docs | {n_tokens/1e6:.1f}M tokens | {rate/1e3:.0f}k tok/s", flush=True)
        if args.max_tokens and n_tokens >= args.max_tokens:
            break
        if len(all_ids) >= 50_000_000:        # flush periodically to keep RAM bounded
            _append_bin(stream_path, all_ids); all_ids = []
    if all_ids:
        _append_bin(stream_path, all_ids)

    # Split 99.5/0.5 into train/val
    stream = np.memmap(stream_path, dtype=np.uint16, mode="r")
    n = len(stream)
    n_val = max(args.val_tokens, int(n * 0.005))
    stream[:n - n_val].tofile(out_dir / "train.bin")
    stream[n - n_val:].tofile(out_dir / "val.bin")
    del stream
    stream_path.unlink()
    print(f"pretrain: {n_docs:,} docs, {n:,} tokens → train {n-n_val:,} / val {n_val:,} "
          f"({time.time()-t0:.0f}s)", flush=True)


def _append_bin(path: Path, ids: list[int]):
    arr = np.array(ids, dtype=np.uint16)
    with open(path, "ab") as f:
        arr.tofile(f)


# ── Stage-B free reading QA (SQuAD2 backbone) ────────────────────────────

def _squad2_to_records(split: str, start_id: int, limit: int | None,
                       include_gold: bool = False):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split=split)
    recs = []
    rid = start_id
    for ex in ds:
        if limit and len(recs) >= limit:
            break
        gold = list(ex["answers"]["text"])
        answerable = len(gold) > 0
        if answerable:
            assistant = f"{gold[0]} (from the retrieved passage)"
        else:
            assistant = ABSTENTION_CANONICAL
        rec = make_record(
            rid, source="squad2", answerable=answerable,
            context=ex["context"],
            messages=[
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": assistant},
            ],
        )
        if include_gold:
            rec["gold_answers"] = gold          # eval-only; for SQuAD EM/F1 (§6.1)
        recs.append(rec)
        rid += 1
    return recs


def _synthetic_records(n: int, start_id: int):
    """Tiny offline sample (no network) for pipeline sanity — §4.3 shaped."""
    recs = []
    facts = [
        ("What is the melting point of tungsten?",
         "Tungsten has a melting point of 3422 degrees Celsius, the highest of all metals.",
         "Tungsten melts at 3422 degrees Celsius (from the retrieved passage)."),
        ("What is Planck's constant?",
         "Planck's constant is approximately 6.626 x 10^-34 joule-seconds.",
         "Planck's constant is about 6.626 x 10^-34 J*s (from the retrieved passage)."),
    ]
    rid = start_id
    for i in range(n):
        if i % 3 == 2:  # every third is unanswerable
            recs.append(make_record(
                rid, "squad2", False,
                context="The Eiffel Tower is a wrought-iron lattice tower in Paris.",
                messages=[{"role": "user", "content": "What is the boiling point of nitrogen?"},
                          {"role": "assistant", "content": ABSTENTION_CANONICAL}]))
        else:
            q, ctx, a = facts[i % len(facts)]
            r = make_record(
                rid, "squad2", True, context=ctx,
                messages=[{"role": "user", "content": q},
                          {"role": "assistant", "content": a}])
            r["gold_answers"] = [a.split(" (from")[0]]
            recs.append(r)
        rid += 1
    return recs


def cmd_qa_sft(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        recs = _synthetic_records(args.limit or 12, 0)
    else:
        recs = _squad2_to_records("train", 0, args.limit)
    for r in recs:
        validate_record(r)
    with out.open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    n_ans = sum(r["answerable"] for r in recs)
    print(f"qa-sft: {len(recs)} records → {out}  (answerable {n_ans}, "
          f"unanswerable {len(recs)-n_ans})")


def cmd_eval(args):
    out_a = Path(args.out_answerable)
    out_u = Path(args.out_unanswerable)
    out_a.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        recs = _synthetic_records(24, 100000)
    else:
        recs = _squad2_to_records("validation", 100000, args.limit, include_gold=True)
    ans = [r for r in recs if r["answerable"]][:args.n]
    una = [r for r in recs if not r["answerable"]][:args.n]
    for r in ans + una:
        validate_record(r)
    with out_a.open("w") as f:
        for r in ans:
            f.write(json.dumps(r) + "\n")
    with out_u.open("w") as f:
        for r in una:
            f.write(json.dumps(r) + "\n")
    print(f"eval: {len(ans)} answerable → {out_a}; {len(una)} unanswerable → {out_u}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pretrain")
    p.add_argument("--tokenizer", default="tokenizer/ada_bpe.json")
    p.add_argument("--hf-dataset", default=None,
                   help="e.g. HuggingFaceFW/fineweb-edu or wikimedia/wikipedia")
    p.add_argument("--hf-config", default=None, help="e.g. sample-10BT / 20231101.en")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--input-dir", default=None, help="local *.txt dir (offline)")
    p.add_argument("--out-dir", default="data/pretrain")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--val-tokens", type=int, default=1_000_000)
    p.set_defaults(func=cmd_pretrain)

    q = sub.add_parser("qa-sft")
    q.add_argument("--out", default="data/qa_sft.jsonl")
    q.add_argument("--limit", type=int, default=None)
    q.add_argument("--dry-run", action="store_true")
    q.set_defaults(func=cmd_qa_sft)

    e = sub.add_parser("eval")
    e.add_argument("--out-answerable", default="data/eval_answerable.jsonl")
    e.add_argument("--out-unanswerable", default="data/eval_unanswerable.jsonl")
    e.add_argument("--n", type=int, default=200)
    e.add_argument("--limit", type=int, default=20000)
    e.add_argument("--dry-run", action="store_true")
    e.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
