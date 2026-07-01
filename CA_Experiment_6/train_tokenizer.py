#!/usr/bin/env python3
"""Train the from-scratch ADA BPE tokenizer (plan §5.1: 16k vocab + specials).

Byte-level BPE over the Stage-A corpus, with the ADA chat special tokens
registered as atomic units so the SFT template tokenizes deterministically.

Usage:
    python train_tokenizer.py --input data/pretrain/corpus_sample.txt \
        --vocab-size 16000 --out tokenizer/ada_bpe.json
    # or stream a directory of .txt shards:
    python train_tokenizer.py --input-dir data/pretrain --out tokenizer/ada_bpe.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ca_assets import SPECIAL_TOKENS


def iter_text_files(paths: list[Path]):
    for p in paths:
        yield str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="*", default=[], help="text file(s)")
    ap.add_argument("--input-dir", default=None, help="dir of *.txt shards")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", default="tokenizer/ada_bpe.json")
    args = ap.parse_args()

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    files = [Path(p) for p in args.input]
    if args.input_dir:
        files += sorted(Path(args.input_dir).glob("*.txt"))
    if not files:
        raise SystemExit("no input files; pass --input or --input-dir")
    print(f"Training BPE on {len(files)} file(s) → vocab {args.vocab_size}")

    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,          # reserved at low ids
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tk.train([str(f) for f in files], trainer)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tk.save(str(out))
    print(f"Saved tokenizer → {out}  (vocab_size={tk.get_vocab_size()})")
    for t in SPECIAL_TOKENS:
        print(f"  {t:16s} -> id {tk.token_to_id(t)}")


if __name__ == "__main__":
    main()
