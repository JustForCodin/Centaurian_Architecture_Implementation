#!/usr/bin/env python3
"""
LoRA fine-tuning for CHA Experiment 2 (§4.4 of the plan).

QLoRA: 4-bit NF4 base + BF16 LoRA adapters on Qwen2.5-7B-Instruct.
Loss is computed only on the final assistant turn (the target response to
the probe), not on the SCI system prompt or conversation history.

Usage (in Colab, after Cell 6 has produced the splits):
  !python3 train_lora_sci.py \\
      --train-path data/lora_train.jsonl \\
      --val-path data/lora_val.jsonl \\
      --output-dir adapters/lora_10k

Optional:
  --max-train-examples N   subset for LoRA-2K / LoRA-5K data-scaling runs
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# DataCollatorForCompletionOnlyLM has moved across TRL versions:
#   - older: top-level `trl`
#   - mid:   `trl.trainer.utils`
#   - latest: removed from top-level (deprecated)
# Try the stable paths, then fall back to a minimal inline implementation.
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    try:
        from trl.trainer.utils import DataCollatorForCompletionOnlyLM
    except ImportError:
        from transformers import DataCollatorForLanguageModeling

        class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
            """Mask labels before the last occurrence of response_template tokens.

            Drop-in replacement for the (older) trl.DataCollatorForCompletionOnlyLM."""
            def __init__(self, response_template, tokenizer, *args, **kwargs):
                kwargs.setdefault("mlm", False)
                super().__init__(tokenizer=tokenizer, *args, **kwargs)
                self.response_template_ids = tokenizer.encode(
                    response_template, add_special_tokens=False
                )

            def torch_call(self, examples):
                batch = super().torch_call(examples)
                tmpl = self.response_template_ids
                tlen = len(tmpl)
                for i in range(batch["labels"].size(0)):
                    ids = batch["input_ids"][i].tolist()
                    start = None
                    # Find LAST occurrence of the response template
                    for j in range(len(ids) - tlen, -1, -1):
                        if ids[j:j + tlen] == tmpl:
                            start = j + tlen
                            break
                    if start is not None:
                        batch["labels"][i, :start] = -100
                    else:
                        # Template missing — mask the whole example so it doesn't bias training
                        batch["labels"][i, :] = -100
                return batch


# ── Constants ────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Qwen2.5 chat template — the assistant turn is delimited by these tokens.
# DataCollatorForCompletionOnlyLM finds the LAST occurrence of this template
# and masks everything before it, so loss falls only on the target response.
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


# ── Data loading ─────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def record_to_messages(record: dict) -> list[dict]:
    """Convert a dataset record to a chat-template messages list.

    System prompt + alternating user/assistant conversation +
    the probe (final user turn, already in record['conversation']) +
    the target (final assistant turn = the supervised label)."""
    messages = [{"role": "system", "content": record["system"]}]
    messages.extend(record["conversation"])  # ends with the probe (user role)
    messages.append({"role": "assistant", "content": record["target"]})
    return messages


def build_dataset(records: list[dict], tokenizer, max_seq_length: int) -> Dataset:
    """Render messages → text via chat template, then drop over-length examples.

    Filtering (rather than truncating) is intentional: truncating would either
    cut the SCI system prompt (breaking persona grounding) or cut the
    conversation tail (breaking the probe→target alignment). The relaxed parser
    that produced full.jsonl allows histories up to 50 turns, so a small
    fraction of examples genuinely exceed 2,048 tokens."""
    rows, dropped = [], 0
    for r in records:
        text = tokenizer.apply_chat_template(
            record_to_messages(r), tokenize=False, add_generation_prompt=False
        )
        # Quick length check — count tokens once
        token_count = len(tokenizer(text, add_special_tokens=False).input_ids)
        if token_count > max_seq_length:
            dropped += 1
            continue
        rows.append({"text": text})
    print(f"  built {len(rows)} examples, dropped {dropped} over {max_seq_length} tokens "
          f"({dropped / (len(rows) + dropped) * 100:.1f}%)")
    return Dataset.from_list(rows)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", required=True)
    p.add_argument("--val-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-train-examples", type=int, default=None,
                   help="Subset training data (for LoRA-2K / LoRA-5K runs)")
    # LoRA hyperparameters (defaults from plan §4.4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj")
    # Training schedule
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--max-seq-length", type=int, default=3072,
                   help="Hard cap on tokens per example. Records above this are dropped during "
                        "dataset build (truncating would corrupt the SCI grounding or probe alignment). "
                        "Plan §4.4 specified 2048; raised to 3072 because the relaxed parser produced "
                        "many >40-turn examples that exceed 2048. Bump to 4096 if VRAM allows and you "
                        "want to recover more of the long-context training signal.")
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ───────────────────────────────────────────────────────
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # SFT convention

    # ── Data ────────────────────────────────────────────────────────────
    print(f"Loading data...")
    train_records = load_jsonl(args.train_path)
    val_records = load_jsonl(args.val_path)
    if args.max_train_examples is not None:
        train_records = train_records[:args.max_train_examples]
    print(f"  train records: {len(train_records)}")
    print(f"  val records:   {len(val_records)}")

    print("Building train dataset...")
    train_ds = build_dataset(train_records, tokenizer, args.max_seq_length)
    print("Building val dataset...")
    val_ds = build_dataset(val_records, tokenizer, args.max_seq_length)

    # ── Base model in 4-bit NF4 ─────────────────────────────────────────
    print(f"\nLoading base model {MODEL_NAME} in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # required for gradient checkpointing
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ── LoRA config ─────────────────────────────────────────────────────
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(f"\nLoRA: r={args.lora_r} alpha={args.lora_alpha} "
          f"dropout={args.lora_dropout} targets={target_modules}")

    # ── Loss-masking collator (loss only on target assistant turn) ──────
    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    # ── Training config ─────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",  # memory-efficient on L4
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=2,
        # Note: max_seq_length / packing / dataset_text_field were removed from
        # SFTConfig in TRL v0.16+. We enforce the length limit during dataset
        # build (build_dataset filters over-length records) so the trainer
        # never sees them, and we let it auto-detect the text column.
    )

    print(f"\nEffective batch size: {args.per_device_batch_size * args.grad_accum_steps}")
    print(f"Total optimizer steps (approx): "
          f"{len(train_ds) * args.num_epochs // (args.per_device_batch_size * args.grad_accum_steps)}")

    # ── Train ───────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        args=sft_config,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # Auto-resume from latest checkpoint if any
    resume = any(out_dir.glob("checkpoint-*"))
    print(f"\n{'Resuming' if resume else 'Starting'} training...")
    trainer.train(resume_from_checkpoint=resume)

    # ── Save final adapter ──────────────────────────────────────────────
    print(f"\nSaving adapter to {out_dir}...")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Quick metrics summary
    metrics = trainer.evaluate()
    print(f"\nFinal eval_loss: {metrics.get('eval_loss', 'n/a')}")
    print(f"Adapter saved to: {out_dir}")


if __name__ == "__main__":
    main()
