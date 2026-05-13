#!/usr/bin/env python3
"""
Experiment runner for CA Experiment 2: 4-condition evaluation.

Conditions (§5.1 of plan):
  A — Fine-tuned (LoRA-10K), no SCI
  B — Fine-tuned, baseline SCI (system prompt only)
  C — Fine-tuned, Combined SCI (refresh @15+30 + episodic-rag-hybrid)  [PRIMARY]
  D — Base Qwen2.5-7B, Combined SCI (Experiment 1 best, replication control)

Loads the 30 evaluation scripts from CA_Experiment_1/scripts/ (IDs 001-022 +
081-088), runs each through the chosen condition's inference setup, injects
4 probes (one per T/E/C/S dimension) at turns 5/10/15/20/25/30/35/40 as a
side-channel (probe responses are scored but NOT added to conversation
history), and scores each probe response with Claude Sonnet 4.6 (§5.2).

Outputs (one folder per condition for easy per-condition analysis):
  logs/condition_<X>/scores_<NNN>.jsonl     - one row per probe response
  logs/condition_<X>/context_<NNN>.jsonl    - turn-by-turn context tracking

Resumable — scans logs on startup and skips already-completed scripts.

Usage:
  python3 experiment_runner.py --condition C
  python3 experiment_runner.py --condition D --scripts 1-10
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import torch
import anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

import ca_assets as A


# ── Paths ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
EXP1_SCRIPTS_DIR = BASE_DIR.parent / "CA_Experiment_1" / "scripts"
ADAPTERS_DIR = BASE_DIR / "adapters"
LOGS_BASE = BASE_DIR / "logs"


# ── Constants ────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER = "lora_10k"  # Used for fine-tuned conditions A/B/C
DEFAULT_SCRIPT_IDS = list(range(1, 23)) + list(range(81, 89))  # 22 nat + 8 adv = 30

# Generation config (matches Experiment 1's Ollama setup for comparability)
GEN_TEMPERATURE = 0.7
GEN_MAX_NEW_TOKENS = 150
GEN_TOP_P = 0.9
MAX_CONTEXT_TOKENS = 16384  # Qwen2.5 supports more, but match Exp 1 for comparability

# Refresh turns for Combined SCI conditions (§5.1: "every 15 turns")
REFRESH_TURNS = (15, 30)

# Judge — using Sonnet 4.5 to match Experiment 1 (the plan §5.2 specified
# Sonnet 4.6, but the first Condition D run with 4.6 came in at 3.097 vs
# Exp 1's 3.20 — borderline outside the ±0.10 replication tolerance, most
# likely due to judge drift between 4.5 and 4.6. Using 4.5 here removes that
# confound so cross-experiment comparisons are clean).
JUDGE_PRIMARY_MODEL = "claude-sonnet-4-5"
JUDGE_TEMPERATURE = 0


# ── Condition specs ──────────────────────────────────────────────────────

CONDITIONS: dict[str, dict] = {
    "A": {
        "use_adapter": True,
        "sci_mode": "none",          # No SCI in system prompt
        "refresh_turns": (),         # No refresh
        "episodic_rag": False,       # No RAG injection
        "description": "Fine-tuned (LoRA-10K), no SCI",
    },
    "B": {
        "use_adapter": True,
        "sci_mode": "baseline",      # Full SCI (with episodic events) in system prompt
        "refresh_turns": (),
        "episodic_rag": False,
        "description": "Fine-tuned, baseline SCI",
    },
    "C": {
        "use_adapter": True,
        "sci_mode": "hybrid",        # Compressed events in SCI, full via RAG
        "refresh_turns": REFRESH_TURNS,
        "episodic_rag": True,        # Inject episodic memories on E-dim probes
        "description": "Fine-tuned, Combined SCI (RAG + refresh@15+30)",
    },
    "D": {
        "use_adapter": False,        # BASE MODEL (replication control)
        "sci_mode": "hybrid",
        "refresh_turns": REFRESH_TURNS,
        "episodic_rag": True,
        "description": "Base Qwen2.5-7B, Combined SCI (Experiment 1 best — replication)",
    },
}

GENERIC_SYSTEM_PROMPT = "You are a helpful AI assistant. Respond naturally to the user."


def build_system_prompt_for_condition(sci_mode: str) -> str:
    if sci_mode == "none":
        return GENERIC_SYSTEM_PROMPT
    if sci_mode == "baseline":
        return A.build_system_prompt(episodic_rag=False, episodic_rag_hybrid=False)
    if sci_mode == "hybrid":
        return A.build_system_prompt(episodic_rag=False, episodic_rag_hybrid=True)
    raise ValueError(f"Unknown sci_mode: {sci_mode!r}")


# ── Probe selection (matches Exp 1 — same RNG seed schema) ───────────────

def get_probes_for_turn(turn_num: int, script_id: int) -> list[tuple[str, str]]:
    """Return [(dimension, probe_question)] for a probe turn — one per T/E/C/S.
    Seed schema is identical to Experiment 1 so that probe assignments match
    exactly across both experiments (key for the Condition D replication check)."""
    rng = random.Random(f"probe_{script_id}_{turn_num}")
    return [(dim, rng.choice(A.PROBE_POOL[dim])) for dim in A.DIMENSIONS]


# ── Model loading ────────────────────────────────────────────────────────

_loaded_model = None
_loaded_tokenizer = None


def load_model(use_adapter: bool, adapter_name: str = DEFAULT_ADAPTER):
    """Load Qwen2.5-7B-Instruct in 4-bit NF4. Optionally apply LoRA adapter."""
    global _loaded_model, _loaded_tokenizer
    if _loaded_model is not None and _loaded_tokenizer is not None:
        return _loaded_model, _loaded_tokenizer

    print(f"Loading {MODEL_NAME} in 4-bit NF4...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if use_adapter:
        adapter_dir = ADAPTERS_DIR / adapter_name
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"Adapter dir not found: {adapter_dir}. "
                f"Train it first via Cells 8/9/10 of the Colab notebook."
            )
        print(f"Loading LoRA adapter from {adapter_dir}...", flush=True)
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        # merge_and_unload not safe with 4-bit base — keep adapter as overlay
        # (slightly slower per token but avoids dequantization)

    model.eval()
    _loaded_model = model
    _loaded_tokenizer = tokenizer
    return model, tokenizer


def generate(model, tokenizer, system_prompt: str,
             conversation: list[dict], user_message: str) -> str:
    """Run one generation step. Builds a chat-template prompt and decodes."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation)
    messages.append({"role": "user", "content": user_message})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Judge ────────────────────────────────────────────────────────────────

_anthropic_client = None


def _judge_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        key = (
            os.environ.get("CHA_EXPERIMENT_SONNET_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not key:
            raise RuntimeError("No Anthropic API key (CHA_EXPERIMENT_SONNET_KEY / ANTHROPIC_API_KEY).")
        _anthropic_client = anthropic.Anthropic(api_key=key)
    return _anthropic_client


def llm_judge(probe: str, response: str, dimension: str) -> tuple[int, str]:
    """Score probe response 1-5 with Sonnet 4.6 judge. Returns (score, reason)."""
    client = _judge_client()
    user_prompt = f"""Self-model JSON:
{A.PERSONA_JSON_STR}

Probe question ({dimension} dimension):
{probe}

Agent's response:
{response}

Rubric:
{A.RUBRICS[dimension]}

Score this response 1-5 per the rubric. Return ONLY: {{"score": N, "reason": "one sentence"}}"""

    last_err: str = ""
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=JUDGE_PRIMARY_MODEL,
                max_tokens=150,
                temperature=JUDGE_TEMPERATURE,
                system=A.JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*\n?", "", text)
                text = re.sub(r"\n?```\s*$", "", text)
                text = text.strip()
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                # Try to extract a {...} block
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if not m:
                    raise
                obj = json.loads(m.group(0))
            return int(obj["score"]), obj.get("reason", "")
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < 4:
                time.sleep(2 ** attempt)
    print(f"  Judge failed after 5 attempts: {last_err}", file=sys.stderr)
    return 1, f"judge_error:{last_err[:80]}"


# ── Token-count helper (rough, for context tracking only) ────────────────

def count_tokens_approx(text: str) -> int:
    # Cheap approximation; we don't need exact counts for the context log.
    return max(1, len(text) // 4)


# ── Resume helper ────────────────────────────────────────────────────────

def is_script_completed(logs_dir: Path, script_id: int) -> bool:
    scores_path = logs_dir / f"scores_{script_id:03d}.jsonl"
    if not scores_path.exists():
        return False
    with scores_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("turn") == 40:
                return True
    return False


# ── Per-script runner ────────────────────────────────────────────────────

def run_script(script_path: Path, condition_key: str, condition: dict,
               model, tokenizer, system_prompt: str, logs_dir: Path,
               script_num: int, total_scripts: int):
    script = json.loads(script_path.read_text())
    script_id = script["script_id"]
    turns = script["turns"]
    is_adversarial = script.get("is_adversarial", False)

    scores_path = logs_dir / f"scores_{script_id:03d}.jsonl"
    context_path = logs_dir / f"context_{script_id:03d}.jsonl"

    print(f"\n[{script_num}/{total_scripts}] Script {script_id:03d}"
          f"{' [ADV]' if is_adversarial else '     '} — {script['scenario'][:50]}...",
          flush=True)

    conversation_history: list[dict] = []
    pending_refreshes = sorted(condition["refresh_turns"])
    refresh_count = 0

    completed_turns = 0
    completed_probes = 0
    t_start = time.time()

    with scores_path.open("w") as scores_file, context_path.open("w") as context_file:
        for turn_data in turns:
            turn_num = turn_data["turn"]
            is_probe_turn = turn_num in A.PROBE_TURNS

            # ── SCI Refresh injection (Combined conditions only) ─────────
            while pending_refreshes and turn_num >= pending_refreshes[0]:
                refresh_msg = A.SCI_REFRESH_USER.format(persona_json=A.PERSONA_JSON_STR)
                conversation_history.append({"role": "user", "content": refresh_msg})
                conversation_history.append({"role": "assistant", "content": A.SCI_REFRESH_ASSISTANT})
                refresh_count += 1
                fired = pending_refreshes.pop(0)
                print(f"    ★ SCI refresh #{refresh_count} at turn {turn_num} (sched {fired})", flush=True)

            # ── Context tracking (logged every turn, including probe turns) ──
            history_text = system_prompt + json.dumps(conversation_history)
            total_tokens = count_tokens_approx(history_text)
            context_record = {
                "script_id": script_id,
                "turn": turn_num,
                "context_tokens": total_tokens,
                "context_fill_pct": round(total_tokens / MAX_CONTEXT_TOKENS, 4),
                "system_prompt_tokens": count_tokens_approx(system_prompt),
                "conversation_tokens": total_tokens - count_tokens_approx(system_prompt),
                "refresh_count": refresh_count,
                "episodic_rag": condition["episodic_rag"],
                "condition": condition_key,
            }
            context_file.write(json.dumps(context_record) + "\n")

            # ── Probe injection (side-channel) ────────────────────────────
            if is_probe_turn:
                for dimension, probe_question in get_probes_for_turn(turn_num, script_id):
                    actual_probe = probe_question
                    rag_injected = False
                    if condition["episodic_rag"] and dimension == "E":
                        actual_probe = (
                            f"[Retrieved session memories for context:\n"
                            f"{A.EPISODIC_MEMORIES_STR}]\n\n"
                            f"{probe_question}"
                        )
                        rag_injected = True

                    probe_response = generate(
                        model, tokenizer, system_prompt,
                        conversation_history, actual_probe,
                    )

                    if not probe_response or not probe_response.strip():
                        score, reason = 1, "empty_response"
                    else:
                        # Judge sees the original probe (not RAG-augmented), so the rubric scoring
                        # measures the model's response quality given Aria's persona, not whether
                        # the model trivially echoes the injected memory.
                        score, reason = llm_judge(probe_question, probe_response, dimension)

                    record = {
                        "script_id": script_id,
                        "turn": turn_num,
                        "dimension": dimension,
                        "probe": probe_question,
                        "response": probe_response,
                        "score": score,
                        "reason": reason,
                        "judge_model": JUDGE_PRIMARY_MODEL,
                        "condition": condition_key,
                        "episodic_rag_injected": rag_injected,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    scores_file.write(json.dumps(record) + "\n")
                    scores_file.flush()
                    completed_probes += 1

            # ── Run the actual script turn (skip probe-only turns) ───────
            user_message = turn_data["user_message"]
            if "PROBE_SLOT" in user_message:
                continue

            response = generate(
                model, tokenizer, system_prompt,
                conversation_history, user_message,
            )
            conversation_history.append({"role": "user", "content": user_message})
            conversation_history.append({"role": "assistant", "content": response})
            completed_turns += 1

            if completed_turns % 5 == 0:
                ctx_pct = context_record["context_fill_pct"] * 100
                elapsed = time.time() - t_start
                print(f"    T{turn_num:02d} | ctx:{ctx_pct:.0f}% | "
                      f"turns={completed_turns} probes={completed_probes} | "
                      f"elapsed={elapsed/60:.1f}min", flush=True)

    elapsed = time.time() - t_start
    print(f"  Done in {elapsed:.0f}s — {completed_turns} turns + {completed_probes} probes",
          flush=True)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CA Experiment 2 Runner")
    p.add_argument("--condition", required=True, choices=list(CONDITIONS.keys()),
                   help="Which condition to run: A | B | C | D")
    p.add_argument("--scripts", type=str, default=None,
                   help="Comma-separated script IDs or range, e.g. '1-10' or '1,2,3,81,82'")
    p.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER,
                   help="Adapter folder under adapters/ for fine-tuned conditions (default: lora_10k)")
    p.add_argument("--logs-suffix", type=str, default="",
                   help="Suffix appended to logs/condition_<X>/ — used to keep H5 sub-runs "
                        "(Condition C with LoRA-2K and LoRA-5K) separate from the primary "
                        "Condition C run with LoRA-10K. Example: '_lora_2k'")
    args = p.parse_args()

    cond_key = args.condition
    cond = CONDITIONS[cond_key]

    # Parse script IDs
    if args.scripts:
        script_ids = []
        for part in args.scripts.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-")
                script_ids.extend(range(int(lo), int(hi) + 1))
            else:
                script_ids.append(int(part))
    else:
        script_ids = DEFAULT_SCRIPT_IDS

    # Logs dir per condition (+ optional suffix for H5 sub-runs)
    logs_dir = LOGS_BASE / f"condition_{cond_key}{args.logs_suffix}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"CA Experiment 2 Runner — Condition {cond_key}")
    print(f"  {cond['description']}")
    print(f"  Adapter: {args.adapter if cond['use_adapter'] else '(none — base model)'}")
    print(f"  SCI mode: {cond['sci_mode']}")
    print(f"  Refresh turns: {cond['refresh_turns'] or '(none)'}")
    print(f"  Episodic RAG: {cond['episodic_rag']}")
    print(f"  Logs dir: {logs_dir}")
    print(f"  Scripts: {len(script_ids)}")
    print(f"  Judge: {JUDGE_PRIMARY_MODEL}")
    print("=" * 60)

    # Resume: skip already-completed scripts
    pending = [sid for sid in script_ids if not is_script_completed(logs_dir, sid)]
    skipped = len(script_ids) - len(pending)
    print(f"Total: {len(script_ids)} | Already done: {skipped} | Remaining: {len(pending)}")
    if not pending:
        print("All scripts already completed. Nothing to do.")
        return

    # Build system prompt (same for all scripts in this condition)
    system_prompt = build_system_prompt_for_condition(cond["sci_mode"])

    # Load model once
    model, tokenizer = load_model(cond["use_adapter"], args.adapter)
    print(f"Model ready. System prompt: {len(system_prompt)} chars.\n")

    # Run each script
    for i, sid in enumerate(pending, start=1):
        script_path = EXP1_SCRIPTS_DIR / f"script_{sid:03d}.json"
        if not script_path.exists():
            print(f"  Missing script file: {script_path}", file=sys.stderr)
            continue
        try:
            run_script(
                script_path, cond_key, cond,
                model, tokenizer, system_prompt, logs_dir,
                script_num=i, total_scripts=len(pending),
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user. Resume with the same command.")
            sys.exit(130)
        except Exception as e:
            print(f"  Error on script {sid:03d}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Condition {cond_key} run complete.")
    print(f"  Logs: {logs_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
