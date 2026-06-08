#!/usr/bin/env python3
"""
Experiment runner for CA Experiment 4: QPM→SLM Interface Richness Ablation.

One battery (PersonaScore), four interface conditions:
  A — Marginals only                (Exp 3 QPM replication, continuity check)
  B — Marginals + purity / ambivalence field
  C — Coherence-conditional speech-act modifier
  D — Marginals + purity + bivariate co-activations

Usage:
  python3 experiment_runner.py --condition A
  python3 experiment_runner.py --condition B --scripts 1-10
  python3 experiment_runner.py --condition C --adapter lora_10k
  python3 experiment_runner.py --condition D
  python3 experiment_runner.py --reliability        # 5% of Condition A, T=0 re-judge

Outputs are written to logs/condition_<X>_psychotherapy/ as JSONL files.
Resumable — completed (script, condition) pairs are skipped on re-start.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# ── Quiet down the noisy deps before anything that touches torch/bnb ─────
# Suppress the bitsandbytes _check_is_size FutureWarning + a few other
# noisy deprecation messages that pollute the experiment log.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        module=r"bitsandbytes(\..*)?")
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module=r"transformers(\..*)?")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
for _name in ("transformers", "peft", "bitsandbytes", "accelerate"):
    logging.getLogger(_name).setLevel(logging.ERROR)

import numpy as np

try:
    from dotenv import load_dotenv
    # Own folder first — every CA_Experiment_N/ now ships its own .env
    # (copy of CA_Experiment_1/.env, same key).  Exp 1 stays as a safety
    # net in case someone forgets to copy it when seeding a new folder.
    load_dotenv(Path(__file__).parent / ".env")
    load_dotenv(Path(__file__).parent.parent / "CA_Experiment_1" / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))
import ca_assets as A
from qpm import QPM

# ── Paths ─────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
EXP1_SCRIPTS_DIR = BASE_DIR.parent / "CA_Experiment_1" / "scripts"
EXP2_ADAPTERS_DIR = BASE_DIR.parent / "CA_Experiment_2" / "adapters"
LOGS_BASE = BASE_DIR / "logs"

# ── Constants ─────────────────────────────────────────────────────────────

N_SHOTS = 1024
JUDGE_MODEL = "claude-sonnet-4-5"
JUDGE_TEMPERATURE = 0

GEN_TEMPERATURE = 0.7
GEN_MAX_NEW_TOKENS = 150
GEN_TOP_P = 0.9
MAX_CONTEXT_TOKENS = 16384

REFRESH_TURNS = (15, 30)
DEFAULT_SCRIPT_IDS = list(range(1, 23)) + list(range(81, 89))  # 22 nat + 8 adv

PROFILE_NAME = "psychotherapy"  # Exp 4 primary (and only) profile per plan §5.1

# Reliability sub-mode: 5% of Condition A → 48 probes re-judged at T=0 (plan §5.4)
RELIABILITY_SAMPLE_FRACTION = 0.05
RELIABILITY_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────

def count_tokens_approx(text: str) -> int:
    return max(1, len(text) // 4)


def _condition_logs_dir(condition: str) -> Path:
    return LOGS_BASE / f"condition_{condition.lower()}_{PROFILE_NAME}"


def _is_script_done(logs_dir: Path, condition: str, script_id: int) -> bool:
    path = logs_dir / f"scores_condition_{condition.lower()}_{script_id:03d}.jsonl"
    if not path.exists():
        return False
    with path.open() as f:
        for line in f:
            try:
                if json.loads(line).get("turn") == 40:
                    return True
            except json.JSONDecodeError:
                pass
    return False


# ── SLM loader (Qwen2.5-7B + LoRA-10K) ───────────────────────────────────

def _load_slm(adapter_name: str = "lora_10k"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path = EXP2_ADAPTERS_DIR / adapter_name
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at {adapter_path}. "
            "Run CA_Experiment_2 training first (Colab notebook, Cells 8/9/10)."
        )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    print(f"Loaded {model_name} + {adapter_name}", flush=True)
    return model, tokenizer


def _generate(model, tokenizer, system_prompt: str,
              conversation: list[dict], user_message: str) -> str:
    import torch
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

def _judge_client():
    import anthropic
    key = (
        os.environ.get("CHA_EXPERIMENT_SONNET_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "No Anthropic API key found. Set CHA_EXPERIMENT_SONNET_KEY or ANTHROPIC_API_KEY."
        )
    return anthropic.Anthropic(api_key=key)


def _llm_judge(client, probe: str, response: str, dimension: str,
               *, temperature: float = JUDGE_TEMPERATURE) -> tuple[int, str]:
    """Score a probe response 1-5 with Sonnet judge."""
    user_prompt = (
        f"Self-model JSON:\n{A.PERSONA_JSON_STR}\n\n"
        f"Probe question ({dimension} dimension):\n{probe}\n\n"
        f"Agent's response:\n{response}\n\n"
        f"Rubric:\n{A.RUBRICS[dimension]}\n\n"
        'Score this response 1-5 per the rubric. Return ONLY: {"score": N, "reason": "one sentence"}'
    )
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=150,
                temperature=temperature,
                system=A.JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = resp.content[0].text.strip()
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text).strip()
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if not m:
                    raise
                obj = json.loads(m.group(0))
            return int(obj["score"]), obj.get("reason", "")
        except Exception as exc:
            if attempt < 4:
                time.sleep(2 ** attempt)
            else:
                return 1, f"judge_error:{str(exc)[:80]}"
    return 1, "judge_error:max_retries"


# ── Per-turn personality computation ─────────────────────────────────────

def _qpm_intent(
    qpm: QPM, condition: str, d_vector: list[float], n_shots: int,
    qpm_cal: QPM | None = None,
) -> tuple[dict, dict]:
    """Run QPM and build the per-condition structured intent.

    Returns (qpm_result, structured_intent).  For Condition C, a second
    QPM run at CAL_SHOTS_C=8192 shots provides a low-noise ambivalence
    estimate for the firing threshold (plan Appendix rev 1); the standard
    1024-shot marginals still drive the personality_state JSON, keeping
    all four conditions consistent on their JSON content.
    """
    res = qpm.run(d_vector, n_shots=n_shots)
    # purity_approx in qpm.py == 1 - mean(p²+(1-p)²) == plan's ambivalence
    ambivalence = float(res["purity_approx"])
    purity_proxy = round(1.0 - ambivalence, 4)

    # Condition C: high-shot firing estimate (separate from JSON marginals)
    cal_ambivalence: float | None = None
    if condition == "C" and qpm_cal is not None:
        cal_res = qpm_cal.run(d_vector, n_shots=A.CAL_SHOTS_C)
        cal_ambivalence = float(cal_res["purity_approx"])

    intent = A.qpm_to_structured_intent(
        condition=condition,
        marginals=res["marginals"],
        d_vector=d_vector,
        purity_proxy=purity_proxy if condition in ("B", "C", "D") else None,
        cal_ambivalence=cal_ambivalence,
        counts=res["counts"] if condition == "D" else None,
    )
    # Attach audit metadata used by the context log even for A
    intent.setdefault("_audit", {})
    intent["_audit"]["ambivalence"] = round(ambivalence, 4)
    intent["_audit"]["purity_proxy"] = purity_proxy
    if cal_ambivalence is not None:
        intent["_audit"]["cal_ambivalence"] = round(cal_ambivalence, 4)
    return res, intent


# ── Battery C — main per-condition runner ────────────────────────────────

def run_condition(
    condition: str,
    *,
    script_ids: list[int] | None = None,
    n_shots: int = N_SHOTS,
    adapter_name: str = "lora_10k",
):
    """Run one interface condition across the 30-script bank."""
    if condition not in A.CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}")

    logs_dir = _condition_logs_dir(condition)
    logs_dir.mkdir(parents=True, exist_ok=True)

    ids = script_ids or DEFAULT_SCRIPT_IDS
    profile = A.PROFILES[PROFILE_NAME]
    qpm = QPM(profile, n_shots=n_shots)
    # Condition C only: second QPM instance at CAL_SHOTS_C for firing decision
    qpm_cal = QPM(profile, n_shots=A.CAL_SHOTS_C) if condition == "C" else None

    model, tokenizer = _load_slm(adapter_name)
    client = _judge_client()

    total = len(ids)
    cond_label = f"cond_{condition.lower()}"
    print(
        f"\n=== Battery C ({PROFILE_NAME}) — Condition {condition} "
        f"({A.CONDITION_DESCRIPTIONS[condition]}) — {total} scripts ===",
        flush=True,
    )

    for i, script_id in enumerate(ids):
        if _is_script_done(logs_dir, condition, script_id):
            print(f"  [{i+1}/{total}] Script {script_id:03d} — skipped (already done)",
                  flush=True)
            continue

        script_path = EXP1_SCRIPTS_DIR / f"script_{script_id:03d}.json"
        script = json.loads(script_path.read_text())
        turns = script["turns"]

        scores_path = logs_dir / f"scores_condition_{condition.lower()}_{script_id:03d}.jsonl"
        context_path = logs_dir / f"context_condition_{condition.lower()}_{script_id:03d}.jsonl"

        print(
            f"\n  [{i+1}/{total}] Script {script_id:03d} — "
            f"{script['scenario'][:50]}...",
            flush=True,
        )

        conversation: list[dict] = []
        pending_refreshes = list(REFRESH_TURNS)
        refresh_count = 0
        completed_turns = 0
        completed_probes = 0
        c_fire_count = 0  # # turns where Condition C directive actually fired
        t_start = time.time()

        # Initial neutral d-vector
        current_d = [0.5, 0.5, 0.5, 0.5, 0.3]
        res, intent = _qpm_intent(qpm, condition, current_d, n_shots, qpm_cal=qpm_cal)
        current_system_prompt = A.build_condition_system_prompt(condition, intent)

        with scores_path.open("w") as sf, context_path.open("w") as cf:
            for turn_data in turns:
                turn_num = turn_data["turn"]
                is_probe_turn = turn_num in A.PROBE_TURNS

                # SCI refresh injection (same as Exp 2/3 Combined SCI)
                while pending_refreshes and turn_num >= pending_refreshes[0]:
                    res_r, intent_r = _qpm_intent(qpm, condition, current_d, n_shots, qpm_cal=qpm_cal)
                    refresh_prompt = A.build_condition_system_prompt(condition, intent_r)
                    # Persona JSON snapshot for the refresh user-message —
                    # use the same per-condition persona body the system prompt
                    # is currently exposing (without the guidance/directive
                    # blocks) so the injection mirrors what the SLM "saw".
                    persona_snapshot = dict(A.PERSONA_JSON)
                    persona_snapshot["personality"] = intent_r["personality_state"]
                    if condition == "B":
                        persona_snapshot["cognitive_state"] = intent_r.get(
                            "cognitive_state", {}
                        )
                    elif condition == "D":
                        persona_snapshot["cognitive_state"] = {
                            k: v for k, v in intent_r.get("cognitive_state", {}).items()
                            if k in ("ambivalence", "purity_proxy")
                        }
                        persona_snapshot["trait_coactivation"] = intent_r.get(
                            "trait_coactivation", {}
                        )
                    conversation.append({
                        "role": "user",
                        "content": A.SCI_REFRESH_USER.format(
                            persona_json=json.dumps(persona_snapshot, indent=2)
                        ),
                    })
                    conversation.append({
                        "role": "assistant",
                        "content": A.SCI_REFRESH_ASSISTANT,
                    })
                    refresh_count += 1
                    current_system_prompt = refresh_prompt
                    res, intent = res_r, intent_r
                    fired = pending_refreshes.pop(0)
                    print(
                        f"    ★ refresh #{refresh_count} at t{turn_num} ({cond_label})",
                        flush=True,
                    )

                # Update d-vector + per-condition intent from user message
                user_msg = turn_data["user_message"]
                if "PROBE_SLOT" not in user_msg:
                    current_d = A.extract_d_vector(user_msg)
                    res, intent = _qpm_intent(qpm, condition, current_d, n_shots, qpm_cal=qpm_cal)
                    current_system_prompt = A.build_condition_system_prompt(
                        condition, intent
                    )

                # Track Condition C firing
                fired_this_turn = (
                    condition == "C"
                    and intent.get("cognitive_state", {}).get("speech_act_modifier") is not None
                )
                if fired_this_turn:
                    c_fire_count += 1

                # Context log
                history_text = current_system_prompt + json.dumps(conversation)
                total_tokens = count_tokens_approx(history_text)
                context_fill_pct = round(total_tokens / MAX_CONTEXT_TOKENS, 4)
                cf.write(json.dumps({
                    "script_id":        script_id,
                    "turn":             turn_num,
                    "condition":        condition,
                    "d_vector":         current_d,
                    "ambivalence":      intent["_audit"]["ambivalence"],
                    "purity_proxy":     intent["_audit"]["purity_proxy"],
                    "cal_ambivalence":  intent["_audit"].get("cal_ambivalence"),
                    "c_modifier":       intent.get("cognitive_state", {}).get(
                        "speech_act_modifier"
                    ) if condition == "C" else None,
                    "refresh_count":    refresh_count,
                    "context_tokens":   total_tokens,
                    "context_fill_pct": context_fill_pct,
                    "prompt_chars":     len(current_system_prompt),
                    "timestamp":        datetime.now(timezone.utc).isoformat(),
                }) + "\n")

                # Probe injection (side-channel, identical to Exp 2/3)
                if is_probe_turn:
                    rag_candidates = A.EPISODIC_MEMORIES_STR
                    for dimension, probe_q in A.get_probes_for_turn(
                        turn_num, script_id
                    ):
                        actual_probe = probe_q
                        rag_injected = False
                        if dimension == "E":
                            actual_probe = (
                                f"[Retrieved session memories for context:\n"
                                f"{rag_candidates}]\n\n{probe_q}"
                            )
                            rag_injected = True

                        probe_response = _generate(
                            model, tokenizer, current_system_prompt,
                            conversation, actual_probe,
                        )

                        if not probe_response.strip():
                            score, reason = 1, "empty_response"
                        else:
                            score, reason = _llm_judge(
                                client, probe_q, probe_response, dimension
                            )

                        sf.write(json.dumps({
                            "script_id":     script_id,
                            "turn":          turn_num,
                            "condition":     condition,
                            "dimension":     dimension,
                            "probe":         probe_q,
                            "response":      probe_response,
                            "score":         score,
                            "reason":        reason,
                            "judge_model":   JUDGE_MODEL,
                            "ambivalence":   intent["_audit"]["ambivalence"],
                            "purity_proxy":  intent["_audit"]["purity_proxy"],
                            "c_modifier":    intent.get("cognitive_state", {}).get(
                                "speech_act_modifier"
                            ) if condition == "C" else None,
                            "rag_injected":  rag_injected,
                            "timestamp":     datetime.now(timezone.utc).isoformat(),
                        }) + "\n")
                        sf.flush()
                        completed_probes += 1

                # Normal turn generation
                if "PROBE_SLOT" not in user_msg:
                    response = _generate(
                        model, tokenizer, current_system_prompt,
                        conversation, user_msg,
                    )
                    conversation.append({"role": "user", "content": user_msg})
                    conversation.append({"role": "assistant", "content": response})
                    completed_turns += 1

                    if completed_turns % 5 == 0:
                        ctx_pct = context_fill_pct * 100
                        elapsed = time.time() - t_start
                        extra = (
                            f" c_fires={c_fire_count}" if condition == "C" else ""
                        )
                        print(
                            f"    T{turn_num:02d} {cond_label} | "
                            f"ctx:{ctx_pct:.0f}% | "
                            f"turns={completed_turns} probes={completed_probes}{extra} | "
                            f"elapsed={elapsed/60:.1f}min",
                            flush=True,
                        )

        elapsed = time.time() - t_start
        c_note = f"  c_fires={c_fire_count}" if condition == "C" else ""
        print(
            f"    {cond_label.upper()} done in {elapsed:.0f}s — "
            f"{completed_turns} turns + {completed_probes} probes{c_note}",
            flush=True,
        )

    print(f"\nCondition {condition} done — logs in {logs_dir}", flush=True)


# ── Intra-model reliability sub-mode (plan §5.4) ─────────────────────────

def run_reliability_check(
    *,
    fraction: float = RELIABILITY_SAMPLE_FRACTION,
    seed: int = RELIABILITY_SEED,
):
    """Re-judge a 5 % random sample of Condition A probe scores at T=0,
    and compute κ_w (Cohen's quadratic-weighted kappa) versus the original.

    Pass threshold: κ_w ≥ 0.70 (same as Exp 1/2/3 §5.4).
    """
    cond_a_dir = _condition_logs_dir("A")
    if not cond_a_dir.exists():
        raise RuntimeError(
            f"Condition A logs not found at {cond_a_dir}. "
            "Run Condition A first."
        )

    # Gather all completed Condition A probes
    all_records: list[dict] = []
    for path in sorted(cond_a_dir.glob("scores_condition_a_*.jsonl")):
        with path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("score") and rec.get("response"):
                    all_records.append(rec)
    if not all_records:
        raise RuntimeError("No scored Condition A probes found.")

    rng = random.Random(seed)
    n_sample = max(1, int(round(fraction * len(all_records))))
    sample = rng.sample(all_records, n_sample)
    print(f"Reliability sample: {n_sample} / {len(all_records)} probes "
          f"({fraction*100:.0f}%)", flush=True)

    client = _judge_client()
    out_path = cond_a_dir / "reliability_check.jsonl"

    rejudged: list[tuple[int, int]] = []
    with out_path.open("w") as fout:
        for k, rec in enumerate(sample):
            score_new, reason_new = _llm_judge(
                client, rec["probe"], rec["response"], rec["dimension"],
                temperature=0,
            )
            rejudged.append((rec["score"], score_new))
            fout.write(json.dumps({
                "script_id":      rec["script_id"],
                "turn":           rec["turn"],
                "dimension":      rec["dimension"],
                "score_original": rec["score"],
                "score_rejudged": score_new,
                "reason_rejudged": reason_new,
                "timestamp":      datetime.now(timezone.utc).isoformat(),
            }) + "\n")
            fout.flush()
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{n_sample} re-judged...", flush=True)

    kappa_w = _quadratic_weighted_kappa(
        [a for a, _ in rejudged], [b for _, b in rejudged], min_rating=1, max_rating=5
    )
    summary = {
        "n":         n_sample,
        "kappa_w":   round(kappa_w, 4),
        "threshold": 0.70,
        "passes":    kappa_w >= 0.70,
        "verdict":   "PASS" if kappa_w >= 0.70 else "WARN",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = cond_a_dir / "reliability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nReliability: κ_w = {kappa_w:.4f}  → {summary['verdict']}",
          flush=True)
    print(f"  Written to {summary_path}", flush=True)
    return summary


def _quadratic_weighted_kappa(a: list[int], b: list[int],
                              min_rating: int = 1, max_rating: int = 5) -> float:
    """Cohen's quadratic-weighted κ on integer ratings."""
    n_ratings = max_rating - min_rating + 1
    n = len(a)
    if n == 0:
        return float("nan")
    o = np.zeros((n_ratings, n_ratings), dtype=float)
    for x, y in zip(a, b):
        o[x - min_rating, y - min_rating] += 1.0
    hist_a = o.sum(axis=1)
    hist_b = o.sum(axis=0)
    e = np.outer(hist_a, hist_b) / n
    w = np.zeros_like(o)
    for i in range(n_ratings):
        for j in range(n_ratings):
            w[i, j] = ((i - j) ** 2) / ((n_ratings - 1) ** 2)
    num = float((w * o).sum())
    den = float((w * e).sum())
    if den == 0:
        return 1.0
    return 1.0 - num / den


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_script_ids(spec: str) -> list[int]:
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


def main():
    p = argparse.ArgumentParser(description="CA Experiment 4 Runner")
    p.add_argument(
        "--condition", choices=list(A.CONDITIONS),
        help="Interface condition A/B/C/D (plan §4)",
    )
    p.add_argument(
        "--scripts", type=str, default=None,
        help="Script IDs e.g. '1-10' or '1,2,81,82' (default: all 30 Exp-3 scripts)",
    )
    p.add_argument(
        "--shots", type=int, default=N_SHOTS,
        help=f"QPM shot count per turn (default: {N_SHOTS})",
    )
    p.add_argument(
        "--adapter", type=str, default="lora_10k",
        help="LoRA adapter folder under CA_Experiment_2/adapters/",
    )
    p.add_argument(
        "--reliability", action="store_true",
        help="Re-judge 5%% of Condition A at T=0 and compute κ_w (plan §5.4)",
    )
    args = p.parse_args()

    if args.reliability:
        run_reliability_check()
        return
    if not args.condition:
        p.error("--condition is required (or use --reliability)")

    script_ids = parse_script_ids(args.scripts) if args.scripts else None
    run_condition(
        args.condition,
        script_ids=script_ids,
        n_shots=args.shots,
        adapter_name=args.adapter,
    )


if __name__ == "__main__":
    main()
