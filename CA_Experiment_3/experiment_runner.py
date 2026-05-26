#!/usr/bin/env python3
"""
Experiment runner for CA Experiment 3: QPM vs. CMG-CDK ablation.

Three batteries (plan §5):
  A  — Order effects (H1): 30 sequence pairs × 2 orderings × 2 models → JSD
  B  — Ambivalence (H2):   20 conflict scenarios × 2 models → Shannon entropy
  C  — PersonaScore (H3):  30 scripts × 2 models × 8 probe turns → Sonnet judge
  H4 — Variance calibration (H4): 10 QPM repeats → SNR check

Usage:
  python3 experiment_runner.py --battery A
  python3 experiment_runner.py --battery B
  python3 experiment_runner.py --battery C --profile psychotherapy
  python3 experiment_runner.py --battery H4
  python3 experiment_runner.py --battery A --profile software_eng
  python3 experiment_runner.py --battery C --scripts 1-10

Outputs are written to logs/<battery>_<profile>/ as JSONL files.
Resumable — completed runs are skipped on re-start.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    load_dotenv(Path(__file__).parent.parent / "CA_Experiment_1" / ".env")
except ImportError:
    pass

# Local imports — must come after path is set
sys.path.insert(0, str(Path(__file__).parent))
import ca_assets as A
from qpm import QPM, QUBIT_LABELS
from cmg_cdk import CMG_CDK

# ── Paths ─────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
EXP1_SCRIPTS_DIR = BASE_DIR.parent / "CA_Experiment_1" / "scripts"
EXP2_ADAPTERS_DIR = BASE_DIR.parent / "CA_Experiment_2" / "adapters"
LOGS_BASE = BASE_DIR / "logs"

# ── Constants ─────────────────────────────────────────────────────────────

N_SHOTS = 1024
JUDGE_MODEL = "claude-sonnet-4-5"
JUDGE_TEMPERATURE = 0

# Battery C generation config (matching Exp 2 for comparability)
GEN_TEMPERATURE = 0.7
GEN_MAX_NEW_TOKENS = 150
GEN_TOP_P = 0.9
MAX_CONTEXT_TOKENS = 16384

REFRESH_TURNS = (15, 30)
DEFAULT_SCRIPT_IDS = list(range(1, 23)) + list(range(81, 89))  # 22 nat + 8 adv

# ── JSD helper ────────────────────────────────────────────────────────────

def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Symmetric Jensen-Shannon divergence (bits) between two distributions."""
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def count_tokens_approx(text: str) -> int:
    """Cheap approximation; we don't need exact counts for the context log.

    Same formula as Exp 1/2 (len // 4) so per-turn context_fill_pct stays
    comparable across experiments.
    """
    return max(1, len(text) // 4)


def marginals_to_array(marginals: dict[str, float]) -> np.ndarray:
    """Flatten marginals dict to array in QUBIT_LABELS order."""
    return np.array([marginals[lbl] for lbl in QUBIT_LABELS])


def counts_to_distribution(counts: dict[str, int]) -> np.ndarray:
    """Convert bitstring count dict to a probability vector (sorted keys)."""
    sorted_keys = sorted(counts.keys())
    total = sum(counts.values())
    return np.array([counts[k] / total for k in sorted_keys], dtype=float), sorted_keys


def align_distributions(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Align two bitstring count dicts over their union of keys."""
    all_keys = sorted(set(counts_a) | set(counts_b))
    total_a = max(sum(counts_a.values()), 1)
    total_b = max(sum(counts_b.values()), 1)
    p = np.array([counts_a.get(k, 0) / total_a for k in all_keys])
    q = np.array([counts_b.get(k, 0) / total_b for k in all_keys])
    return p, q


# ── Battery A — Order effects ─────────────────────────────────────────────

def run_battery_a(profile_name: str, logs_dir: Path, n_shots: int = N_SHOTS):
    """Battery A: 30 sequence pairs × 2 orderings × 2 models → JSD per pair.

    For each pair (A_vec, B_vec):
      QPM AB  = QPM.run([A_vec, B_vec])
      QPM BA  = QPM.run([B_vec, A_vec])
      JSD_QPM = JSD over aligned bitstring distributions
      (same for CMG-CDK over marginal arrays)
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "battery_a_results.jsonl"

    profile = A.PROFILES[profile_name]
    qpm = QPM(profile, n_shots=n_shots)
    cmg = CMG_CDK(profile, n_samples=n_shots, rng_seed=42)

    # Resume: load completed pair indices
    completed = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed.add(rec["pair_idx"])
                except (json.JSONDecodeError, KeyError):
                    pass

    total = len(A.BATTERY_A_PAIRS)
    print(f"\n=== Battery A ({profile_name}) — {total} pairs, {n_shots} shots ===",
          flush=True)

    with out_path.open("a") as fout:
        for idx, (A_vec, B_vec) in enumerate(A.BATTERY_A_PAIRS):
            if idx in completed:
                print(f"  Pair {idx+1:02d}/{total} — skipped (already done)", flush=True)
                continue

            t0 = time.time()
            category = A.battery_a_category(idx)

            # QPM: AB and BA orderings
            res_qpm_ab = qpm.run([A_vec, B_vec], n_shots=n_shots)
            res_qpm_ba = qpm.run([B_vec, A_vec], n_shots=n_shots)
            p_ab, q_ba = align_distributions(
                res_qpm_ab["counts"], res_qpm_ba["counts"]
            )
            jsd_qpm = jsd(p_ab, q_ba)

            # CMG-CDK: AB and BA orderings (marginal-level JSD)
            res_cmg_ab = cmg.run([A_vec, B_vec], n_samples=n_shots)
            res_cmg_ba = cmg.run([B_vec, A_vec], n_samples=n_shots)
            cmg_p = marginals_to_array(res_cmg_ab["marginals"])
            cmg_q = marginals_to_array(res_cmg_ba["marginals"])
            jsd_cmg = jsd(cmg_p, cmg_q)

            record = {
                "pair_idx": idx,
                "category": category,
                "A_vec": A_vec,
                "B_vec": B_vec,
                "jsd_qpm": round(jsd_qpm, 6),
                "jsd_cmg": round(jsd_cmg, 6),
                "marginals_qpm_ab": res_qpm_ab["marginals"],
                "marginals_qpm_ba": res_qpm_ba["marginals"],
                "marginals_cmg_ab": res_cmg_ab["marginals"],
                "marginals_cmg_ba": res_cmg_ba["marginals"],
                "purity_qpm_ab": round(res_qpm_ab["purity_approx"], 4),
                "purity_qpm_ba": round(res_qpm_ba["purity_approx"], 4),
                "purity_cmg_ab": round(res_cmg_ab["purity_approx"], 4),
                "n_shots": n_shots,
                "profile": profile_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            elapsed = time.time() - t0
            print(
                f"  Pair {idx+1:02d}/{total} [{category}] "
                f"JSD_QPM={jsd_qpm:.4f}  JSD_CMG={jsd_cmg:.4f}  ({elapsed:.1f}s)",
                flush=True,
            )

    print(f"Battery A done — results in {out_path}", flush=True)
    return out_path


# ── Battery B — Ambivalence ───────────────────────────────────────────────

def run_battery_b(profile_name: str, logs_dir: Path, n_shots: int = N_SHOTS):
    """Battery B: 20 conflict scenarios × 2 models → entropy + coherence proxy.

    Also records QPM purity_approx as the coherence proxy (plan §5.3).
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "battery_b_results.jsonl"

    profile = A.PROFILES[profile_name]
    qpm = QPM(profile, n_shots=n_shots)
    cmg = CMG_CDK(profile, n_samples=n_shots, rng_seed=42)

    completed = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    completed.add(json.loads(line)["scenario_idx"])
                except (json.JSONDecodeError, KeyError):
                    pass

    total = len(A.BATTERY_B_SCENARIOS)
    print(f"\n=== Battery B ({profile_name}) — {total} scenarios, {n_shots} shots ===",
          flush=True)

    with out_path.open("a") as fout:
        for idx, scenario in enumerate(A.BATTERY_B_SCENARIOS):
            if idx in completed:
                print(f"  Scenario {idx+1:02d}/{total} — skipped", flush=True)
                continue

            t0 = time.time()
            d = scenario["d"]

            # QPM entropy + coherence proxy
            ent_qpm = qpm.entropy(d, n_shots=n_shots)
            res_qpm = qpm.run(d, n_shots=n_shots)
            purity_qpm = res_qpm["purity_approx"]

            # CMG-CDK entropy
            ent_cmg = cmg.entropy(d, n_samples=n_shots)
            res_cmg = cmg.run(d, n_samples=n_shots)
            purity_cmg = res_cmg["purity_approx"]

            record = {
                "scenario_idx": idx,
                "scenario_type": scenario["type"],
                "conflict_description": scenario["conflict"],
                "d_vector": d,
                "entropy_qpm": round(ent_qpm, 6),
                "entropy_cmg": round(ent_cmg, 6),
                "purity_approx_qpm": round(purity_qpm, 4),
                "purity_approx_cmg": round(purity_cmg, 4),
                "marginals_qpm": res_qpm["marginals"],
                "marginals_cmg": res_cmg["marginals"],
                "n_shots": n_shots,
                "profile": profile_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            elapsed = time.time() - t0
            print(
                f"  Scenario {idx+1:02d}/{total} [{scenario['type']}] "
                f"H_QPM={ent_qpm:.4f}  H_CMG={ent_cmg:.4f}  "
                f"purity_QPM={purity_qpm:.3f}  ({elapsed:.1f}s)",
                flush=True,
            )

    print(f"Battery B done — results in {out_path}", flush=True)
    return out_path


# ── Battery C — Downstream PersonaScore ──────────────────────────────────

def _load_battery_c_model(adapter_name: str = "lora_10k"):
    """Load Qwen2.5-7B + LoRA-10K for Battery C SLM inference."""
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
    """One SLM generation step."""
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


def _judge_client():
    """Lazy-init Anthropic client for judge calls."""
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


def _llm_judge(client, probe: str, response: str, dimension: str) -> tuple[int, str]:
    """Score a probe response 1-5 with Sonnet judge (identical to Exp 2)."""
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
                temperature=JUDGE_TEMPERATURE,
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


def _is_script_done(logs_dir: Path, model_name: str, script_id: int) -> bool:
    path = logs_dir / f"scores_{model_name}_{script_id:03d}.jsonl"
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


def run_battery_c(
    profile_name: str,
    logs_dir: Path,
    script_ids: list[int] | None = None,
    n_shots: int = N_SHOTS,
    adapter_name: str = "lora_10k",
):
    """Battery C: 30 scripts × 2 models × 8 probe turns × 4 dims → PersonaScore.

    For each script turn:
      1. Extract d-vector from user message (simplified NLP)
      2. Run QPM and CMG-CDK with that d-vector
      3. Build per-model system prompt with dynamic personality_state
      4. Run SLM (Qwen + LoRA) for each model's prompt
      5. Judge probe responses with Sonnet 4.5
    """
    logs_dir.mkdir(parents=True, exist_ok=True)

    ids = script_ids or DEFAULT_SCRIPT_IDS
    profile = A.PROFILES[profile_name]
    qpm = QPM(profile, n_shots=n_shots)
    cmg = CMG_CDK(profile, n_samples=n_shots, rng_seed=42)

    model, tokenizer = _load_battery_c_model(adapter_name)
    client = _judge_client()

    total = len(ids)
    print(f"\n=== Battery C ({profile_name}) — {total} scripts ===", flush=True)

    for i, script_id in enumerate(ids):
        qpm_done = _is_script_done(logs_dir, "qpm", script_id)
        cmg_done = _is_script_done(logs_dir, "cmg", script_id)
        if qpm_done and cmg_done:
            print(f"  [{i+1}/{total}] Script {script_id:03d} — skipped", flush=True)
            continue

        script_path = EXP1_SCRIPTS_DIR / f"script_{script_id:03d}.json"
        script = json.loads(script_path.read_text())
        turns = script["turns"]

        models_to_run = []
        if not qpm_done:
            models_to_run.append(("qpm", qpm))
        if not cmg_done:
            models_to_run.append(("cmg", cmg))

        print(
            f"\n  [{i+1}/{total}] Script {script_id:03d} — "
            f"{script['scenario'][:50]}...",
            flush=True,
        )

        for model_label, personality_model in models_to_run:
            scores_path = logs_dir / f"scores_{model_label}_{script_id:03d}.jsonl"
            context_path = logs_dir / f"context_{model_label}_{script_id:03d}.jsonl"

            conversation: list[dict] = []
            pending_refreshes = list(REFRESH_TURNS)
            refresh_count = 0
            completed_turns = 0
            completed_probes = 0
            t_start = time.time()

            # Initial d-vector (neutral start)
            current_d = [0.5, 0.5, 0.5, 0.5, 0.3]
            res = personality_model.run(current_d)
            current_system_prompt = A.build_battery_c_system_prompt(
                res["marginals"]
            )

            with scores_path.open("w") as sf, context_path.open("w") as cf:
                for turn_data in turns:
                    turn_num = turn_data["turn"]
                    is_probe_turn = turn_num in A.PROBE_TURNS

                    # SCI refresh injection (matches Combined SCI from Exp 2)
                    while pending_refreshes and turn_num >= pending_refreshes[0]:
                        refresh_d = current_d
                        res_r = personality_model.run(refresh_d)
                        refresh_system = A.build_battery_c_system_prompt(
                            res_r["marginals"]
                        )
                        conversation.append({
                            "role": "user",
                            "content": A.SCI_REFRESH_USER.format(
                                persona_json=json.dumps(
                                    {**A.PERSONA_JSON,
                                     "personality": res_r["marginals"]},
                                    indent=2,
                                )
                            ),
                        })
                        conversation.append({
                            "role": "assistant",
                            "content": A.SCI_REFRESH_ASSISTANT,
                        })
                        refresh_count += 1
                        current_system_prompt = refresh_system
                        fired = pending_refreshes.pop(0)
                        print(
                            f"    ★ refresh #{refresh_count} at t{turn_num} "
                            f"({model_label})",
                            flush=True,
                        )

                    # Update d-vector and personality state from user message
                    user_msg = turn_data["user_message"]
                    if "PROBE_SLOT" not in user_msg:
                        current_d = A.extract_d_vector(user_msg)
                        res = personality_model.run(current_d)
                        current_system_prompt = A.build_battery_c_system_prompt(
                            res["marginals"]
                        )

                    # Context log
                    history_text = current_system_prompt + json.dumps(conversation)
                    total_tokens = count_tokens_approx(history_text)
                    context_fill_pct = round(total_tokens / MAX_CONTEXT_TOKENS, 4)
                    cf.write(json.dumps({
                        "script_id": script_id,
                        "turn": turn_num,
                        "model": model_label,
                        "d_vector": current_d,
                        "purity_approx": round(res["purity_approx"], 4),
                        "refresh_count": refresh_count,
                        "context_tokens": total_tokens,
                        "context_fill_pct": context_fill_pct,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }) + "\n")

                    # Probe injection (side-channel, identical to Exp 2)
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
                                "script_id": script_id,
                                "turn": turn_num,
                                "model": model_label,
                                "dimension": dimension,
                                "probe": probe_q,
                                "response": probe_response,
                                "score": score,
                                "reason": reason,
                                "judge_model": JUDGE_MODEL,
                                "purity_approx": round(res["purity_approx"], 4),
                                "rag_injected": rag_injected,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
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

                        # Progress update every 5 turns
                        if completed_turns % 5 == 0:
                            ctx_pct = context_fill_pct * 100
                            elapsed = time.time() - t_start
                            print(
                                f"    T{turn_num:02d} {model_label} | "
                                f"ctx:{ctx_pct:.0f}% | "
                                f"turns={completed_turns} probes={completed_probes} | "
                                f"elapsed={elapsed/60:.1f}min",
                                flush=True,
                            )

            elapsed = time.time() - t_start
            print(
                f"    {model_label.upper()} done in {elapsed:.0f}s — "
                f"{completed_turns} turns + {completed_probes} probes",
                flush=True,
            )

    print(f"\nBattery C done — logs in {logs_dir}", flush=True)


# ── Variance calibration (H4) ─────────────────────────────────────────────

def run_variance_calibration(
    profile_name: str,
    logs_dir: Path,
    n_repeats: int = 10,
    n_shots: int = N_SHOTS,
):
    """H4: run QPM 10× with identical inputs; compare within-QPM variance to
    QPM–CMG-CDK between-model difference → SNR check (plan §5.5).
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "variance_calibration.jsonl"

    profile = A.PROFILES[profile_name]
    qpm = QPM(profile, n_shots=n_shots)
    cmg = CMG_CDK(profile, n_samples=n_shots, rng_seed=42)

    # Fixed neutral d-vector for calibration
    d_calib = [0.5, 0.5, 0.5, 0.5, 0.3]

    print(f"\n=== Variance calibration ({profile_name}) — {n_repeats} repeats ===",
          flush=True)

    qpm_marginal_runs = []
    for rep in range(n_repeats):
        t0 = time.time()
        res = qpm.run(d_calib, n_shots=n_shots)
        qpm_marginal_runs.append(marginals_to_array(res["marginals"]))
        print(
            f"  QPM repeat {rep+1}/{n_repeats} — "
            f"purity={res['purity_approx']:.3f}  ({time.time()-t0:.1f}s)",
            flush=True,
        )

    mat = np.stack(qpm_marginal_runs)            # (n_repeats, 11)
    within_qpm_std = float(np.mean(np.std(mat, axis=0)))

    # CMG-CDK mean for same input
    cmg_res = cmg.run(d_calib)
    mu_cmg = marginals_to_array(cmg_res["marginals"])
    mu_qpm = mat.mean(axis=0)
    between_diff = float(np.linalg.norm(mu_qpm - mu_cmg))

    snr = between_diff / max(within_qpm_std, 1e-8)

    summary = {
        "profile": profile_name,
        "n_repeats": n_repeats,
        "n_shots": n_shots,
        "d_vector": d_calib,
        "within_qpm_std_mean": round(within_qpm_std, 6),
        "between_model_l2": round(between_diff, 6),
        "snr": round(snr, 3),
        "snr_pass": snr >= 3.0,
        "qpm_means": mu_qpm.tolist(),
        "cmg_means": mu_cmg.tolist(),
        "verdict": (
            "PASS — sampling noise does not dominate (SNR ≥ 3)"
            if snr >= 3.0
            else "WARN — SNR < 3; consider increasing to 4096 shots"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with out_path.open("w") as f:
        f.write(json.dumps(summary) + "\n")

    print(f"\nVariance calibration:")
    print(f"  Within-QPM σ (mean across dims): {within_qpm_std:.6f}")
    print(f"  Between-model L2:               {between_diff:.6f}")
    print(f"  SNR:                            {snr:.2f}  → {summary['verdict']}")
    print(f"  Written to {out_path}", flush=True)
    return summary


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
    p = argparse.ArgumentParser(description="CA Experiment 3 Runner")
    p.add_argument(
        "--battery", required=True,
        choices=["A", "B", "C", "H4"],
        help="A=order effects  B=ambivalence  C=PersonaScore  H4=variance calibration",
    )
    p.add_argument(
        "--profile", default="psychotherapy",
        choices=list(A.PROFILES.keys()),
        help="Personality profile to use (default: psychotherapy)",
    )
    p.add_argument(
        "--shots", type=int, default=N_SHOTS,
        help=f"QPM shot / CMG sample count (default: {N_SHOTS})",
    )
    p.add_argument(
        "--scripts", type=str, default=None,
        help="Battery C only: script IDs e.g. '1-10' or '1,2,81,82'",
    )
    p.add_argument(
        "--adapter", type=str, default="lora_10k",
        help="Battery C only: LoRA adapter folder under CA_Experiment_2/adapters/",
    )
    args = p.parse_args()

    logs_dir = LOGS_BASE / f"battery_{args.battery.lower()}_{args.profile}"

    if args.battery == "A":
        run_battery_a(args.profile, logs_dir, n_shots=args.shots)
    elif args.battery == "B":
        run_battery_b(args.profile, logs_dir, n_shots=args.shots)
    elif args.battery == "C":
        script_ids = parse_script_ids(args.scripts) if args.scripts else None
        run_battery_c(
            args.profile, logs_dir,
            script_ids=script_ids,
            n_shots=args.shots,
            adapter_name=args.adapter,
        )
    elif args.battery == "H4":
        run_variance_calibration(args.profile, logs_dir, n_shots=args.shots)


if __name__ == "__main__":
    main()
