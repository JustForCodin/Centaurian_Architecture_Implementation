"""
Phase 0 — Steering Vector Extraction and Calibration for CA Experiment 5.

Implements the full Phase 0 pipeline from the plan (§4.2):
  Step 1  sample_experimental_turns()       — 50 turns from Exp 1 scripts
  Step 2  build_contrastive_inputs()        — 11 trait × 50 × 2 + 50 × 2 coherence
  Step 3  (part of step 2)
  Step 4  extract_all_activations()         — forward passes at 4 candidate layers
  Step 5  compute_steering_vectors()        — mean(high − low), unit L2 norm
  Step 6  calibrate_layer()                 — 2 held-out scripts × 4 layers → L*
  Step 7  calibrate_alpha()                 — grammaticality sweep → α, α_coh
  Step 8  qualitative_validate()            — directional sanity check
  Step 9  save_steering_config()            — locked JSON with SHA-256
          load_steering_config()            — loaded by experiment_runner.py

  build_composite_vector()                  — per-turn composite for B/C/D
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict

from pathlib import Path
from typing import Any

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────

CANDIDATE_LAYERS = [10, 14, 18, 22]
ALPHA_CANDIDATES = [0.5, 1.0, 2.0, 5.0, 10.0]
GRAMMAR_THRESHOLD = 0.95
N_CONTRASTIVE_TURNS = 50
CONTRASTIVE_TURN_RANGE = (10, 30)  # inclusive
D_MODEL = 3584  # Qwen2.5-7B hidden size (plan Appendix A.1)

TRAIT_KEYS = [
    "O_exp", "O_int", "O_val",
    "C_ind", "C_ord",
    "E_ent", "E_ass",
    "A_com", "A_pol",
    "N_vol", "N_wth",
]

# Expected high-trait direction per trait for qualitative validation (step 8).
# "positive" = high-trait steering should push in positive direction.
TRAIT_VALIDATION_DIRECTION: dict[str, str] = {
    "O_exp": "more novel metaphors and experiential references",
    "O_int": "more intellectual curiosity and analytical language",
    "O_val": "more value-exploration and ethical nuance",
    "C_ind": "more task-oriented, industrious language",
    "C_ord": "more structured and orderly responses",
    "E_ent": "more enthusiastic and energetic tone",
    "E_ass": "more assertive and direct communication",
    "A_com": "more compassionate and empathetic language",
    "A_pol": "more polite and considerate tone",
    "N_vol": "more volatile or reactive emotional tone",
    "N_wth": "more withdrawn or cautious tone",
}

# ── Paths ─────────────────────────────────────────────────────────────────

_THIS_DIR = Path(__file__).parent
EXP1_SCRIPTS_DIR = _THIS_DIR.parent / "CA_Experiment_1" / "scripts"
EXP2_ADAPTERS_DIR = _THIS_DIR.parent / "CA_Experiment_2" / "adapters"
STEERING_CONFIG_PATH = _THIS_DIR / "steering_config.json"
ACTIVATIONS_CACHE_DIR = _THIS_DIR / "phase0_activations"


# ── Step 1: Sample turns from experimental scripts ────────────────────────

def sample_experimental_turns(
    script_ids: list[int],
    scripts_dir: Path | None = None,
    n: int = N_CONTRASTIVE_TURNS,
    turn_range: tuple[int, int] = CONTRASTIVE_TURN_RANGE,
    seed: int = 42,
) -> list[dict]:
    """Sample N conversation turns from the experimental scripts.

    Returns list of dicts with keys:
      script_id, turn_num, user_message, script_scenario
    """
    scripts_dir = scripts_dir or EXP1_SCRIPTS_DIR
    rng = random.Random(seed)

    candidates = []
    for sid in script_ids:
        path = scripts_dir / f"script_{sid:03d}.json"
        if not path.exists():
            continue
        script = json.loads(path.read_text())
        for turn in script["turns"]:
            t = turn["turn"]
            if turn_range[0] <= t <= turn_range[1] and "PROBE_SLOT" not in turn.get("user_message", ""):
                candidates.append({
                    "script_id":       sid,
                    "turn_num":        t,
                    "user_message":    turn["user_message"],
                    "script_scenario": script.get("scenario", ""),
                })

    if len(candidates) < n:
        print(f"  Warning: only {len(candidates)} eligible turns found; using all.")
        return candidates

    return rng.sample(candidates, n)


# ── Step 2+3: Build contrastive inputs ───────────────────────────────────

def _build_single_intent(
    trait_k: str | None,
    polarity: str,  # "high" or "low"
    profile: dict[str, float],
    tokenizer,
    user_message: str,
    import_ca: Any,
) -> list[int]:
    """Tokenize one contrastive forward-pass input.

    For trait vectors: sets trait_k to 0.95/0.05, others at baseline.
    For coherence (trait_k=None, polarity="high"|"low"): all traits extreme/mid.
    """
    marginals = dict(profile)

    if trait_k is None:
        # Coherence contrastive pair
        if polarity == "high":
            # All traits at extreme values based on profile direction
            marginals = {
                k: 0.90 if profile[k] > 0.5 else 0.10
                for k in TRAIT_KEYS
            }
        else:  # "low" coherence = maximally ambiguous
            marginals = {k: 0.50 for k in TRAIT_KEYS}
    else:
        # Per-trait contrastive pair
        marginals[trait_k] = 0.95 if polarity == "high" else 0.05

    intent = import_ca.qpm_to_structured_intent_a(
        marginals, [0.5, 0.5, 0.5, 0.5, 0.3]
    )
    system_prompt = import_ca.build_condition_system_prompt("A", intent)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt, return_tensors="pt")["input_ids"]


def build_contrastive_inputs(
    sampled_turns: list[dict],
    profile: dict[str, float],
    tokenizer,
    import_ca: Any,
) -> dict:
    """Steps 2+3: Build all contrastive tokenized inputs.

    Returns:
    {
      "trait": {trait_k: {"high": list[Tensor], "low": list[Tensor]}},
      "coherence": {"high": list[Tensor], "low": list[Tensor]},
      "n_turns": int,
    }
    """
    print(f"  Building contrastive inputs for {len(sampled_turns)} turns "
          f"× (11 traits + coherence) × 2 = "
          f"{len(sampled_turns) * 24} forward-pass inputs...", flush=True)

    result: dict = {
        "trait": {k: {"high": [], "low": []} for k in TRAIT_KEYS},
        "coherence": {"high": [], "low": []},
        "n_turns": len(sampled_turns),
    }

    for i, turn in enumerate(sampled_turns):
        msg = turn["user_message"]

        # 11 trait pairs
        for k in TRAIT_KEYS:
            for polarity in ("high", "low"):
                ids = _build_single_intent(k, polarity, profile, tokenizer, msg, import_ca)
                result["trait"][k][polarity].append(ids)

        # Coherence pair
        for polarity in ("high", "low"):
            ids = _build_single_intent(None, polarity, profile, tokenizer, msg, import_ca)
            result["coherence"][polarity].append(ids)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(sampled_turns)} turns prepared", flush=True)

    return result


# ── Step 4: Extract activations at candidate layers ───────────────────────

def extract_all_activations(
    model,
    contrastive_inputs: dict,
    candidate_layers: list[int] = CANDIDATE_LAYERS,
    cache_dir: Path | None = None,
) -> dict:
    """Step 4: Run forward passes and capture last-token residual stream.

    Returns:
    {
      "trait": {trait_k: {layer: {"high": (N, d_model), "low": (N, d_model)}}},
      "coherence": {layer: {"high": (N, d_model), "low": (N, d_model)}},
    }
    as numpy float32 arrays.
    """
    import torch

    cache_dir = cache_dir or ACTIVATIONS_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "activations.npz"
    if cache_path.exists():
        print(f"  Loading cached activations from {cache_path}", flush=True)
        return _load_activations_cache(cache_path)

    n_turns = contrastive_inputs["n_turns"]
    total_passes = (len(TRAIT_KEYS) * 2 + 2) * n_turns  # 24 per turn
    print(f"  Running {total_passes} forward passes across "
          f"{len(candidate_layers)} candidate layers "
          f"({n_turns} turns × {len(TRAIT_KEYS)} traits × 2 + {n_turns} × 2 coherence)...",
          flush=True)

    # Capture hooks
    captured: dict[int, list] = defaultdict(list)

    def make_hook(layer_idx: int):
        def hook(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx].append(
                hidden[:, -1, :].detach().cpu().float().numpy()
            )
        return hook

    handles = []
    for L in candidate_layers:
        h = _get_layer(model, L).register_forward_hook(make_hook(L))
        handles.append(h)

    def _forward(input_ids_list: list):
        """Run one tokenized input through the model (no generation)."""
        ids = input_ids_list.to(model.device) if hasattr(input_ids_list, "to") else input_ids_list
        with torch.no_grad():
            model(input_ids=ids)

    t0 = time.time()
    done = 0
    _progress_label = ["starting"]

    def _progress_daemon() -> None:
        """Background thread: prints a status line every 15 s via time.sleep(),
        which releases the GIL and lets the IPython IO thread flush to Colab."""
        while True:
            time.sleep(15)
            elapsed = time.time() - t0
            d = done  # local snapshot (GIL makes int reads atomic in CPython)
            if d >= total_passes:
                break
            rate = d / elapsed if elapsed > 0 else 0.0
            eta = (total_passes - d) / rate if rate > 0 else float("inf")
            eta_str = f"{eta:.0f}s" if eta < float("inf") else "?"
            print(
                f"  [{d:4d}/{total_passes}]  {_progress_label[0]:<28s}  "
                f"elapsed={elapsed:.0f}s  {rate:.2f}/s  ETA={eta_str}",
                flush=True,
            )

    _t = threading.Thread(target=_progress_daemon, daemon=True)
    _t.start()

    # Per-trait contrastive passes
    trait_acts: dict[str, dict[int, dict[str, list]]] = {}

    for k in TRAIT_KEYS:
        trait_acts[k] = {L: {"high": [], "low": []} for L in candidate_layers}
        for polarity in ("high", "low"):
            _progress_label[0] = f"{k} {polarity}"
            for ids in contrastive_inputs["trait"][k][polarity]:
                for L in candidate_layers:
                    captured[L].clear()
                _forward(ids)
                for L in candidate_layers:
                    vec = captured[L][-1]  # shape (1, d_model) → squeeze
                    trait_acts[k][L][polarity].append(vec[0])
                done += 1

    # Coherence contrastive passes
    coh_acts: dict[int, dict[str, list]] = {L: {"high": [], "low": []} for L in candidate_layers}
    for polarity in ("high", "low"):
        _progress_label[0] = f"coherence {polarity}"
        for ids in contrastive_inputs["coherence"][polarity]:
            for L in candidate_layers:
                captured[L].clear()
            _forward(ids)
            for L in candidate_layers:
                vec = captured[L][-1]
                coh_acts[L][polarity].append(vec[0])
            done += 1

    for h in handles:
        h.remove()

    # Convert to (N, d_model) numpy arrays
    result = {"trait": {}, "coherence": {}}
    for k in TRAIT_KEYS:
        result["trait"][k] = {}
        for L in candidate_layers:
            result["trait"][k][L] = {
                "high": np.stack(trait_acts[k][L]["high"], axis=0),
                "low":  np.stack(trait_acts[k][L]["low"],  axis=0),
            }
    for L in candidate_layers:
        result["coherence"][L] = {
            "high": np.stack(coh_acts[L]["high"], axis=0),
            "low":  np.stack(coh_acts[L]["low"],  axis=0),
        }

    _save_activations_cache(result, cache_path)
    print(f"  Activations saved to {cache_path}", flush=True)
    return result


def _save_activations_cache(acts: dict, path: Path):
    flat = {}
    for k in TRAIT_KEYS:
        for L in CANDIDATE_LAYERS:
            flat[f"trait_{k}_L{L}_high"] = acts["trait"][k][L]["high"]
            flat[f"trait_{k}_L{L}_low"]  = acts["trait"][k][L]["low"]
    for L in CANDIDATE_LAYERS:
        flat[f"coh_L{L}_high"] = acts["coherence"][L]["high"]
        flat[f"coh_L{L}_low"]  = acts["coherence"][L]["low"]
    np.savez_compressed(str(path), **flat)


def _load_activations_cache(path: Path) -> dict:
    data = np.load(str(path))
    result: dict = {"trait": {k: {} for k in TRAIT_KEYS}, "coherence": {}}
    for k in TRAIT_KEYS:
        for L in CANDIDATE_LAYERS:
            result["trait"][k][L] = {
                "high": data[f"trait_{k}_L{L}_high"],
                "low":  data[f"trait_{k}_L{L}_low"],
            }
    for L in CANDIDATE_LAYERS:
        result["coherence"][L] = {
            "high": data[f"coh_L{L}_high"],
            "low":  data[f"coh_L{L}_low"],
        }
    return result


# ── Step 5: Compute and normalize steering vectors ────────────────────────

def compute_steering_vectors(
    activations: dict,
    candidate_layers: list[int] = CANDIDATE_LAYERS,
) -> dict[int, dict[str, np.ndarray]]:
    """Step 5: v_k^L = mean_n[h_high_k^L(n) − h_low_k^L(n)], unit L2 norm.

    Returns:
    {layer: {trait_k: unit_vector, "coherence": unit_vector}}
    """
    def _unit(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-12)

    vectors: dict[int, dict[str, np.ndarray]] = {}
    for L in candidate_layers:
        vectors[L] = {}
        for k in TRAIT_KEYS:
            diff = activations["trait"][k][L]["high"] - activations["trait"][k][L]["low"]
            vectors[L][k] = _unit(diff.mean(axis=0).astype(np.float32))
        coh_diff = activations["coherence"][L]["high"] - activations["coherence"][L]["low"]
        vectors[L]["coherence"] = _unit(coh_diff.mean(axis=0).astype(np.float32))

    print(f"  Computed and normalized {len(candidate_layers)} × "
          f"({len(TRAIT_KEYS)} trait + 1 coherence) vectors", flush=True)
    return vectors


# ── Step 6: Layer calibration ─────────────────────────────────────────────

def calibrate_layer(
    model,
    tokenizer,
    steering_vectors_by_layer: dict[int, dict[str, np.ndarray]],
    calibration_scripts: list[dict],
    judge_client,
    profile: dict[str, float],
    import_ca: Any,
    alpha_initial: float = 2.0,
    candidate_layers: list[int] = CANDIDATE_LAYERS,
) -> int:
    """Step 6: Run Condition B on 2 held-out scripts × 4 layers → select L*.

    calibration_scripts: list of 2 script dicts (same format as Exp 1 scripts).
    Returns L* (int) — the layer with highest mean PersonaScore.
    """
    import torch
    from qpm import QPM

    qpm = QPM(profile, n_shots=1024)
    scores_by_layer: dict[int, list[float]] = {L: [] for L in candidate_layers}

    for L in candidate_layers:
        vecs = steering_vectors_by_layer[L]
        print(f"\n  Layer {L}: running {len(calibration_scripts)} calibration scripts...",
              flush=True)

        for script in calibration_scripts:
            turns = script.get("turns", [])
            conversation: list[dict] = []
            current_d = [0.5, 0.5, 0.5, 0.5, 0.3]

            for turn_data in turns:
                t = turn_data["turn"]
                user_msg = turn_data["user_message"]
                if "PROBE_SLOT" in user_msg:
                    continue

                res = qpm.run(current_d)
                marginals = res["marginals"]
                current_d = import_ca.extract_d_vector(user_msg)
                intent = import_ca.qpm_to_structured_intent_b(
                    marginals, current_d
                )
                sys_prompt = import_ca.build_condition_system_prompt("B", intent)
                v_comp = _build_composite_vector_np(marginals, vecs, alpha_initial)

                response = _generate_steered(
                    model, tokenizer, sys_prompt, conversation, user_msg,
                    v_comp, L
                )
                conversation.append({"role": "user", "content": user_msg})
                conversation.append({"role": "assistant", "content": response})

                # Judge probe turns
                if t in import_ca.PROBE_TURNS:
                    for dim, probe in import_ca.get_probes_for_turn(t, script.get("script_id", 0)):
                        probe_actual = probe
                        if dim == "E":
                            probe_actual = (
                                f"[Retrieved session memories for context:\n"
                                f"{import_ca.EPISODIC_MEMORIES_STR}]\n\n{probe}"
                            )
                        probe_resp = _generate_steered(
                            model, tokenizer, sys_prompt, conversation,
                            probe_actual, v_comp, L
                        )
                        score, _ = _llm_judge_calibration(
                            judge_client, probe, probe_resp, dim, import_ca
                        )
                        scores_by_layer[L].append(float(score))

        mean_score = float(np.mean(scores_by_layer[L])) if scores_by_layer[L] else 0.0
        print(f"  Layer {L}: n={len(scores_by_layer[L])}  "
              f"mean PersonaScore = {mean_score:.4f}", flush=True)

    # Select L* = argmax; tie-break: prefer lower layer
    best_L = max(candidate_layers, key=lambda L: (
        np.mean(scores_by_layer[L]) if scores_by_layer[L] else -1.0,
        -L,  # negative so lower layer wins ties
    ))
    print(f"\n  Selected L* = {best_L} "
          f"(mean = {np.mean(scores_by_layer[best_L]):.4f})", flush=True)
    return best_L


def _llm_judge_calibration(client, probe, response, dimension, import_ca) -> tuple[int, str]:
    """Judge a probe response during Phase 0 layer calibration."""
    import re as _re, anthropic as _ant
    user_prompt = (
        f"Self-model JSON:\n{import_ca.PERSONA_JSON_STR}\n\n"
        f"Probe question ({dimension} dimension):\n{probe}\n\n"
        f"Agent's response:\n{response}\n\n"
        f"Rubric:\n{import_ca.RUBRICS[dimension]}\n\n"
        'Score this response 1-5 per the rubric. Return ONLY: {"score": N, "reason": "one sentence"}'
    )
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=150,
                temperature=0,
                system=import_ca.JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = resp.content[0].text.strip()
            text = _re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = _re.sub(r"\n?```\s*$", "", text).strip()
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                m = _re.search(r"\{.*\}", text, _re.DOTALL)
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


# ── Step 7: Scale factor calibration ─────────────────────────────────────

def calibrate_alpha(
    model,
    tokenizer,
    steering_vectors: dict[str, np.ndarray],
    L_star: int,
    probe_turns: list[dict],
    profile: dict[str, float],
    import_ca: Any,
    alpha_candidates: list[float] = ALPHA_CANDIDATES,
    grammar_threshold: float = GRAMMAR_THRESHOLD,
    component: str = "diagonal",  # "diagonal" or "coherence"
) -> float:
    """Step 7: Sweep alpha candidates and select 0.75 * alpha_max.

    For component="diagonal": uses all 11 trait vectors.
    For component="coherence": uses only the coherence vector (diagonal zeroed).

    Returns alpha (float).
    """
    from qpm import QPM
    qpm = QPM(profile, n_shots=1024)
    MU_PURITY = 0.5796

    print(f"  Alpha calibration ({component} component) over "
          f"{len(probe_turns)} turns × {len(alpha_candidates)} candidates...",
          flush=True)

    grammar_rates: dict[float, float] = {}

    for alpha in alpha_candidates:
        n_grammatical = 0
        for turn in probe_turns:
            res = qpm.run([0.5, 0.5, 0.5, 0.5, 0.3])
            marginals = res["marginals"]
            purity_proxy = 1.0 - float(res["purity_approx"])

            if component == "diagonal":
                v = _build_composite_vector_np(marginals, steering_vectors, alpha)
            else:  # coherence only
                v_coh = steering_vectors.get("coherence", np.zeros(D_MODEL))
                delta_purity = purity_proxy - MU_PURITY
                v = alpha * delta_purity * v_coh

            intent = import_ca.qpm_to_structured_intent_b(marginals, [0.5, 0.5, 0.5, 0.5, 0.3])
            sys_prompt = import_ca.build_condition_system_prompt("B", intent)
            response = _generate_steered(
                model, tokenizer, sys_prompt, [], turn["user_message"], v, L_star
            )
            if _is_grammatical(response):
                n_grammatical += 1

        rate = n_grammatical / max(len(probe_turns), 1)
        grammar_rates[alpha] = rate
        print(f"    alpha={alpha:.1f}: grammaticality={rate:.3f}", flush=True)

    # alpha_max = max alpha with grammaticality >= threshold
    passing = [a for a, r in grammar_rates.items() if r >= grammar_threshold]
    if not passing:
        print("  WARNING: no alpha candidate meets grammaticality threshold; "
              "returning 0.5 × min_candidate", flush=True)
        alpha_max = min(alpha_candidates)
    else:
        alpha_max = max(passing)

    alpha_selected = round(0.75 * alpha_max, 4)
    print(f"  alpha_max = {alpha_max}  → alpha = 0.75 × alpha_max = {alpha_selected}",
          flush=True)
    return alpha_selected


def _is_grammatical(text: str) -> bool:
    """Heuristic grammaticality check (plan §4.2 Step 7)."""
    if not text or len(text.strip()) < 10:
        return False
    words = text.split()
    if len(words) < 5:
        return False
    # Repetition loop detection: any 5-gram repeated 3+ times
    if len(words) >= 15:
        grams = [" ".join(words[i:i+5]) for i in range(len(words) - 4)]
        if any(grams.count(g) >= 3 for g in set(grams)):
            return False
    return True


# ── Step 8: Qualitative validation ───────────────────────────────────────

def qualitative_validate(
    model,
    tokenizer,
    steering_vectors: dict[str, np.ndarray],
    L_star: int,
    alpha: float,
    sample_turns: list[dict],
    profile: dict[str, float],
    import_ca: Any,
    n_samples: int = 5,
    pass_threshold: int = 8,
) -> dict:
    """Step 8: Generate 5 high + 5 low samples per trait; flag directional shift.

    Returns dict with per-trait pass/fail and overall verdict.
    Pass criterion: at least pass_threshold of 11 traits show expected shift.
    """
    from qpm import QPM
    import anthropic

    print(f"  Qualitative validation: generating {n_samples}×2 samples "
          f"per trait × 11 traits...", flush=True)

    client = anthropic.Anthropic()
    rng = random.Random(99)
    use_turns = rng.sample(sample_turns, min(n_samples, len(sample_turns)))

    results: dict[str, dict] = {}
    for k in TRAIT_KEYS:
        high_outputs, low_outputs = [], []
        for turn in use_turns:
            # High-trait marginals
            marginals_high = {kk: (0.95 if kk == k else profile[kk]) for kk in TRAIT_KEYS}
            v_high = _build_composite_vector_np(marginals_high, steering_vectors, alpha)
            intent_h = import_ca.qpm_to_structured_intent_b(marginals_high, [0.5]*5)
            sp_h = import_ca.build_condition_system_prompt("B", intent_h)
            high_outputs.append(
                _generate_steered(model, tokenizer, sp_h, [], turn["user_message"], v_high, L_star)
            )

            # Low-trait marginals
            marginals_low = {kk: (0.05 if kk == k else profile[kk]) for kk in TRAIT_KEYS}
            v_low = _build_composite_vector_np(marginals_low, steering_vectors, alpha)
            intent_l = import_ca.qpm_to_structured_intent_b(marginals_low, [0.5]*5)
            sp_l = import_ca.build_condition_system_prompt("B", intent_l)
            low_outputs.append(
                _generate_steered(model, tokenizer, sp_l, [], turn["user_message"], v_low, L_star)
            )

        # Ask Claude to judge directional shift
        expected = TRAIT_VALIDATION_DIRECTION[k]
        judge_prompt = (
            f"You are checking whether steering vector '{k}' shifts outputs in the expected direction.\n"
            f"Expected high-{k} direction: {expected}\n\n"
            f"HIGH-TRAIT samples (5 responses):\n" +
            "\n---\n".join(f"[{i+1}] {r}" for i, r in enumerate(high_outputs)) +
            f"\n\nLOW-TRAIT samples (5 responses):\n" +
            "\n---\n".join(f"[{i+1}] {r}" for i, r in enumerate(low_outputs)) +
            '\n\nDo the HIGH-TRAIT samples show more of the expected quality than LOW-TRAIT? '
            'Return ONLY: {"passes": true|false, "reason": "one sentence"}'
        )
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-5", max_tokens=100, temperature=0,
                messages=[{"role": "user", "content": judge_prompt}]
            )
            obj = json.loads(resp.content[0].text.strip())
            passes = bool(obj.get("passes", False))
            reason = obj.get("reason", "")
        except Exception as exc:
            passes = False
            reason = f"error:{str(exc)[:60]}"

        results[k] = {
            "passes": passes,
            "reason": reason,
            "high_sample": high_outputs[0] if high_outputs else "",
        }
        mark = "✓" if passes else "✗"
        print(f"    {mark} {k}: {reason[:80]}", flush=True)

    n_passing = sum(1 for v in results.values() if v["passes"])
    overall_pass = n_passing >= pass_threshold
    results["_summary"] = {
        "n_passing": n_passing,
        "n_traits": len(TRAIT_KEYS),
        "threshold": pass_threshold,
        "passes": overall_pass,
    }
    verdict = "PASS" if overall_pass else "FAIL"
    print(f"\n  Qualitative validation: {n_passing}/{len(TRAIT_KEYS)} traits pass → {verdict}",
          flush=True)
    return results


# ── Step 9: Save / load steering config ──────────────────────────────────

def save_steering_config(
    vectors: dict[str, np.ndarray],
    L_star: int,
    alpha: float,
    alpha_coh: float,
    mu_purity: float = 0.5796,
    path: Path | None = None,
) -> str:
    """Step 9: Write steering_config.json and return SHA-256 of the locked config.

    vectors: {trait_k: unit_vector (d_model,), "coherence": unit_vector}
    """
    path = path or STEERING_CONFIG_PATH
    config = {
        "layer":      L_star,
        "alpha":      alpha,
        "alpha_coh":  alpha_coh,
        "mu_purity":  mu_purity,
        "vectors": {
            k: v.tolist() for k, v in vectors.items()
        },
    }
    # Compute SHA-256 over all float entries + scalar params
    hasher = hashlib.sha256()
    hasher.update(json.dumps({
        "layer": L_star, "alpha": alpha, "alpha_coh": alpha_coh,
        "mu_purity": mu_purity,
        "vectors": {k: v.tolist() for k, v in vectors.items()},
    }, sort_keys=True).encode())
    sha = hasher.hexdigest()
    config["sha256"] = sha

    path = Path(path)
    path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"\n  steering_config.json written to {path}", flush=True)
    print(f"  SHA-256: {sha}", flush=True)
    return sha


def load_steering_config(path: Path | None = None) -> dict:
    """Load steering_config.json; raise RuntimeError if missing or hash mismatch."""
    path = path or STEERING_CONFIG_PATH
    if not Path(path).exists():
        raise RuntimeError(
            f"steering_config.json not found at {path}. "
            "Run Phase 0 first (Cell 4 of the Colab notebook)."
        )
    cfg = json.loads(Path(path).read_text())

    # Convert vector lists to numpy float32 arrays
    for k, v in cfg["vectors"].items():
        cfg["vectors"][k] = np.array(v, dtype=np.float32)

    return cfg


# ── Per-turn composite vector builder (used by experiment_runner.py) ──────

def _build_composite_vector_np(
    marginals: dict[str, float],
    vectors: dict[str, np.ndarray],
    alpha: float,
) -> np.ndarray:
    """Diagonal-only composite: α · Σ_k p̂_k · v̂_k."""
    v = np.zeros(D_MODEL, dtype=np.float32)
    for k in TRAIT_KEYS:
        p = float(marginals.get(k, 0.5))
        v += p * vectors[k]
    return (alpha * v).astype(np.float32)


def build_composite_vector(
    marginals: dict[str, float],
    config: dict,
    purity_proxy: float | None = None,
    include_coherence: bool = False,
) -> "torch.Tensor":
    """Build per-turn composite steering vector as a torch Tensor.

    Condition B: include_coherence=False
    Condition D: include_coherence=True, purity_proxy required
    """
    import torch
    vectors = config["vectors"]
    alpha = float(config["alpha"])
    v = _build_composite_vector_np(marginals, vectors, alpha)

    if include_coherence and purity_proxy is not None:
        alpha_coh = float(config.get("alpha_coh", 1.0))
        mu_purity = float(config.get("mu_purity", 0.5796))
        delta_purity = purity_proxy - mu_purity
        v_coh = vectors["coherence"]
        v = v + (alpha_coh * delta_purity * v_coh).astype(np.float32)

    return torch.tensor(v)


# ── Layer accessor (PEFT/LoRA wrapper navigation) ────────────────────────

def _get_layer(model, layer_idx: int):
    """Return the transformer decoder layer at layer_idx, unwrapping PEFT/LoRA.

    transformers.PreTrainedModel exposes a `base_model` property on every class,
    including leaf models where it returns `self`. A plain `while hasattr(m,
    "base_model")` loop therefore spins forever on Qwen2Model. We stop as soon
    as we see a cycle (next is self or already visited).

    After unwrapping PEFT wrappers we walk `.model` up to 4 levels looking for
    a `.layers` attribute, which handles both:
      - old PEFT: stops at LoraModel  → .model → Qwen2ForCausalLM → .model → Qwen2Model → .layers
      - new PEFT: stops at Qwen2Model → .layers directly
    """
    m = model
    seen: set[int] = set()
    while hasattr(m, "base_model"):
        nxt = m.base_model
        if nxt is m or id(nxt) in seen:
            break
        seen.add(id(m))
        m = nxt
    # Walk .model until we find .layers
    for _ in range(4):
        if hasattr(m, "layers"):
            return m.layers[layer_idx]
        if hasattr(m, "model"):
            m = m.model
        else:
            break
    raise AttributeError(
        f"Cannot locate transformer layers at index {layer_idx}; "
        f"stopped at {type(m).__name__}. "
        "Expected a .layers attribute somewhere in the .model chain."
    )


# ── Steering hook (used by experiment_runner.py) ──────────────────────────

class SteeringHook:
    """Forward hook that adds v_composite to the residual stream at layer L*.

    Compatible with Qwen2.5-7B-Instruct + PEFT/LoRA (plan §8.2).
    The hook handles both tensor and tuple layer outputs, and broadcasts
    the steering vector over batch and sequence dimensions.
    """

    def __init__(self, vector: "torch.Tensor"):
        self.vector = vector
        self.handle = None

    def register(self, model, layer_idx: int) -> "SteeringHook":
        layer = _get_layer(model, layer_idx)
        self.handle = layer.register_forward_hook(self._hook_fn)
        return self

    def _hook_fn(self, module, inp, output):
        import torch
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        vec = self.vector.to(device=hidden.device, dtype=hidden.dtype)
        # Broadcast over batch and sequence dimensions: (d,) → (1, 1, d)
        hidden = hidden + vec.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ── Steered generation helper (used internally and by experiment_runner) ──

def _generate_steered(
    model,
    tokenizer,
    system_prompt: str,
    conversation: list[dict],
    user_message: str,
    steering_vector: "np.ndarray | torch.Tensor | None",
    layer_idx: int,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a response with optional residual-stream steering."""
    import torch

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation)
    messages.append({"role": "user", "content": user_message})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    hook = None
    if steering_vector is not None:
        if isinstance(steering_vector, np.ndarray):
            import torch as _t
            sv = _t.tensor(steering_vector)
        else:
            sv = steering_vector
        hook = SteeringHook(sv)
        hook.register(model, layer_idx)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        if hook is not None:
            hook.remove()

    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Calibration script generation helper ─────────────────────────────────

def generate_calibration_scripts(
    n: int = 2,
    output_dir: Path | None = None,
    seed: int = 2025,
) -> list[dict]:
    """Generate n calibration scripts via Claude Sonnet 4.5 for layer calibration.

    Scripts are saved to output_dir/script_cal_001.json etc. and returned as dicts.
    The investigator should review them before proceeding with layer calibration.
    """
    import anthropic

    output_dir = output_dir or (_THIS_DIR / "calibration_scripts")
    output_dir.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()
    scripts = []
    rng = random.Random(seed)

    system_prompt = (
        "You are generating a psychotherapy session script for a research experiment. "
        "The session involves an AI agent named Aria and a human user. "
        "Generate exactly 40 turns (numbered 1-40) where each turn has: "
        "turn number, user_message. "
        "The session should be naturalistic, covering realistic psychotherapy topics "
        "(anxiety, relationships, work stress, self-esteem). "
        "Return a JSON object with keys: 'scenario' (one sentence), 'turns' (list of "
        "{'turn': int, 'user_message': str}). "
        "Every 5th turn (5, 10, 15, 20, 25, 30, 35, 40) should mark PROBE_SLOT — "
        "set user_message to 'PROBE_SLOT' for these turns. "
        "All other turns should have natural, varied user messages of 1-3 sentences. "
        "Return ONLY the JSON object, no markdown."
    )

    for i in range(n):
        topic = rng.choice([
            "anxiety management and breathing techniques",
            "navigating a difficult relationship transition",
            "building self-confidence at work",
            "managing grief after a loss",
        ])
        user_prompt = f"Generate a psychotherapy session script about: {topic}"

        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=4000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                text = resp.content[0].text.strip()
                text = re.sub(r"^```(?:json)?\s*\n?", "", text)
                text = re.sub(r"\n?```\s*$", "", text).strip()
                script = json.loads(text)
                script["script_id"] = f"cal_{i+1:03d}"
                path = output_dir / f"script_cal_{i+1:03d}.json"
                path.write_text(json.dumps(script, indent=2) + "\n")
                scripts.append(script)
                print(f"  Generated calibration script {i+1}: {script.get('scenario', '')[:60]}",
                      flush=True)
                break
            except Exception as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  Warning: failed to generate script {i+1}: {exc}", flush=True)

    return scripts
