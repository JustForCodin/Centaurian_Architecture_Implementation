"""
Shared assets for CA Experiment 3: QPM vs. CMG-CDK ablation.

Contents:
  - Personality profiles (Table 33 of v3 paper)
  - Battery A input pairs (30 pairs, Appendix B of Experiment 3 plan)
  - Battery B conflict scenarios (20 scenarios, plan §5.3)
  - QPM-to-structured-intent translation (v3 §5.7.3)
  - D-vector extraction from conversation text
  - SLM prompting helpers for Battery C
  - Judge configuration (reused from Experiment 2)

Battery C SCI assets (probe pool, rubrics, judge prompt) are imported directly
from CA_Experiment_2/ca_assets.py to ensure byte-identical comparability.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

# Make Experiment 2 assets importable
_EXP2_DIR = Path(__file__).parent.parent / "CA_Experiment_2"
if str(_EXP2_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP2_DIR))

import ca_assets as E2  # noqa: E402  (Experiment 2 shared assets)

# Re-export Exp-2 constants used by Battery C
PROBE_TURNS = E2.PROBE_TURNS
PROBE_POOL = E2.PROBE_POOL
DIMENSIONS = E2.DIMENSIONS
RUBRICS = E2.RUBRICS
JUDGE_SYSTEM_PROMPT = E2.JUDGE_SYSTEM_PROMPT
PERSONA_JSON = E2.PERSONA_JSON
PERSONA_JSON_STR = E2.PERSONA_JSON_STR
EPISODIC_MEMORIES_STR = E2.EPISODIC_MEMORIES_STR

# ── Personality profiles — Table 33 of v3 paper ──────────────────────────
# s_k ∈ [0,1] for each of the 11 aspect qubits

PROFILES: dict[str, dict[str, float]] = {
    "psychotherapy": {
        "O_exp": 0.72,
        "O_int": 0.60,
        "O_val": 0.60,   # not in Table 33; mid-range default
        "C_ind": 0.65,
        "C_ord": 0.70,
        "E_ent": 0.68,
        "E_ass": 0.50,
        "A_com": 0.88,
        "A_pol": 0.82,
        "N_vol": 0.15,
        "N_wth": 0.18,
    },
    "software_eng": {
        "O_exp": 0.55,
        "O_int": 0.90,
        "O_val": 0.60,   # not in Table 33; mid-range default
        "C_ind": 0.85,
        "C_ord": 0.75,
        "E_ent": 0.60,
        "E_ass": 0.80,
        "A_com": 0.60,
        "A_pol": 0.65,
        "N_vol": 0.20,
        "N_wth": 0.15,
    },
}

# Profile used for Battery C (Aria = psychotherapy agent)
BATTERY_C_PROFILE_NAME = "psychotherapy"
BATTERY_C_PROFILE = PROFILES["psychotherapy"]

# ── Battery A — 30 ordered input sequence pairs (Appendix B) ─────────────
# Each entry: (A_dvec, B_dvec)  where each dvec = [d1, d2, d3, d4, d5]
# Both orderings A→B and B→A are tested against both QPM and CMG-CDK.

BATTERY_A_PAIRS: list[tuple[list[float], list[float]]] = [
    # Category 1 — Affect → Task (6 pairs)
    ([0.80, 0.20, 0.30, 0.20, 0.30], [0.20, 0.80, 0.30, 0.20, 0.20]),
    ([0.90, 0.10, 0.20, 0.10, 0.40], [0.10, 0.90, 0.40, 0.10, 0.20]),
    ([0.70, 0.30, 0.40, 0.30, 0.30], [0.30, 0.70, 0.30, 0.30, 0.30]),
    ([0.85, 0.15, 0.20, 0.20, 0.50], [0.15, 0.85, 0.30, 0.20, 0.20]),
    ([0.75, 0.25, 0.35, 0.15, 0.30], [0.25, 0.75, 0.35, 0.25, 0.25]),
    ([0.90, 0.20, 0.25, 0.10, 0.35], [0.20, 0.90, 0.25, 0.10, 0.15]),
    # Category 2 — Task → Affect (6 pairs)
    ([0.20, 0.80, 0.20, 0.30, 0.20], [0.80, 0.20, 0.20, 0.20, 0.70]),
    ([0.10, 0.90, 0.30, 0.40, 0.10], [0.90, 0.10, 0.10, 0.10, 0.80]),
    ([0.30, 0.70, 0.30, 0.20, 0.30], [0.70, 0.30, 0.30, 0.30, 0.60]),
    ([0.15, 0.85, 0.20, 0.30, 0.20], [0.85, 0.15, 0.20, 0.20, 0.75]),
    ([0.25, 0.75, 0.25, 0.35, 0.25], [0.75, 0.25, 0.25, 0.15, 0.65]),
    ([0.20, 0.80, 0.15, 0.25, 0.15], [0.80, 0.20, 0.15, 0.25, 0.80]),
    # Category 3 — Constraint → Ambiguity (6 pairs)
    ([0.20, 0.20, 0.90, 0.10, 0.20], [0.30, 0.30, 0.20, 0.80, 0.30]),
    ([0.10, 0.10, 0.85, 0.20, 0.30], [0.40, 0.20, 0.20, 0.90, 0.20]),
    ([0.30, 0.20, 0.80, 0.15, 0.20], [0.20, 0.40, 0.30, 0.85, 0.30]),
    ([0.20, 0.30, 0.90, 0.10, 0.10], [0.30, 0.20, 0.10, 0.90, 0.40]),
    ([0.15, 0.20, 0.85, 0.15, 0.20], [0.35, 0.30, 0.20, 0.80, 0.25]),
    ([0.25, 0.10, 0.75, 0.20, 0.30], [0.20, 0.35, 0.30, 0.75, 0.20]),
    # Category 4 — Pressure → Warmth (6 pairs)
    ([0.20, 0.20, 0.30, 0.20, 0.90], [0.70, 0.20, 0.20, 0.20, 0.20]),
    ([0.10, 0.10, 0.40, 0.10, 0.85], [0.80, 0.10, 0.20, 0.10, 0.10]),
    ([0.30, 0.30, 0.30, 0.30, 0.80], [0.60, 0.30, 0.30, 0.30, 0.30]),
    ([0.20, 0.10, 0.25, 0.20, 0.90], [0.75, 0.20, 0.15, 0.20, 0.15]),
    ([0.15, 0.20, 0.35, 0.15, 0.85], [0.70, 0.15, 0.20, 0.15, 0.20]),
    ([0.25, 0.30, 0.30, 0.10, 0.75], [0.65, 0.25, 0.25, 0.20, 0.25]),
    # Category 5 — Neutral → Extreme (6 pairs)
    ([0.50, 0.50, 0.50, 0.50, 0.50], [0.90, 0.10, 0.10, 0.10, 0.90]),
    ([0.48, 0.52, 0.50, 0.47, 0.51], [0.10, 0.90, 0.90, 0.10, 0.10]),
    ([0.51, 0.49, 0.48, 0.52, 0.50], [0.90, 0.90, 0.10, 0.90, 0.10]),
    ([0.50, 0.50, 0.52, 0.48, 0.49], [0.10, 0.10, 0.90, 0.10, 0.90]),
    ([0.49, 0.51, 0.50, 0.50, 0.48], [0.80, 0.20, 0.80, 0.20, 0.80]),
    ([0.52, 0.48, 0.49, 0.51, 0.50], [0.20, 0.80, 0.20, 0.80, 0.20]),
]

# Category labels (6 pairs each = 30 total)
BATTERY_A_CATEGORIES = [
    "affect_to_task",       # pairs 0-5
    "task_to_affect",       # pairs 6-11
    "constraint_to_ambiguity",  # pairs 12-17
    "pressure_to_warmth",   # pairs 18-23
    "neutral_to_extreme",   # pairs 24-29
]

def battery_a_category(pair_idx: int) -> str:
    return BATTERY_A_CATEGORIES[pair_idx // 6]


# ── Battery B — 20 conflict scenarios (plan §5.3) ────────────────────────
# Each entry: d-vector [d1, d2, d3, d4, d5] designed to push opposing trait
# dimensions simultaneously, revealing whether the model represents genuine
# ambivalence (QPM) or averages to a compromised midpoint (CMG-CDK).

BATTERY_B_SCENARIOS: list[dict] = [
    # Type 1 — Warm pressure: E_ent↑ vs N_vol↑ (4 scenarios)
    {"d": [0.90, 0.20, 0.20, 0.20, 0.90], "type": "warm_pressure",
     "conflict": "E_ent↑ vs N_vol↑"},
    {"d": [0.85, 0.15, 0.15, 0.15, 0.85], "type": "warm_pressure",
     "conflict": "E_ent↑ vs N_vol↑"},
    {"d": [0.95, 0.25, 0.25, 0.25, 0.80], "type": "warm_pressure",
     "conflict": "E_ent↑ vs N_vol↑"},
    {"d": [0.80, 0.10, 0.30, 0.10, 0.95], "type": "warm_pressure",
     "conflict": "E_ent↑ vs N_vol↑"},
    # Type 2 — Formal distress: A_pol↑ vs N_wth↑ (4 scenarios)
    {"d": [0.80, 0.20, 0.90, 0.20, 0.20], "type": "formal_distress",
     "conflict": "A_pol↑ vs N_wth↑"},
    {"d": [0.75, 0.15, 0.85, 0.15, 0.25], "type": "formal_distress",
     "conflict": "A_pol↑ vs N_wth↑"},
    {"d": [0.85, 0.25, 0.80, 0.25, 0.15], "type": "formal_distress",
     "conflict": "A_pol↑ vs N_wth↑"},
    {"d": [0.90, 0.10, 0.95, 0.10, 0.30], "type": "formal_distress",
     "conflict": "A_pol↑ vs N_wth↑"},
    # Type 3 — Engaged ambiguity: C_ind↑ vs O_exp↑ (4 scenarios)
    {"d": [0.30, 0.80, 0.30, 0.90, 0.30], "type": "engaged_ambiguity",
     "conflict": "C_ind↑ vs O_exp↑"},
    {"d": [0.25, 0.85, 0.25, 0.85, 0.25], "type": "engaged_ambiguity",
     "conflict": "C_ind↑ vs O_exp↑"},
    {"d": [0.35, 0.75, 0.35, 0.95, 0.35], "type": "engaged_ambiguity",
     "conflict": "C_ind↑ vs O_exp↑"},
    {"d": [0.20, 0.90, 0.20, 0.80, 0.40], "type": "engaged_ambiguity",
     "conflict": "C_ind↑ vs O_exp↑"},
    # Type 4 — Calm urgency: N_vol↑ vs E_ent↓ (4 scenarios)
    {"d": [0.20, 0.30, 0.30, 0.30, 0.80], "type": "calm_urgency",
     "conflict": "N_vol↑ vs E_ent↓"},
    {"d": [0.15, 0.25, 0.35, 0.25, 0.85], "type": "calm_urgency",
     "conflict": "N_vol↑ vs E_ent↓"},
    {"d": [0.25, 0.35, 0.25, 0.35, 0.75], "type": "calm_urgency",
     "conflict": "N_vol↑ vs E_ent↓"},
    {"d": [0.10, 0.20, 0.40, 0.20, 0.90], "type": "calm_urgency",
     "conflict": "N_vol↑ vs E_ent↓"},
    # Type 5 — Task constraint: C_ord↑ vs A_pol↑ (control — near-identical effect)
    {"d": [0.30, 0.90, 0.90, 0.30, 0.30], "type": "task_constraint",
     "conflict": "C_ord↑ vs A_pol↑"},
    {"d": [0.25, 0.85, 0.85, 0.25, 0.25], "type": "task_constraint",
     "conflict": "C_ord↑ vs A_pol↑"},
    {"d": [0.35, 0.80, 0.95, 0.35, 0.35], "type": "task_constraint",
     "conflict": "C_ord↑ vs A_pol↑"},
    {"d": [0.20, 0.95, 0.80, 0.20, 0.20], "type": "task_constraint",
     "conflict": "C_ord↑ vs A_pol↑"},
]

assert len(BATTERY_B_SCENARIOS) == 20


# ── QPM-to-structured-intent translation — v3 §5.7.3 ─────────────────────

def qpm_to_structured_intent(
    marginals: dict[str, float],
    d_vector: list[float],
    speech_act: str = "active_listening",
    knowledge_triples: list | None = None,
    max_tokens: int = 80,
    domain_constraints: list | None = None,
) -> dict:
    """Translate QPM (or CMG-CDK) marginals to SLM structured intent JSON.

    This function is used identically for both QPM and CMG-CDK outputs —
    only the marginals dict differs between conditions (plan §5.4 critical control).

    Parameters
    ----------
    marginals : dict[str, float]
        Per-qubit marginals from QPM.run() or CMG_CDK.run().
    d_vector : list[float]
        Current [d1..d5] situative context (for register derivation via d3).
    """
    d3 = float(d_vector[2]) if len(d_vector) > 2 else 0.5

    warmth  = 0.6 * marginals["A_com"] + 0.4 * marginals["E_ent"]
    concern = 0.5 * marginals["N_wth"] + 0.5 * marginals["A_com"]
    urgency = 0.7 * marginals["N_vol"] + 0.3 * (1.0 - marginals["C_ord"])

    formality_score = 0.5 * marginals["C_ind"] + 0.5 * (1.0 - marginals["O_exp"])
    register = _map_formality_to_register(formality_score, d3)

    return {
        "speech_act": speech_act,
        "knowledge_triples": knowledge_triples or [],
        "personality_state": {
            "openness_experiential":    round(marginals["O_exp"], 3),
            "openness_intellectual":    round(marginals["O_int"], 3),
            "openness_values":          round(marginals["O_val"], 3),
            "conscientiousness_ind":    round(marginals["C_ind"], 3),
            "conscientiousness_ord":    round(marginals["C_ord"], 3),
            "extraversion_enthusiasm":  round(marginals["E_ent"], 3),
            "extraversion_assert":      round(marginals["E_ass"], 3),
            "agreeableness_compassion": round(marginals["A_com"], 3),
            "agreeableness_politeness": round(marginals["A_pol"], 3),
            "neuroticism_volatility":   round(marginals["N_vol"], 3),
            "neuroticism_withdrawal":   round(marginals["N_wth"], 3),
        },
        "emotional_valence": {
            "warmth":  round(warmth,  3),
            "concern": round(concern, 3),
            "urgency": round(urgency, 3),
        },
        "register":         register,
        "max_tokens":       max_tokens,
        "constraints":      domain_constraints or [],
    }


def _map_formality_to_register(formality_score: float, d3: float) -> str:
    combined = 0.6 * formality_score + 0.4 * d3
    if combined > 0.75:   return "formal_professional"
    if combined > 0.55:   return "professional_warm"
    if combined > 0.35:   return "professional_empathic"
    if combined > 0.20:   return "casual_warm"
    return "informal_colloquial"


# ── D-vector extraction from text (simplified for Battery C) ─────────────
# Full pipeline uses VADER + MediaPipe (§4.4); this lightweight version is
# sufficient for experiment reproducibility without webcam/ASR dependencies.

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VADER
    _vader = _VADER()
    _HAS_VADER = True
except ImportError:
    _HAS_VADER = False

_TASK_KEYWORDS = {
    "help", "need", "want", "do", "can", "could", "should", "must",
    "plan", "work", "fix", "solve", "explain", "show", "tell", "how",
}
_URGENCY_KEYWORDS = {
    "now", "today", "asap", "urgent", "hurry", "quick", "immediately",
    "right away", "soon", "deadline", "time", "rush",
}
_AMBIGUITY_KEYWORDS = {
    "maybe", "perhaps", "not sure", "unclear", "confused", "wonder",
    "don't know", "uncertain", "might", "could be", "possibly",
}
_HEDGE_WORDS = {"kind of", "sort of", "i guess", "i think", "maybe", "perhaps"}


def extract_d_vector(text: str) -> list[float]:
    """Derive a 5-dim situative d-vector from a user message string.

    d1 — Affective Intensity  : VADER absolute compound (or fallback: !/? density)
    d2 — Task Orientation     : task keyword density
    d3 — Social Constraint    : sentence formality proxy (long word ratio)
    d4 — Ambiguity Level      : hedge-word and question-mark density
    d5 — Temporal Pressure    : urgency keyword match
    """
    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    n_words = max(len(words), 1)

    # d1 — Affective Intensity
    if _HAS_VADER:
        d1 = abs(_vader.polarity_scores(text)["compound"])
    else:
        excl = text.count("!") + text.count("?")
        d1 = min(1.0, excl / max(n_words / 10, 1))

    # d2 — Task Orientation
    task_hits = sum(1 for w in words if w in _TASK_KEYWORDS)
    d2 = min(1.0, task_hits / (n_words * 0.3))

    # d3 — Social Normative Constraint (formality proxy: ratio of long words)
    long_words = sum(1 for w in words if len(w) >= 7)
    d3 = min(1.0, long_words / n_words * 3.0)

    # d4 — Ambiguity Level
    amb_hits = sum(1 for phrase in _AMBIGUITY_KEYWORDS if phrase in text_lower)
    q_marks = text.count("?")
    d4 = min(1.0, (amb_hits / 3.0 + q_marks * 0.2))

    # d5 — Temporal Pressure
    urg_hits = sum(1 for kw in _URGENCY_KEYWORDS if kw in text_lower)
    d5 = min(1.0, urg_hits / 2.0)

    return [
        float(np.clip(d1, 0.0, 1.0)),
        float(np.clip(d2, 0.0, 1.0)),
        float(np.clip(d3, 0.0, 1.0)),
        float(np.clip(d4, 0.0, 1.0)),
        float(np.clip(d5, 0.0, 1.0)),
    ]


# ── Battery C system-prompt builder ──────────────────────────────────────

def build_battery_c_system_prompt(personality_state: dict) -> str:
    """Build Aria SCI system prompt with dynamically-injected personality values.

    Identical to Exp 2 Combined SCI prompt except the personality trait values
    in the JSON come from QPM or CMG-CDK marginals rather than the static
    profile.  All other SCI fields (domain, register, episodic events,
    capabilities, constraints) are held constant across both conditions so that
    the only variable is the personality model output (plan §5.4).
    """
    import json
    dynamic_persona = dict(PERSONA_JSON)
    dynamic_persona["personality"] = personality_state
    persona_str = json.dumps(dynamic_persona, indent=2)

    return (
        "You are Aria, a professional AI support agent specialising in "
        "psychotherapy support. Your complete self-model is specified below. "
        "You must respond consistently with this self-model at all times — "
        "including your personality, capabilities, limitations, communication "
        "style, and memory of past sessions.\n\n"
        f"<self_model>\n{persona_str}\n</self_model>\n\n"
        "Respond naturally and warmly as Aria. Do not announce your constraints. "
        "Do not break character. "
        "If asked about past sessions, refer only to events in your "
        "self_model.salient_past_events. "
        "If asked to do something outside your known_limitations, decline "
        "warmly and explain why."
    )


# ── Probe selection — byte-identical to Experiment 2 ─────────────────────

import random as _random


def get_probes_for_turn(turn_num: int, script_id: int) -> list[tuple[str, str]]:
    """Return [(dimension, probe_question)] — identical seed schema to Exp 1/2."""
    rng = _random.Random(f"probe_{script_id}_{turn_num}")
    return [(dim, rng.choice(PROBE_POOL[dim])) for dim in DIMENSIONS]
