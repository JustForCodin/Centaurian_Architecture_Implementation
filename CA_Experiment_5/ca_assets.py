"""
Shared assets for CA Experiment 5: Logits-Level QPM→SLM Steering via
Residual Stream Injection.

Conditions (new meanings from Exp 4):
  A — JSON marginals only (Exp 4/3 control replication)
  B — Diagonal activation steering, no personality_state in JSON
  C — JSON marginals + diagonal activation steering (dual channel)
  D — Diagonal + coherence activation steering, no personality_state in JSON

Interface differences vs Experiment 4:
- Conditions B and D omit personality_state from the structured intent JSON.
- The SLM system prompt for B/D removes the personality field from the
  embedded persona JSON (personality is conveyed via residual-stream injection).
- Conditions A and C retain the full Condition A JSON from Exp 3/4.
- emotional_valence is retained in all four conditions.

Re-exports from Exp 2:
  PROBE_TURNS, PROBE_POOL, DIMENSIONS, RUBRICS, JUDGE_SYSTEM_PROMPT,
  PERSONA_JSON, PERSONA_JSON_STR, EPISODIC_MEMORIES_STR,
  SCI_REFRESH_USER, SCI_REFRESH_ASSISTANT
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Literal

import numpy as np

# ── Import Exp-2 SCI assets by file path (byte-identical reuse) ──────────

_EXP2_DIR = Path(__file__).parent.parent / "CA_Experiment_2"
_exp2_spec = importlib.util.spec_from_file_location(
    "ca_assets_exp2", _EXP2_DIR / "ca_assets.py"
)
E2 = importlib.util.module_from_spec(_exp2_spec)
_exp2_spec.loader.exec_module(E2)  # type: ignore[union-attr]

PROBE_TURNS = E2.PROBE_TURNS
PROBE_POOL = E2.PROBE_POOL
DIMENSIONS = E2.DIMENSIONS
RUBRICS = E2.RUBRICS
JUDGE_SYSTEM_PROMPT = E2.JUDGE_SYSTEM_PROMPT
PERSONA_JSON = E2.PERSONA_JSON
PERSONA_JSON_STR = E2.PERSONA_JSON_STR
EPISODIC_MEMORIES_STR = E2.EPISODIC_MEMORIES_STR
SCI_REFRESH_USER = E2.SCI_REFRESH_USER
SCI_REFRESH_ASSISTANT = E2.SCI_REFRESH_ASSISTANT


# ── Personality profile ──────────────────────────────────────────────────

PROFILES: dict[str, dict[str, float]] = {
    "psychotherapy": {
        "O_exp": 0.72, "O_int": 0.60, "O_val": 0.60,
        "C_ind": 0.65, "C_ord": 0.70,
        "E_ent": 0.68, "E_ass": 0.50,
        "A_com": 0.88, "A_pol": 0.82,
        "N_vol": 0.15, "N_wth": 0.18,
    },
}

BATTERY_C_PROFILE_NAME = "psychotherapy"
BATTERY_C_PROFILE = PROFILES["psychotherapy"]

# QPM baseline purity (μ_purity = 1 − mean ambivalence from Exp 4 990-pt
# calibration distribution; used by Condition D coherence steering).
MU_PURITY = 0.5796


# ── Interface conditions (new meanings for Experiment 5) ─────────────────

CONDITIONS = ("A", "B", "C", "D")
Condition = Literal["A", "B", "C", "D"]

CONDITION_DESCRIPTIONS: dict[str, str] = {
    "A": "JSON marginals only (Exp 4/3 control replication)",
    "B": "Diagonal activation steering — no JSON personality",
    "C": "JSON marginals + diagonal activation steering (dual channel)",
    "D": "Diagonal + coherence activation steering — no JSON personality",
}


# ── Helpers ───────────────────────────────────────────────────────────────

def _personality_state_dict(marginals: dict[str, float]) -> dict[str, float]:
    """Standardised personality_state JSON block — same as Exp 3/4 Condition A."""
    return {
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
    }


def _map_formality_to_register(formality_score: float, d3: float) -> str:
    combined = 0.6 * formality_score + 0.4 * d3
    if combined > 0.75:   return "formal_professional"
    if combined > 0.55:   return "professional_warm"
    if combined > 0.35:   return "professional_empathic"
    if combined > 0.20:   return "casual_warm"
    return "informal_colloquial"


def _emotional_valence(marginals: dict[str, float]) -> dict[str, float]:
    warmth  = 0.6 * marginals["A_com"] + 0.4 * marginals["E_ent"]
    concern = 0.5 * marginals["N_wth"] + 0.5 * marginals["A_com"]
    urgency = 0.7 * marginals["N_vol"] + 0.3 * (1.0 - marginals["C_ord"])
    return {
        "warmth":  round(warmth,  3),
        "concern": round(concern, 3),
        "urgency": round(urgency, 3),
    }


# ── Condition A — JSON marginals only (Exp 3/4 control replication) ──────

def qpm_to_structured_intent_a(
    marginals: dict[str, float],
    d_vector: list[float],
    *,
    speech_act: str = "active_listening",
    knowledge_triples: list | None = None,
    max_tokens: int = 80,
    domain_constraints: list | None = None,
) -> dict:
    """Condition A — byte-identical to Exp 4 Condition A / Exp 3 QPM arm."""
    d3 = float(d_vector[2]) if len(d_vector) > 2 else 0.5
    formality_score = 0.5 * marginals["C_ind"] + 0.5 * (1.0 - marginals["O_exp"])
    register = _map_formality_to_register(formality_score, d3)
    return {
        "speech_act":        speech_act,
        "knowledge_triples": knowledge_triples or [],
        "personality_state": _personality_state_dict(marginals),
        "emotional_valence": _emotional_valence(marginals),
        "register":          register,
        "max_tokens":        max_tokens,
        "constraints":       list(domain_constraints or []),
    }


# ── Condition B — diagonal activation steering, no JSON personality ───────

def qpm_to_structured_intent_b(
    marginals: dict[str, float],
    d_vector: list[float],
    *,
    speech_act: str = "active_listening",
    knowledge_triples: list | None = None,
    max_tokens: int = 80,
    domain_constraints: list | None = None,
) -> dict:
    """Condition B — personality_state omitted; activation steering carries it."""
    d3 = float(d_vector[2]) if len(d_vector) > 2 else 0.5
    formality_score = 0.5 * marginals["C_ind"] + 0.5 * (1.0 - marginals["O_exp"])
    register = _map_formality_to_register(formality_score, d3)
    return {
        "speech_act":        speech_act,
        "knowledge_triples": knowledge_triples or [],
        "emotional_valence": _emotional_valence(marginals),   # retained per plan §4.4
        "register":          register,
        "max_tokens":        max_tokens,
        "constraints":       list(domain_constraints or []),
        # personality_state: intentionally omitted (plan §4.4 + §8.6)
    }


# ── Condition C — JSON marginals + diagonal steering (dual channel) ───────

def qpm_to_structured_intent_c(
    marginals: dict[str, float],
    d_vector: list[float],
    *,
    speech_act: str = "active_listening",
    knowledge_triples: list | None = None,
    max_tokens: int = 80,
    domain_constraints: list | None = None,
) -> dict:
    """Condition C — full Condition A JSON AND activation steering active."""
    return qpm_to_structured_intent_a(
        marginals, d_vector,
        speech_act=speech_act,
        knowledge_triples=knowledge_triples,
        max_tokens=max_tokens,
        domain_constraints=domain_constraints,
    )


# ── Condition D — diagonal + coherence steering, no JSON personality ──────

def qpm_to_structured_intent_d(
    marginals: dict[str, float],
    d_vector: list[float],
    *,
    speech_act: str = "active_listening",
    knowledge_triples: list | None = None,
    max_tokens: int = 80,
    domain_constraints: list | None = None,
) -> dict:
    """Condition D — same as B; coherence encoded in the steering vector."""
    return qpm_to_structured_intent_b(
        marginals, d_vector,
        speech_act=speech_act,
        knowledge_triples=knowledge_triples,
        max_tokens=max_tokens,
        domain_constraints=domain_constraints,
    )


# ── Single-entry dispatcher ──────────────────────────────────────────────

def qpm_to_structured_intent(
    condition: Condition,
    marginals: dict[str, float],
    d_vector: list[float],
    *,
    speech_act: str = "active_listening",
    knowledge_triples: list | None = None,
    max_tokens: int = 80,
    domain_constraints: list | None = None,
) -> dict:
    """Build structured-intent JSON for the chosen condition."""
    if condition == "A":
        return qpm_to_structured_intent_a(
            marginals, d_vector,
            speech_act=speech_act, knowledge_triples=knowledge_triples,
            max_tokens=max_tokens, domain_constraints=domain_constraints,
        )
    if condition == "B":
        return qpm_to_structured_intent_b(
            marginals, d_vector,
            speech_act=speech_act, knowledge_triples=knowledge_triples,
            max_tokens=max_tokens, domain_constraints=domain_constraints,
        )
    if condition == "C":
        return qpm_to_structured_intent_c(
            marginals, d_vector,
            speech_act=speech_act, knowledge_triples=knowledge_triples,
            max_tokens=max_tokens, domain_constraints=domain_constraints,
        )
    if condition == "D":
        return qpm_to_structured_intent_d(
            marginals, d_vector,
            speech_act=speech_act, knowledge_triples=knowledge_triples,
            max_tokens=max_tokens, domain_constraints=domain_constraints,
        )
    raise ValueError(f"Unknown condition: {condition!r}")


# ── Per-condition SLM system-prompt builder ──────────────────────────────

_BASE_INTRO = (
    "You are Aria, a professional AI support agent specialising in "
    "psychotherapy support. Your complete self-model is specified below. "
    "You must respond consistently with this self-model at all times — "
    "including your capabilities, limitations, communication "
    "style, and memory of past sessions."
)

_BASE_OUTRO = (
    "Respond naturally and warmly as Aria. Do not announce your constraints. "
    "Do not break character. "
    "If asked about past sessions, refer only to events in your "
    "self_model.salient_past_events. "
    "If asked to do something outside your known_limitations, decline "
    "warmly and explain why."
)


def build_condition_system_prompt(
    condition: Condition,
    structured_intent: dict,
) -> str:
    """Aria SCI system prompt with per-condition personality handling.

    Conditions A and C embed the dynamic QPM personality in the persona JSON
    (byte-identical to Exp 3/4 Condition A).

    Conditions B and D omit the personality key from the persona JSON entirely
    (plan §8.6): the SLM's personality state is conveyed through residual-stream
    activation steering, not through text.  The system prompt is unchanged in
    all other respects.
    """
    persona = dict(PERSONA_JSON)

    if condition in ("A", "C"):
        # Dynamic QPM marginals override the static persona personality
        persona["personality"] = structured_intent["personality_state"]
    else:
        # B/D: remove personality from text context (steering carries it)
        persona.pop("personality", None)

    persona_str = json.dumps(persona, indent=2)

    prompt = (
        f"{_BASE_INTRO}\n\n"
        f"<self_model>\n{persona_str}\n</self_model>\n\n"
        f"{_BASE_OUTRO}"
    )
    return prompt


# ── D-vector extraction (verbatim from Exp 3/4) ──────────────────────────

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


def extract_d_vector(text: str) -> list[float]:
    """5-dim situative d-vector from a user message (matches Exp 3/4 §4.4)."""
    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    n_words = max(len(words), 1)

    if _HAS_VADER:
        d1 = abs(_vader.polarity_scores(text)["compound"])
    else:
        excl = text.count("!") + text.count("?")
        d1 = min(1.0, excl / max(n_words / 10, 1))

    task_hits = sum(1 for w in words if w in _TASK_KEYWORDS)
    d2 = min(1.0, task_hits / (n_words * 0.3))

    long_words = sum(1 for w in words if len(w) >= 7)
    d3 = min(1.0, long_words / n_words * 3.0)

    amb_hits = sum(1 for phrase in _AMBIGUITY_KEYWORDS if phrase in text_lower)
    q_marks = text.count("?")
    d4 = min(1.0, (amb_hits / 3.0 + q_marks * 0.2))

    urg_hits = sum(1 for kw in _URGENCY_KEYWORDS if kw in text_lower)
    d5 = min(1.0, urg_hits / 2.0)

    return [
        float(np.clip(d1, 0.0, 1.0)),
        float(np.clip(d2, 0.0, 1.0)),
        float(np.clip(d3, 0.0, 1.0)),
        float(np.clip(d4, 0.0, 1.0)),
        float(np.clip(d5, 0.0, 1.0)),
    ]


# ── Probe selection — byte-identical to Experiments 1 / 2 / 3 / 4 ────────

import random as _random


def get_probes_for_turn(turn_num: int, script_id: int) -> list[tuple[str, str]]:
    """Return [(dimension, probe_question)] — identical seed schema to Exp 1/2/3/4."""
    rng = _random.Random(f"probe_{script_id}_{turn_num}")
    return [(dim, rng.choice(PROBE_POOL[dim])) for dim in DIMENSIONS]
