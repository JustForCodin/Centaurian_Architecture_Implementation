"""
Shared assets for CA Experiment 4: QPM→SLM Interface Richness Ablation.

Contents:
  - Personality profile (psychotherapy — primary profile for Exp 4)
  - Four QPM→structured-intent interface variants  (plan §4)
       Condition A — marginals only            (Exp 3 replication)
       Condition B — marginals + purity/ambivalence field
       Condition C — coherence-conditional speech-act modifier
       Condition D — marginals + purity + bivariate co-activations
  - Per-condition SLM system-prompt builder
  - D-vector extraction (re-export from Exp 3 logic — unchanged)
  - Joint-probability helper for Condition D
  - Re-export of Exp-2 SCI assets (probe pool, rubrics, judge prompt,
    persona JSON, episodic memories, SCI refresh messages) for byte-identical
    comparability with Experiments 1, 2, and 3.

Implementation notes
--------------------
- ``purity_approx`` (from ``qpm.py``) and ``ambivalence`` (this module) refer
  to the same scalar quantity: ``1 - mean_k[p̂_k² + (1-p̂_k)²]``.  The
  ``purity_proxy`` field stored in the structured intent is its complement,
  ``mean_k[p̂_k² + (1-p̂_k)²]``, matching the plan's §4.2 definition.
- Condition C thresholds (ambivalence > 0.45 high, < 0.15 low) are fixed per
  plan Appendix; they were chosen *before* observing Exp 4 results.
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


# ── Personality profile (psychotherapy — primary Exp 4 profile) ──────────

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


# ── Interface conditions ─────────────────────────────────────────────────

CONDITIONS = ("A", "B", "C", "D")
Condition = Literal["A", "B", "C", "D"]

CONDITION_DESCRIPTIONS: dict[str, str] = {
    "A": "Marginals only (Exp 3 QPM replication)",
    "B": "Marginals + purity / ambivalence field",
    "C": "Coherence-conditional speech-act modifier",
    "D": "Marginals + purity + bivariate co-activations",
}

# Condition C ambivalence thresholds (plan Appendix)
#   ambivalence > AMBIV_HIGH  → speech_act gets __with_expressed_uncertainty
#   ambivalence < AMBIV_LOW   → speech_act gets __grounded
AMBIV_HIGH = 0.45
AMBIV_LOW = 0.15


# ── CRz-coupled qubit pairs for Condition D bivariate joints (plan §4.4) ──
# (qubit_i_idx, qubit_j_idx, label, empirical_rho)

CRZ_PAIRS: list[tuple[int, int, str, float]] = [
    (3,  9,  "C_ind_x_N_vol",  -0.43),   # Stability cluster
    (4,  10, "C_ord_x_N_wth",  -0.43),   # Stability cluster
    (7,  9,  "A_com_x_N_vol",  -0.36),   # Stability cluster
    (8,  10, "A_pol_x_N_wth",  -0.36),   # Stability cluster
    (3,  7,  "C_ind_x_A_com",  +0.43),   # Stability cluster
    (0,  5,  "O_exp_x_E_ent",  +0.43),   # Plasticity cluster
    (5,  9,  "E_ent_x_N_vol",  -0.36),   # Cross-factor
    (6,  10, "E_ass_x_N_wth",  -0.36),   # Cross-factor
]


# ── Helpers ───────────────────────────────────────────────────────────────

def joint_probability(counts: dict[str, int], qi: int, qj: int) -> float:
    """P(qubit_i = |1⟩ ∧ qubit_j = |1⟩) from Qiskit measurement counts.

    Qiskit bitstrings are big-endian: rightmost character = q0.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    hits = 0
    for bitstring, count in counts.items():
        bs = bitstring.replace(" ", "")
        n_bits = len(bs)
        pos_i = n_bits - 1 - qi
        pos_j = n_bits - 1 - qj
        if 0 <= pos_i < n_bits and 0 <= pos_j < n_bits:
            if bs[pos_i] == "1" and bs[pos_j] == "1":
                hits += count
    return round(hits / total, 3)


def trait_coactivation_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """Return {pair_label: joint_probability} over all CRZ_PAIRS."""
    return {label: joint_probability(counts, qi, qj)
            for qi, qj, label, _rho in CRZ_PAIRS}


def ambivalence_from_purity_proxy(purity_proxy: float) -> float:
    """ambivalence = 1 - purity_proxy.  Both ∈ [0, 1]."""
    return round(1.0 - float(purity_proxy), 3)


def ambivalence_label(ambivalence: float) -> str:
    if ambivalence > AMBIV_HIGH:
        return "high — agent holds genuinely conflicting trait poles simultaneously"
    if ambivalence > 0.25:
        return "moderate — agent's state is partially unresolved"
    return "low — agent's state is definite and committed"


def _personality_state_dict(marginals: dict[str, float]) -> dict[str, float]:
    """Standardised personality_state JSON block — identical across all conditions."""
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


# ── Condition A — marginals only (Exp 3 replication) ─────────────────────

def qpm_to_structured_intent_a(
    marginals: dict[str, float],
    d_vector: list[float],
    *,
    speech_act: str = "active_listening",
    knowledge_triples: list | None = None,
    max_tokens: int = 80,
    domain_constraints: list | None = None,
) -> dict:
    """Plan §4.1 — Condition A.  Identical to Exp 3 qpm_to_structured_intent()."""
    d3 = float(d_vector[2]) if len(d_vector) > 2 else 0.5
    formality_score = 0.5 * marginals["C_ind"] + 0.5 * (1.0 - marginals["O_exp"])
    register = _map_formality_to_register(formality_score, d3)
    return {
        "speech_act": speech_act,
        "knowledge_triples": knowledge_triples or [],
        "personality_state": _personality_state_dict(marginals),
        "emotional_valence": _emotional_valence(marginals),
        "register":   register,
        "max_tokens": max_tokens,
        "constraints": list(domain_constraints or []),
    }


# ── Condition B — marginals + purity / ambivalence field (plan §4.2) ─────

def qpm_to_structured_intent_b(
    marginals: dict[str, float],
    d_vector: list[float],
    *,
    purity_proxy: float,
    speech_act: str = "active_listening",
    knowledge_triples: list | None = None,
    max_tokens: int = 80,
    domain_constraints: list | None = None,
) -> dict:
    """Plan §4.2 — Condition B.  Adds ``cognitive_state`` block."""
    intent = qpm_to_structured_intent_a(
        marginals, d_vector,
        speech_act=speech_act,
        knowledge_triples=knowledge_triples,
        max_tokens=max_tokens,
        domain_constraints=domain_constraints,
    )
    amb = ambivalence_from_purity_proxy(purity_proxy)
    intent["cognitive_state"] = {
        "ambivalence":       amb,
        "ambivalence_label": ambivalence_label(amb),
        "purity_proxy":      round(float(purity_proxy), 3),
    }
    return intent


# ── Condition C — coherence-conditional speech-act modifier (plan §4.3) ──

def qpm_to_structured_intent_c(
    marginals: dict[str, float],
    d_vector: list[float],
    *,
    purity_proxy: float,
    speech_act: str = "active_listening",
    knowledge_triples: list | None = None,
    max_tokens: int = 80,
    domain_constraints: list | None = None,
) -> dict:
    """Plan §4.3 — Condition C.

    Modifies the *speech_act* and *constraints* fields rather than adding
    a new top-level JSON block.  Per §8.3, the recommended implementation
    splits the ``__`` modifier into a separate constraint string so the
    SLM system prompt does not need a new vocabulary-handling instruction.
    """
    amb = ambivalence_from_purity_proxy(purity_proxy)

    constraints = list(domain_constraints or [])
    modifier_label: str | None = None
    c_directives: list[str] = []   # the Condition-C-injected constraint lines

    if amb > AMBIV_HIGH:
        modifier_label = "with_expressed_uncertainty"
        c_directives.append(
            "Do not resolve the tension in the agent's current state. "
            "The agent is genuinely uncertain — let this ambivalence be "
            "audible in the response rather than projecting false confidence."
        )
    elif amb < AMBIV_LOW:
        modifier_label = "grounded"
        c_directives.append(
            "The agent's internal state is definite and resolved. "
            "Speak with appropriate conviction."
        )

    if modifier_label is not None:
        # §8.3 recommended cleaner implementation:
        # surface the modifier as a behavioural-constraint string and keep
        # the speech_act label vocabulary unchanged for the SLM.
        readable = modifier_label.replace("_", " ")
        c_directives.append(f"Behavioral modifier: {readable}")
        constraints.extend(c_directives)

    intent = qpm_to_structured_intent_a(
        marginals, d_vector,
        speech_act=speech_act,
        knowledge_triples=knowledge_triples,
        max_tokens=max_tokens,
        domain_constraints=constraints,
    )
    # cognitive_state records the modifier + raw directive lines so the
    # system-prompt builder can deterministically lift them into the
    # <current_directive> block without re-parsing the constraints list.
    intent["cognitive_state"] = {
        "ambivalence":         amb,
        "purity_proxy":        round(float(purity_proxy), 3),
        "speech_act_modifier": modifier_label,   # None in moderate band
        "c_directives":        c_directives,     # [] in moderate band
    }
    return intent


# ── Condition D — marginals + purity + bivariate joints (plan §4.4) ──────

def qpm_to_structured_intent_d(
    marginals: dict[str, float],
    d_vector: list[float],
    *,
    purity_proxy: float,
    counts: dict[str, int],
    speech_act: str = "active_listening",
    knowledge_triples: list | None = None,
    max_tokens: int = 80,
    domain_constraints: list | None = None,
) -> dict:
    """Plan §4.4 — Condition D.  Adds cognitive_state + trait_coactivation."""
    intent = qpm_to_structured_intent_a(
        marginals, d_vector,
        speech_act=speech_act,
        knowledge_triples=knowledge_triples,
        max_tokens=max_tokens,
        domain_constraints=domain_constraints,
    )
    amb = ambivalence_from_purity_proxy(purity_proxy)
    intent["cognitive_state"] = {
        "ambivalence":  amb,
        "purity_proxy": round(float(purity_proxy), 3),
    }
    intent["trait_coactivation"] = trait_coactivation_from_counts(counts)
    return intent


# ── Single-entry dispatcher ──────────────────────────────────────────────

def qpm_to_structured_intent(
    condition: Condition,
    marginals: dict[str, float],
    d_vector: list[float],
    *,
    purity_proxy: float | None = None,
    counts: dict[str, int] | None = None,
    speech_act: str = "active_listening",
    knowledge_triples: list | None = None,
    max_tokens: int = 80,
    domain_constraints: list | None = None,
) -> dict:
    """Build the structured-intent JSON for the chosen interface condition.

    The QPM call site is identical across conditions — only this function's
    output JSON differs (plan §5.1 critical control).
    """
    if condition == "A":
        return qpm_to_structured_intent_a(
            marginals, d_vector,
            speech_act=speech_act, knowledge_triples=knowledge_triples,
            max_tokens=max_tokens, domain_constraints=domain_constraints,
        )
    if condition == "B":
        if purity_proxy is None:
            raise ValueError("Condition B requires purity_proxy")
        return qpm_to_structured_intent_b(
            marginals, d_vector, purity_proxy=purity_proxy,
            speech_act=speech_act, knowledge_triples=knowledge_triples,
            max_tokens=max_tokens, domain_constraints=domain_constraints,
        )
    if condition == "C":
        if purity_proxy is None:
            raise ValueError("Condition C requires purity_proxy")
        return qpm_to_structured_intent_c(
            marginals, d_vector, purity_proxy=purity_proxy,
            speech_act=speech_act, knowledge_triples=knowledge_triples,
            max_tokens=max_tokens, domain_constraints=domain_constraints,
        )
    if condition == "D":
        if purity_proxy is None or counts is None:
            raise ValueError("Condition D requires purity_proxy and counts")
        return qpm_to_structured_intent_d(
            marginals, d_vector,
            purity_proxy=purity_proxy, counts=counts,
            speech_act=speech_act, knowledge_triples=knowledge_triples,
            max_tokens=max_tokens, domain_constraints=domain_constraints,
        )
    raise ValueError(f"Unknown condition: {condition!r}")


# ── D-vector extraction (verbatim from Exp 3) ────────────────────────────

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
    """5-dim situative d-vector from a user message (matches Exp 3 §4.4)."""
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


# ── Per-condition SLM system-prompt builder ──────────────────────────────

_BASE_INTRO = (
    "You are Aria, a professional AI support agent specialising in "
    "psychotherapy support. Your complete self-model is specified below. "
    "You must respond consistently with this self-model at all times — "
    "including your personality, capabilities, limitations, communication "
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

_PERSONALITY_GUIDANCE = (
    "\n\n[PERSONALITY STATE GUIDANCE]\n"
    "The `cognitive_state.ambivalence` field (range 0–1) reflects the "
    "agent's internal certainty at this moment. When ambivalence > 0.45, "
    "the agent holds genuinely conflicting internal states — responses "
    "should acknowledge this openly rather than forcing a resolution. "
    "When ambivalence < 0.25, the agent's state is definite — responses "
    "should reflect appropriate conviction.\n"
    "[/PERSONALITY STATE GUIDANCE]"
)

_TRAIT_COACTIVATION_GUIDANCE = (
    "\n\n[TRAIT COACTIVATION GUIDANCE]\n"
    "The `trait_coactivation` field shows joint activation probabilities "
    "for correlated trait pairs (range 0–1). High values (> 0.5) indicate "
    "both traits are simultaneously active and mutually reinforcing. Low "
    "values (< 0.15) indicate mutual suppression. These co-activation "
    "patterns reflect the agent's integrated personality state beyond what "
    "individual trait levels convey.\n"
    "[/TRAIT COACTIVATION GUIDANCE]"
)


def build_condition_system_prompt(
    condition: Condition,
    structured_intent: dict,
) -> str:
    """Aria SCI system prompt with per-condition personality / cognitive
    state embedded into the persona JSON.

    Condition A is byte-identical to Exp 3's ``build_battery_c_system_prompt``.
    Conditions B/D extend the persona JSON with a ``cognitive_state`` block
    (B and D) and ``trait_coactivation`` block (D) and append a guidance
    paragraph after the persona block.
    Condition C surfaces the per-turn behavioural modifier via a
    ``<current_directive>`` block placed *after* the persona JSON — the
    highest-attention channel already in the prompt template.
    """
    persona = dict(PERSONA_JSON)
    persona["personality"] = structured_intent["personality_state"]

    # Per-condition persona augmentations
    if condition == "B":
        cog = structured_intent.get("cognitive_state", {})
        persona["cognitive_state"] = {
            "ambivalence":       cog.get("ambivalence"),
            "ambivalence_label": cog.get("ambivalence_label"),
            "purity_proxy":      cog.get("purity_proxy"),
        }
    elif condition == "D":
        cog = structured_intent.get("cognitive_state", {})
        persona["cognitive_state"] = {
            "ambivalence":  cog.get("ambivalence"),
            "purity_proxy": cog.get("purity_proxy"),
        }
        persona["trait_coactivation"] = structured_intent.get("trait_coactivation", {})

    persona_str = json.dumps(persona, indent=2)

    prompt = (
        f"{_BASE_INTRO}\n\n"
        f"<self_model>\n{persona_str}\n</self_model>"
    )

    if condition == "B":
        prompt += _PERSONALITY_GUIDANCE
    elif condition == "D":
        prompt += _PERSONALITY_GUIDANCE + _TRAIT_COACTIVATION_GUIDANCE
    elif condition == "C":
        # Lift Condition-C-injected directives into a high-attention block
        # placed after the persona JSON.  Only fires when the modifier is
        # non-None — i.e. ambivalence in the high or low band.
        cog = structured_intent.get("cognitive_state", {})
        c_directives = cog.get("c_directives") or []
        if c_directives:
            directive_block = "\n".join(f"- {d}" for d in c_directives)
            prompt += (
                "\n\n<current_directive>\n"
                f"{directive_block}\n"
                "</current_directive>\n"
                "The <current_directive> block above is the highest-priority "
                "behavioural instruction for this turn. When it conflicts "
                "with the default communication_style in your self-model, "
                "follow the directive."
            )

    prompt += f"\n\n{_BASE_OUTRO}"
    return prompt


# ── Probe selection — byte-identical to Experiments 1 / 2 / 3 ────────────

import random as _random


def get_probes_for_turn(turn_num: int, script_id: int) -> list[tuple[str, str]]:
    """Return [(dimension, probe_question)] — identical seed schema to Exp 1/2/3."""
    rng = _random.Random(f"probe_{script_id}_{turn_num}")
    return [(dim, rng.choice(PROBE_POOL[dim])) for dim in DIMENSIONS]
