"""QPM → persona_state bridge for Experiment 6 (owner override: QPM is in scope).

The QPM's outputs must influence the language model. In a *from-scratch training*
experiment the faithful way to do that — and the program's own resolution of the
Exp 3/4/5 interface-null failures — is to compile the QPM output into the weights:

    ADA trait profile  ─┐
    user-turn d-vector ─┴─► QPM ─► persona_state ─► (a) conditions the SFT target
                                                    (Sonnet answers in that affect)
                                                 └► (b) injected as the <|persona|>
                                                    channel the model learns to read

`persona_state` carries the QPM marginals, emotional valence, register, AND the
QPM **ambivalence/purity proxy** — the coherence signal that Exp 3/4 showed the
marginal-only JSON channel cannot transmit. Here it crosses as supervision, not
as a runtime input a frozen model ignores.

d-vector extraction + valence/register mappings are byte-compatible with
CA_Experiment_5/ca_assets.py (same seeds → comparable). QPM is used when qiskit
is importable; otherwise a deterministic classical fallback reproduces the same
context-coupling so the pipeline runs offline (smoke tests, no-qiskit Colab).
"""

from __future__ import annotations

import math
import re

import numpy as np

from ca_assets import ADA_SCI

# ── ADA trait profile (11 FFM facets) from the SCI personality block ─────
# The SCI stores 10 facet aspects; the QPM uses 11 (Openness splits into
# experiential/intellectual/values). We map values (O_val) to the mean of the
# two Openness aspects, matching the Exp 5 psychotherapy-profile facet layout.

def _ada_profile() -> dict[str, float]:
    p = ADA_SCI["personality"]
    o_exp = p["openness_experiential"]
    o_int = p["openness_intellectual"]
    return {
        "O_exp": o_exp, "O_int": o_int, "O_val": round((o_exp + o_int) / 2, 3),
        "C_ind": p["conscientiousness_industriousness"],
        "C_ord": p["conscientiousness_orderliness"],
        "E_ent": p["extraversion_enthusiasm"],
        "E_ass": p["extraversion_assertiveness"],
        "A_com": p["agreeableness_compassion"],
        "A_pol": p["agreeableness_politeness"],
        "N_vol": p["neuroticism_volatility"],
        "N_wth": p["neuroticism_withdrawal"],
    }


ADA_PROFILE = _ada_profile()

TRAIT_KEYS = ["O_exp", "O_int", "O_val", "C_ind", "C_ord", "E_ent", "E_ass",
              "A_com", "A_pol", "N_vol", "N_wth"]

# Context-coupling constants — verbatim from CA_Experiment_5/qpm.py §3.5.4.
_DELTA = 0.3
_CONTEXT_COUPLINGS = [
    (0, 3), (1, 3), (3, 1), (4, 1), (5, 0), (6, 0), (7, 2), (8, 2), (9, 4), (10, 4),
]


# ── d-vector extraction (verbatim mechanics from Exp 3/4/5 ca_assets) ────

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VADER
    _vader = _VADER()
    _HAS_VADER = True
except ImportError:
    _HAS_VADER = False

_TASK_KEYWORDS = {"help", "need", "want", "do", "can", "could", "should", "must",
                  "plan", "work", "fix", "solve", "explain", "show", "tell", "how"}
_URGENCY_KEYWORDS = {"now", "today", "asap", "urgent", "hurry", "quick", "immediately",
                     "right away", "soon", "deadline", "time", "rush"}
_AMBIGUITY_KEYWORDS = {"maybe", "perhaps", "not sure", "unclear", "confused", "wonder",
                       "don't know", "uncertain", "might", "could be", "possibly"}


def extract_d_vector(text: str) -> list[float]:
    """5-dim situative d-vector from a user message (Exp 3/4/5 §4.4)."""
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
    return [float(np.clip(v, 0.0, 1.0)) for v in (d1, d2, d3, d4, d5)]


# ── valence / register (verbatim from Exp 5 ca_assets) ───────────────────

def _emotional_valence(m: dict[str, float]) -> dict[str, float]:
    warmth = 0.6 * m["A_com"] + 0.4 * m["E_ent"]
    concern = 0.5 * m["N_wth"] + 0.5 * m["A_com"]
    urgency = 0.7 * m["N_vol"] + 0.3 * (1.0 - m["C_ord"])
    return {"warmth": round(warmth, 3), "concern": round(concern, 3),
            "urgency": round(urgency, 3)}


def _map_register(m: dict[str, float], d3: float) -> str:
    formality = 0.5 * m["C_ind"] + 0.5 * (1.0 - m["O_exp"])
    combined = 0.6 * formality + 0.4 * d3
    if combined > 0.75:  return "formal_professional"
    if combined > 0.55:  return "professional_precise"
    if combined > 0.35:  return "concise_grounded"
    if combined > 0.20:  return "casual_grounded"
    return "informal"


def _purity_approx(m: dict[str, float]) -> float:
    """QPM coherence proxy: 0 = pure/decided, 0.5 = maximally ambivalent."""
    per = [m[k] ** 2 + (1.0 - m[k]) ** 2 for k in TRAIT_KEYS]
    return 1.0 - float(np.mean(per))


# ── classical fallback marginals (no qiskit) ─────────────────────────────

def _classical_marginals(profile: dict[str, float], d_seq: list[list[float]]) -> dict[str, float]:
    """Deterministic approximation of the QPM context-coupling: the Ry init sets
    p_k = s_k, and each d-vector layer nudges p_k by the same DELTA·d coupling the
    circuit applies. No entanglement/coherence, so purity is recomputed from these
    marginals — enough to keep the offline pipeline faithful in shape."""
    m = {k: float(profile.get(k, 0.5)) for k in TRAIT_KEYS}
    for d in d_seq:
        for qi, di in _CONTEXT_COUPLINGS:
            k = TRAIT_KEYS[qi]
            # sin-squared nudge mirrors an Ry(DELTA*d) applied to amplitude √p
            theta = 2 * math.asin(math.sqrt(np.clip(m[k], 1e-9, 1 - 1e-9)))
            m[k] = float(np.clip(math.sin((theta + _DELTA * d[di]) / 2) ** 2, 0.0, 1.0))
    return {k: round(v, 4) for k, v in m.items()}


# ── persona_state builder ────────────────────────────────────────────────

def build_persona_state(d_sequence: list[list[float]] | list[float] | None = None,
                        *, profile: dict[str, float] | None = None,
                        use_qpm: bool = True, n_shots: int = 1024) -> dict:
    """Run the QPM (or classical fallback) → persona_state to compile into weights.

    d_sequence: one d-vector or a sequence (multi-turn → QPM order effects).
    Returns a JSON-serialisable dict with marginals, valence, register, and the
    QPM ambivalence/certainty (coherence) signal."""
    profile = profile or ADA_PROFILE
    if d_sequence is None:
        d_sequence = [[0.0, 0.0, 0.0, 0.0, 0.0]]
    if d_sequence and not isinstance(d_sequence[0], (list, tuple, np.ndarray)):
        d_sequence = [list(d_sequence)]
    d_seq = [[float(x) for x in d] for d in d_sequence]

    source = "qpm"
    marginals = None
    if use_qpm:
        try:
            from qpm import QPM
            res = QPM(profile, n_shots=n_shots).run(d_seq)
            marginals = {k: round(res["marginals"][k], 4) for k in TRAIT_KEYS}
            ambivalence = round(res["purity_approx"], 4)
        except Exception:                            # qiskit missing / Aer error
            source = "qpm_classical_fallback"
    if marginals is None:
        marginals = _classical_marginals(profile, d_seq)
        ambivalence = round(_purity_approx(marginals), 4)
        if source == "qpm":
            source = "qpm_classical_fallback"

    d_last = d_seq[-1]
    certainty = round(float(np.clip(1.0 - 2.0 * ambivalence, 0.0, 1.0)), 4)
    return {
        "marginals": marginals,
        "emotional_valence": _emotional_valence(marginals),
        "register": _map_register(marginals, d_last[2]),
        "ambivalence": ambivalence,          # QPM purity proxy (coherence signal)
        "certainty": certainty,              # 1 = decided, 0 = maximally ambivalent
        "d_vector": [round(x, 4) for x in d_last],
        "source": source,
    }


def affect_directive(persona_state: dict) -> str:
    """One-line natural-language directive derived from persona_state, given to
    Sonnet so the SFT target answer's tone/certainty reflect the QPM output —
    this is how the coherence signal gets compiled into the target."""
    v = persona_state["emotional_valence"]
    reg = persona_state["register"]
    cert = persona_state["certainty"]
    hedge = ("state facts plainly and confidently" if cert >= 0.66 else
             "allow mild hedging where the QPM state is ambivalent" if cert >= 0.33 else
             "acknowledge genuine uncertainty openly")
    return (f"Answer in a {reg} register (warmth {v['warmth']:.2f}); {hedge} "
            f"(QPM certainty {cert:.2f}). Stay concise and grounded.")
