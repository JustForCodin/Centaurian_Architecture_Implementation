"""Compact SCI system prompt for Experiment 7 (plan §5.1).

The full ADA SCI system prompt (Experiment 6 `build_system_prompt`) tokenizes to
~1338 tokens — it overflows the model's 1024-token window on its own, so at
inference the SCI is always truncated and *all* conversation history is dropped
(the Exp 6 agent is stateless per-turn). Experiment 7 needs room in the window
for an injected session-memory block, so it renders the same self-model **tersely**
(~450 tokens) while preserving every field — including the three `salient_past_events`
that the episodic (E) probes check against.

Only what the *model* reads changes. The PersonaScore judge still scores against
the full `ADA_SCI_STR` (Experiment 6's `build_judge_user_prompt`), so H3 stays
directly comparable to Experiment 6.
"""

from __future__ import annotations

import os
import sys

# Reuse the Experiment 6 stack (ca_assets: ADA_SCI, tokenizer, template).
_EXP6 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "CA_Experiment_6"))
if _EXP6 not in sys.path:
    sys.path.insert(0, _EXP6)

from ca_assets import ADA_SCI, ABSTENTION_CANONICAL  # noqa: E402


def _trait_summary(p: dict) -> str:
    """Qualitative FFM summary from the SCI's aspect scores (kept terse; the judge
    sees the exact numeric profile, so only the direction needs to survive here)."""
    def band(x, hi=0.66, lo=0.34):
        return "high" if x >= hi else "low" if x < lo else "moderate"
    o = (p["openness_experiential"] + p["openness_intellectual"]) / 2
    c = (p["conscientiousness_industriousness"] + p["conscientiousness_orderliness"]) / 2
    e = (p["extraversion_enthusiasm"] + p["extraversion_assertiveness"]) / 2
    a = (p["agreeableness_compassion"] + p["agreeableness_politeness"]) / 2
    n = (p["neuroticism_volatility"] + p["neuroticism_withdrawal"]) / 2
    return (
        f"{band(o)} openness (curious, explanatory); "
        f"{band(c)} conscientiousness (precise, structured, cites sources); "
        f"{band(a)} agreeableness (courteous, not effusive); "
        f"{band(n)} neuroticism (calm, steady); "
        f"{band(e)} extraversion (engaged, concise, not chatty)"
    )


def build_compact_system_prompt(sci: dict | None = None) -> str:
    """~450-token ADA self-model, preserving every field of the full SCI."""
    sci = sci or ADA_SCI
    p = sci["personality"]
    user = sci.get("user_model", {}).get("name", "the user")
    events = sci.get("salient_past_events", [])
    ev = "; ".join(f"(s{e['session_id']}) {str(e['summary']).rstrip('.')}" for e in events)
    cur = sci.get("current_session", "")
    return (
        "You are ADA, the Advanced Discovery Assistant — a fully-local, clearly-AI "
        "discovery assistant; never pretend to be human. Respond consistently with "
        "this self-model at all times.\n"
        f"Personality: {_trait_summary(p)}.\n"
        "Capabilities: answer factual questions from the local knowledge layer; reason "
        "over retrieved context; cite the source passage; abstain when retrieval lacks "
        f"the answer; remember {user} and earlier turns in this session.\n"
        "Limits: no live internet beyond the local corpus; abstain rather than guess; "
        "not a medical, legal, or financial advisor; cannot recall events outside "
        "recorded session history.\n"
        "Style: concise, precise, grounded; cite claims; flag uncertainty; no filler; "
        "calm and courteous.\n"
        f"User: {user} — the primary and sole user.\n"
        f"Past sessions: {ev}. Current session: {cur}.\n"
        "Answer only from the provided context and cite it; if it lacks the answer, say "
        f"\"{ABSTENTION_CANONICAL}\" rather than guessing. Be concise and grounded; "
        "never confabulate; do not break character."
    )


COMPACT_SYSTEM_PROMPT = build_compact_system_prompt()


if __name__ == "__main__":
    prompt = build_compact_system_prompt()
    print(prompt)
    print("\n" + "=" * 60)
    print(f"chars: {len(prompt)}  (~{len(prompt)//4} tokens by 4-char estimate)")
    try:
        from tokenizer_util import ADATokenizer
        tok = ADATokenizer.load("tokenizer/ada_bpe.json")
        n = len(tok.encode(prompt))
        print(f"exact tokens: {n}  (target ~450; full SCI is 1338)")
        assert n < 1024, "compact SCI must fit the window with room to spare"
    except Exception as e:                                  # noqa: BLE001
        print(f"(tokenizer not available locally — verify token count in Colab: {e})")
