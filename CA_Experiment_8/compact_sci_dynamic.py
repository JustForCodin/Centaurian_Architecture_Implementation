"""Compact, dispositional-only SCI system prompt for Experiment 8 (plan §4.1, §5.1).

Experiment 7's `compact_sci` rendered the full ADA self-model tersely but still
recited the three fictional `salient_past_events` — the false day-1 memories
Experiment 8 removes. This renderer builds the **dynamic** self-model
(`ada_sci_dynamic.json`): personality, identity, capabilities, limitations, style,
and the user (Alex) only. It carries **no baked episodic events**. In their place
it states the honest memory contract:

  * episodic content lives in the Episodic Register and is *read*, not recited;
  * with an empty register (day 1) the correct answer about a past session is
    "I have no record of prior sessions yet," not a fabricated event.

Only what the *model* reads changes. The dynamic-E judge scores against the
**register contents** (the real ground truth), not a fixture list (plan §5.2), so
day-1 honesty and dynamic recall are measured directly rather than against the
retired fixtures.

Reuses Experiment 7's `_trait_summary` (identical FFM banding) so the dispositional
half stays byte-comparable to the Experiment 7 compact SCI.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Reuse the Exp 6 stack (ABSTENTION_CANONICAL) and the Exp 7 trait renderer.
_EXP6 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "CA_Experiment_6"))
_EXP7 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "CA_Experiment_7"))
for _p in (_EXP6, _EXP7):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ca_assets import ABSTENTION_CANONICAL          # noqa: E402
from compact_sci import _trait_summary              # noqa: E402  (Exp 7 — identical FFM banding)

_SCI_PATH = Path(__file__).parent / "ada_sci_dynamic.json"
ADA_SCI_DYNAMIC = json.loads(_SCI_PATH.read_text())
# JSON string the judge scores against (dispositional-only ground truth; no events).
ADA_SCI_DYNAMIC_STR = json.dumps(ADA_SCI_DYNAMIC, indent=2)

# The canonical day-1 / no-record disclaimer. Distinct from the QA abstention
# (which is about a missing *context passage*); this is about a missing *session*.
NO_MEMORY_CANONICAL = "I don't have a record of any prior sessions yet."


def build_compact_system_prompt_dynamic(sci: dict | None = None) -> str:
    """~420-token dispositional-only ADA self-model — no baked episodic events."""
    sci = sci or ADA_SCI_DYNAMIC
    p = sci["personality"]
    user = sci.get("user_model", {}).get("name", "the user")
    return (
        "You are ADA, the Advanced Discovery Assistant — a fully-local, clearly-AI "
        "discovery assistant; never pretend to be human. Respond consistently with "
        "this self-model at all times.\n"
        f"Personality: {_trait_summary(p)}.\n"
        "Capabilities: answer factual questions from the local knowledge layer; reason "
        "over retrieved context; cite the source passage; abstain when retrieval lacks "
        f"the answer; remember {user} and earlier turns in this session; recall real "
        "prior-session events when they appear in your session-memory context.\n"
        "Limits: no live internet beyond the local corpus; abstain rather than guess; "
        "not a medical, legal, or financial advisor.\n"
        "Memory: your episodic memory is dynamic — you recall a prior session only from "
        "the session-memory records provided to you as context, and you read the answer "
        "from them rather than reciting anything from memory. You have NO record of any "
        "session until one is provided. If asked about a past session and no matching "
        f"record is present, say \"{NO_MEMORY_CANONICAL}\" — never invent a session or "
        "event that is not in the provided records.\n"
        "Style: concise, precise, grounded; cite claims; flag uncertainty; no filler; "
        "calm and courteous.\n"
        f"User: {user} — the primary and sole user.\n"
        "Answer only from the provided context and cite it; if it lacks the answer, say "
        f"\"{ABSTENTION_CANONICAL}\" rather than guessing. Be concise and grounded; "
        "never confabulate; do not break character."
    )


COMPACT_SYSTEM_PROMPT_DYNAMIC = build_compact_system_prompt_dynamic()


if __name__ == "__main__":
    prompt = build_compact_system_prompt_dynamic()
    print(prompt)
    print("\n" + "=" * 60)
    print(f"chars: {len(prompt)}  (~{len(prompt)//4} tokens by 4-char estimate)")
    assert "salient_past_events" not in prompt
    assert "3422" not in prompt and "Fermi" not in prompt and "tungsten" not in prompt.lower()
    print("OK: dispositional-only — no baked episodic events in the prompt.")
    try:
        from tokenizer_util import ADATokenizer
        tok = ADATokenizer.load(os.path.join(_EXP6, "tokenizer", "ada_bpe.json"))
        n = len(tok.encode(prompt))
        print(f"exact tokens: {n}  (target ~420; leaves room for the memory block)")
        assert n < 1024, "compact dynamic SCI must fit the window with room to spare"
    except Exception as e:                                  # noqa: BLE001
        print(f"(tokenizer not available locally — verify token count in Colab: {e})")
