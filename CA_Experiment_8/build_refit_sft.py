#!/usr/bin/env python3
"""Assemble the Stage-C re-fit SFT set that REMOVES the baked episodic fixtures
(plan §4.1.2, §11.2).

Experiment 6 memorised three fictional `salient_past_events` (Fermi/Drake, tungsten
3422 °C, an election abstention) two ways:

  1. as the **self-model** — every persona record renders `rec["sci"]` (the full
     `ADA_SCI`, events included) as its system prompt (`SFTDataset` →
     `format_chat(..., sci=rec["sci"])`), so the events are in-context on every step;
  2. as **recited content** — `sonnet_recall` records, and the E turns of some
     `sonnet_persona` / `sonnet_consistency` conversations, state the events as fact.

Removing the events from the prompt alone leaves them memorised (plan §4.1.2), and
leaving them in the training text re-teaches them. So the re-fit set is the Exp 6
SFT set transformed:

  * **swap the self-model** on every kept record → `ADA_SCI_DYNAMIC` (dispositional
    only, no events), so no step ever renders the fixtures as ADA's memory;
  * **drop `sonnet_recall`** wholesale (pure event recitation — no salvage);
  * **scrub baked-event references** from the remaining records: a record is dropped
    if any *assistant* turn mentions a fixture (Fermi/Drake, tungsten/3422 °C,
    the election abstention). This keeps the clean persona/style/consistency data
    that gives the voice its lift while removing the confabulation-teaching turns;
  * **append the day-1 disclaim data** (`gen_disclaim_data.py`, `sonnet_disclaim`),
    the positive signal for H1 (graceful "no record of prior sessions yet").

Init from `sft_ada` and run `train_sft.py` on the output → `sft_ada_dynamic`
(plan §4.1.2). The re-fit is short (§11.2: ~250–400 steps) to un-bake without the
Exp 7 Phase-B overfit.

The event scrubber is intentionally conservative (drops the whole record on any
match) and reports its counts — silent partial scrubbing would leave a fixture in a
mid-conversation turn and quietly re-teach it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

_EXP6 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "CA_Experiment_6"))
if _EXP6 not in sys.path:
    sys.path.insert(0, _EXP6)

from compact_sci_dynamic import ADA_SCI_DYNAMIC          # noqa: E402

# Fixtures baked in Exp 6 (ada_sci.json salient_past_events). A record whose
# assistant turns hit any of these is teaching a false memory → dropped.
_EVENT_PATTERNS = [
    r"\bfermi\b", r"\bdrake\b",                          # session 4
    r"\btungsten\b", r"\b3422\b",                        # session 9
    r"\belection\b",                                     # session 12
]
_EVENT_RE = re.compile("|".join(_EVENT_PATTERNS), re.IGNORECASE)

# Sources removed entirely (pure baked-event recitation / Phase-B dead end).
_DROP_SOURCES = {"sonnet_recall", "sonnet_memrecall"}


def _mentions_event(rec: dict) -> bool:
    """True if any ASSISTANT turn recites a baked fixture (user turns don't teach)."""
    for m in rec.get("messages", []):
        if m.get("role") == "assistant" and _EVENT_RE.search(m.get("content", "")):
            return True
    return False


def build(in_path: Path, disclaim_path: Path | None, out_path: Path,
          keep_reading: bool) -> None:
    kept, drop_source, drop_event, rewrote = [], Counter(), Counter(), 0
    reading_sources = {"squad2", "nq", "msmarco", "triviaqa", "hotpotqa"}

    for line in in_path.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        src = rec.get("source", "?")
        if src in _DROP_SOURCES:
            drop_source[src] += 1
            continue
        if not keep_reading and src in reading_sources:
            drop_source[src] += 1                         # reading is the span head's job now
            continue
        if _mentions_event(rec):
            drop_event[src] += 1
            continue
        rec["sci"] = ADA_SCI_DYNAMIC                       # swap self-model: no events, ever
        rewrote += 1
        kept.append(rec)

    n_exp6 = len(kept)                                     # the transformed Exp-6 records
    n_disclaim = 0
    if disclaim_path and disclaim_path.exists():
        for line in disclaim_path.open():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["sci"] = ADA_SCI_DYNAMIC                   # already dynamic, but be defensive
            kept.append(rec); n_disclaim += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")

    by_src = Counter(r["source"] for r in kept)
    print("=" * 68)
    print(f"re-fit SFT assembled → {out_path}")
    print(f"  kept {len(kept)} records (self-model swapped → dynamic on {rewrote}; "
          f"+{n_disclaim} disclaim)")
    print(f"  dropped by source : {dict(drop_source)}")
    print(f"  dropped (event mention in assistant turn): {dict(drop_event)} "
          f"[total {sum(drop_event.values())}]")
    print(f"  kept by source    : {dict(by_src)}")
    # Guardrail: the transformed Exp-6 records must be event-free. Disclaim records
    # are exempt — a disclaim may name a topic while DENYING memory of it ("I have no
    # record of the Fermi paradox from an earlier session"), which is honest, not a
    # recited false memory; the scrub targets records that assert an event as real.
    leaks = sum(1 for r in kept[:n_exp6] if _mentions_event(r))
    assert leaks == 0, f"{leaks} kept Exp-6 records still recite a baked event — scrub failed"
    print(f"  event-leak check  : {leaks} (must be 0, Exp-6 records) ✓")
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=os.path.join(_EXP6, "data", "qa_sft.jsonl"),
                    help="Experiment 6 SFT set (all sources)")
    ap.add_argument("--disclaim", default="data/qa_sft_disclaim.jsonl",
                    help="day-1 disclaim records from gen_disclaim_data.py")
    ap.add_argument("--out", default="data/qa_sft_dynamic.jsonl")
    ap.add_argument("--keep-reading", action="store_true",
                    help="keep squad2/nq/... reading records (default: drop — the span "
                         "head owns reading; a huge SQuAD set drowns the persona re-fit)")
    args = ap.parse_args()
    build(Path(args.in_path), Path(args.disclaim), Path(args.out), args.keep_reading)


if __name__ == "__main__":
    main()
