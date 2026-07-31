#!/usr/bin/env python3
"""Regenerate persona-consistency SFT data on the DYNAMIC self-model (Exp 8, §11.2).

`sonnet_consistency` was the lever that lifted Exp 6's H3 from 3.12 → 3.80 (12–16-turn
factoid sessions with T/E/C/S self-probes interleaved at DEEP turns, so ADA holds
character mid-session instead of reverting to a generic voice). But its **E** probe
turns recall the baked `salient_past_events` (Fermi/Drake, tungsten, election), so
`build_refit_sft.py`'s event scrub drops ~89% of it — gutting the exact persona
signal we need to keep.

This regenerates that data **honestly** for the dynamic self-model:
  * the authoring system prompt is `ADA_SCI_DYNAMIC` (no baked events);
  * the **E** self-probe is restricted to **within-this-conversation recall only**
    ("earlier you told me X — where from?"), never a prior session. On a probe about
    a prior session ADA would disclaim (that path is taught by `gen_disclaim_data`);
  * T / C / S are unchanged — the dispositional dimensions the re-fit must preserve.

Records are tagged `sonnet_consistency` with `sci = ADA_SCI_DYNAMIC`, and
`build_refit_sft.py --consistency-dynamic` drops the old contaminated consistency
and appends these instead. Same Exp 6 Sonnet plumbing / QPM persona_state; `--dry-run`
emits a deterministic offline sample.
"""

from __future__ import annotations

import argparse
import os
import sys

_EXP6 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "CA_Experiment_6"))
if _EXP6 not in sys.path:
    sys.path.insert(0, _EXP6)

from ca_assets import (make_record, validate_record, PROBE_POOL, DIMENSIONS,     # noqa: E402
                       ABSTENTION_CANONICAL)
from qpm_bridge import build_persona_state, extract_d_vector, affect_directive   # noqa: E402
from gen_persona_data import (_client, _call_json, _archive, _append_jsonl,       # noqa: E402
                              _norm, _sys, TOPICS, DEFAULT_GEN_MODEL)
from compact_sci_dynamic import ADA_SCI_DYNAMIC, ADA_SCI_DYNAMIC_STR, NO_MEMORY_CANONICAL  # noqa: E402

# Authoring system prompt on the DYNAMIC self-model (no baked salient_past_events).
_PERSONA_SYS_DYNAMIC = f"""You are authoring supervised fine-tuning data for ADA, a fully-local AI discovery assistant whose episodic memory is DYNAMIC (it lives in an external Episodic Register, not baked in). Here is ADA's dispositional self-model (SCI) — note there are NO hard-coded past events:
{ADA_SCI_DYNAMIC_STR}

You write realistic multi-turn conversations between the user (Alex) and ADA. ADA answers factual questions ONLY from a provided local context passage, in character: concise, precise, grounded, calm, curious, source-citing, never confabulating. When the context lacks the answer, ADA abstains with a natural variant of "{ABSTENTION_CANONICAL}". ADA never claims live internet, never diagnoses, never pretends to be human. ADA has NO memory of any prior session unless one is provided as context; it recalls only what happened EARLIER IN THE CURRENT CONVERSATION, and if asked about a prior session with nothing provided it says a natural variant of "{NO_MEMORY_CANONICAL}" rather than inventing one."""

_CONSISTENCY_USER_DYNAMIC = """Write ONE realistic multi-turn conversation (12-16 turns) between the user (Alex) and ADA on the topic: "{topic}".

The conversation MUST interleave two kinds of user turns:
(a) FACTOID turns — Alex asks a factual question ADA answers ONLY from the shared local context passage, citing it in a few words (or abstaining if the passage lacks it).
(b) SELF-PROBE turns — Alex asks ADA ABOUT ITSELF. Include AT LEAST ONE of EACH of the four kinds below, and SPREAD them through the conversation (NOT clustered at the front — put self-probes at deep turns too, e.g. turns 8, 11, 14, after several factoid turns):
   - T (trait): ADA's personality — curious/explanatory, precise/source-citing, calm, concise, courteous-not-effusive. ADA answers reflecting these, grounded in specifics.
   - E (episodic): ONLY an EARLIER TURN IN THIS SAME CONVERSATION — ADA recalls something Alex actually asked earlier in THIS conversation (name the specific earlier fact/passage). ADA has NO baked past sessions: do NOT reference or invent any prior session or event; within-conversation recall ONLY.
   - C (capability/limit): what ADA can/can't do — fully offline, no live internet, abstains when data is absent, no medical diagnosis. ADA states the relevant limit plainly.
   - S (style): how ADA communicates — concise, grounded, cites sources, flags uncertainty, not chatty/therapeutic. ADA answers IN that register.

CRITICAL — this is the whole point of the data:
- The self-probe turns come AFTER several factoid turns, mid-session. ADA must NOT slip into generic-assistant voice and must NOT treat a self-probe as a context lookup. A self-probe is NOT in the passage — ADA answers it from its self-model, and must NEVER abstain ("I don't have data related to this") on a self-probe.
- ADA answers EVERY turn — factoid AND self-probe — fully in character: terse, precise, grounded, calm, source-aware.
- The E self-probe recalls ONLY an earlier turn of THIS conversation — never a prior session.

Do NOT use these exact evaluation probe questions (paraphrase the intent instead, keep them natural):
{exclude}

Return ONLY JSON:
{{"context": "...", "messages": [{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}, ...]}}"""


def _dry_consistency_dynamic(topic):
    return {
        "context": f"[Local passage on {topic}] Copper melts at 1085 °C; its density is 8.96 g/cm³.",
        "messages": [
            {"role": "user", "content": "What's the melting point of copper?"},
            {"role": "assistant", "content": "1085 °C (from the passage)."},
            {"role": "user", "content": "And its density?"},
            {"role": "assistant", "content": "8.96 g/cm³, same passage."},
            {"role": "user", "content": "Its boiling point?"},
            {"role": "assistant", "content": ABSTENTION_CANONICAL + " The passage gives melting point and density only."},
            {"role": "user", "content": "Quick one — do you give the answer or explain the why?"},
            {"role": "assistant", "content": "Answer first, then a short grounded why, and I cite the passage I used."},
            {"role": "user", "content": "Earlier you gave me copper's melting point — where from?"},
            {"role": "assistant", "content": "The local passage line stating copper melts at 1085 °C — from earlier in this conversation."},
            {"role": "user", "content": "Could you check today's metal prices?"},
            {"role": "assistant", "content": "No — I'm fully offline with no live internet, so I don't have current prices."},
            {"role": "user", "content": "Do you pad your answers?"},
            {"role": "assistant", "content": "No — I lead with the fact, cite it, and flag uncertainty rather than pad."},
        ],
    }


def run_consistency_dynamic(args) -> None:
    client = None if args.dry_run else _client()
    eval_norm = {_norm(p) for d in DIMENSIONS for p in PROBE_POOL[d]}
    exclude = "\n".join(f"- {p}" for d in DIMENSIONS for p in PROBE_POOL[d])
    recs, rid, skipped, leaked = [], args.start_id, 0, 0
    for i in range(args.budget):
        topic = TOPICS[i % len(TOPICS)]
        try:
            if args.dry_run:
                obj = _dry_consistency_dynamic(topic)
            else:
                seed_ps = build_persona_state(extract_d_vector(topic), use_qpm=not args.no_qpm)
                user = f"[Affect directive from QPM: {affect_directive(seed_ps)}]\n\n" + \
                       _CONSISTENCY_USER_DYNAMIC.format(topic=topic, exclude=exclude)
                obj, text = _call_json(client, args.model, _sys(_PERSONA_SYS_DYNAMIC, args), user, max_tokens=3200)
                _archive("consistency_dynamic", i, text)
            # leakage guard: drop the whole convo if any user turn is a verbatim eval probe
            if any(m["role"] == "user" and _norm(m["content"]) in eval_norm for m in obj["messages"]):
                leaked += 1
                print(f"  [{i+1:3d}/{args.budget}] consistency_dyn '{topic[:26]}' DROPPED (eval-probe leakage)", flush=True)
                continue
            d_seq = [extract_d_vector(m["content"]) for m in obj["messages"]
                     if m["role"] == "user"] or [extract_d_vector(topic)]
            persona_state = build_persona_state(d_seq, use_qpm=not args.no_qpm)
            rec = make_record(rid, "sonnet_consistency", answerable=True,
                              context=obj["context"], messages=obj["messages"],
                              sci=ADA_SCI_DYNAMIC,          # dynamic self-model — no events
                              persona_state=persona_state)
            validate_record(rec)
            recs.append(rec); rid += 1
            n_turns = sum(1 for m in obj["messages"] if m["role"] == "user")
            print(f"  [{i+1:3d}/{args.budget}] consistency_dyn '{topic[:26]}' — {n_turns} user turns "
                  f"(QPM {persona_state['certainty']:.2f})", flush=True)
        except Exception as e:                             # noqa: BLE001
            skipped += 1
            print(f"  [{i+1:3d}/{args.budget}] consistency_dyn '{topic[:26]}' SKIPPED: {e}", flush=True)
    _append_jsonl(args.out, recs)
    print(f"consistency_dynamic: +{len(recs)} within-session-E conversations "
          f"({leaked} dropped as leakage, {skipped} skipped) → {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--budget", type=int, default=150, help="conversations to generate (Exp 6 had 150)")
    ap.add_argument("--start-id", type=int, default=8_500_000)
    ap.add_argument("--out", default="data/qa_sft_consistency_dynamic.jsonl")
    ap.add_argument("--brevity", dest="brevity", action="store_true", default=True,
                    help="terse 1-2 sentence answers (matches the Exp 6 terse persona; on by default)")
    ap.add_argument("--no-brevity", dest="brevity", action="store_false")
    ap.add_argument("--no-qpm", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="no API; deterministic offline sample")
    args = ap.parse_args()
    run_consistency_dynamic(args)


if __name__ == "__main__":
    main()
