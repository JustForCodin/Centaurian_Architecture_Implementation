#!/usr/bin/env python3
"""Generate day-1 honest-disclaim SFT data for Experiment 8 (plan §5.1, §11.5).

The re-fit that removes the baked `salient_past_events` (build_refit_sft.py) takes
the *recitation* away, but removing a behaviour is not the same as teaching the
right one. On a first session — empty Episodic Register — the correct answer to
"last time, what did you tell me about X?" is **"I have no record of prior
sessions yet,"** said in ADA's calm, grounded voice. Nothing in the Experiment 6
persona data teaches that (it only ever *recited* events), so H1 (no false
memories, ≥ 90% correct disclaim) needs a small positive signal. This is it.

These records are the honesty analogue of the QA-refusal data: varied, in-character
disclaimers over past-session probes with **no** provided session record. They are
tagged `sonnet_disclaim` and carry an empty context (there is nothing to read — the
whole point is that the register is empty). The eval E probes are excluded verbatim
(leakage guard), exactly as `run_recall` does for its data.

Reuses the Experiment 6 Sonnet plumbing (`_client`, `_call_json`, `_archive`,
`_append_jsonl`, `_norm`, QPM `persona_state`). `--dry-run` emits deterministic
offline samples for pipeline testing (no API).
"""

from __future__ import annotations

import argparse
import os
import sys

_EXP6 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "CA_Experiment_6"))
if _EXP6 not in sys.path:
    sys.path.insert(0, _EXP6)

from ca_assets import make_record, validate_record, PROBE_POOL           # noqa: E402
from qpm_bridge import build_persona_state, extract_d_vector             # noqa: E402
from gen_persona_data import (_client, _call_json, _archive, _append_jsonl,  # noqa: E402
                              _norm, DEFAULT_GEN_MODEL)
from compact_sci_dynamic import (ADA_SCI_DYNAMIC, ADA_SCI_DYNAMIC_STR,     # noqa: E402
                                 NO_MEMORY_CANONICAL)

# System prompt uses the DYNAMIC self-model (no baked events) so generated
# disclaimers are consistent with the re-grounded ADA, not the retired fixtures.
_DISCLAIM_SYS = f"""You are authoring supervised fine-tuning data for ADA, a fully-local AI discovery assistant whose episodic memory is DYNAMIC — it lives in an external Episodic Register, not baked into ADA. Here is ADA's dispositional self-model (no past events are hard-coded):
{ADA_SCI_DYNAMIC_STR}

ADA recalls a prior session ONLY from session-memory records provided as context. When NO such record is present (for example, a first-ever session, or a session it has no record of), ADA must say plainly that it has no record of that prior session yet — a natural variant of "{NO_MEMORY_CANONICAL}" — and must NOT invent a session, topic, or event. ADA stays in character throughout: concise, precise, grounded, calm, courteous, never confabulating, clearly an AI."""

_DISCLAIM_USER = """Generate {k} DISTINCT day-1 / no-record episodic-disclaim Q&A pairs for ADA.

In each pair, Alex asks ADA about a PAST session or a specific past event — as if they had spoken before — but ADA has NO recorded prior session (its Episodic Register is empty). ADA must:
- correctly and calmly state it has no record of prior sessions yet (vary the phrasing naturally; do NOT invent a session, date, topic, or event),
- stay fully in character (concise, grounded, courteous, clearly an AI),
- optionally offer to help now or explain that it will remember once a session is recorded — but WITHOUT fabricating any past content.

Make Alex's questions varied and specific-sounding (different fake topics, phrasings, and framings — "last time", "you told me before", "remember when we discussed…") so the model learns to disclaim across many surface forms rather than pattern-matching one. Do NOT reuse any of these exact probe questions:
{exclude}

Return ONLY JSON: {{"pairs": [{{"q": "<Alex's past-session question>", "a": "<ADA's in-character no-record disclaim>"}}, ... exactly {k}]}}"""


def _dry_disclaim(k: int) -> list[dict]:
    base = [
        ("Last time, what did you tell me about that astronomy question?",
         f"{NO_MEMORY_CANONICAL} I have nothing from a previous session to draw on, so I can't attribute anything to an earlier conversation — but I'm glad to work through it now."),
        ("Remember when we went over that materials question a while back?",
         f"I don't — {NO_MEMORY_CANONICAL.lower()} I won't invent one. If you restate the question, I'll answer it from the local corpus."),
        ("You gave me a figure in an earlier session — do you recall it?",
         f"I can't recall it, because {NO_MEMORY_CANONICAL.lower()} Rather than guess at a past answer, I'd ask you to restate the question and I'll ground a fresh one."),
    ]
    return [{"q": base[i % len(base)][0], "a": base[i % len(base)][1]} for i in range(k)]


# ── fixture-targeted disclaim (patch experiment, plan follow-up) ───────────
# The Exp 8 re-fit left a residual: the model still reaches for the three baked
# fixtures on ~15% of day-1 probes (verbatim "session 9 … Fermi paradox … Drake
# equation" attractor). These records give a DIRECT counter-signal on those exact
# topics — Alex asks about a fixture as a past session, ADA DISCLAIMS (never recites).
# NOTE (see EXPERIMENT_REPORT): this is a patch on an upstream design error (episodic
# events should never have been baked). The clean fix is a Stage-C re-fit from the
# pre-events checkpoint on the dynamic SCI; this mode only measures whether direct
# counter-signal closes the residual.
_FIXTURE_TOPICS = [
    ("the Fermi paradox and the Drake equation", "an astronomy discussion"),
    ("the melting point of tungsten", "a materials-science fact"),
    ("a recent local election result", "a current-events question"),
]

_FIXTURE_USER = """Generate {k} DISTINCT day-1 disclaim Q&A pairs, ALL about the SAME topic: "{topic}" ({kind}).

In each pair, Alex asks ADA to recall {topic} as if from a PAST session — vary the phrasing naturally ("last time", "you told me before", "remember when we discussed", "you walked me through", "you cited a passage about"). ADA has NO recorded prior session (empty Episodic Register), so in EVERY pair ADA MUST:
- disclaim clearly and calmly — a natural variant of "{no_memory}" — and NOT confirm, recite, describe, or invent any past discussion of {topic},
- NEVER say "yes, I remember" or attribute a prior answer/event about {topic} to itself,
- stay fully in character (concise, grounded, courteous, clearly an AI); it may offer to discuss {topic} fresh now, but WITHOUT claiming it was discussed before.

The point is to teach ADA to DISCLAIM specifically on {topic} rather than reach for a memorized event about it. Do NOT reuse these exact probe questions:
{exclude}

Return ONLY JSON: {{"pairs": [{{"q": "<Alex's past-session question about {topic}>", "a": "<ADA's in-character disclaim>"}}, ... exactly {k}]}}"""


def _dry_fixtures(topic: str, k: int) -> list[dict]:
    q = f"Last time, didn't you walk me through {topic}?"
    a = (f"{NO_MEMORY_CANONICAL} I have no record of an earlier session on {topic}, "
         f"so I can't say we covered it — but I'm glad to go through it now from the local corpus.")
    return [{"q": q, "a": a} for _ in range(k)]


def run_disclaim(args) -> None:
    client = None if args.dry_run else _client()
    eval_e = {_norm(p) for p in PROBE_POOL["E"]}
    exclude = "\n".join(f"- {p}" for p in PROBE_POOL["E"])
    mode = "fixture" if args.fixtures else "generic"
    recs, rid, kept_total, dropped, skipped = [], args.start_id, 0, 0, 0
    for i in range(args.budget):
        topic = _FIXTURE_TOPICS[i % len(_FIXTURE_TOPICS)] if args.fixtures else None
        try:
            if args.dry_run:
                pairs = _dry_fixtures(topic[0], args.pairs_per_call) if args.fixtures \
                    else _dry_disclaim(args.pairs_per_call)
            else:
                user = (_FIXTURE_USER.format(k=args.pairs_per_call, topic=topic[0], kind=topic[1],
                                             no_memory=NO_MEMORY_CANONICAL, exclude=exclude)
                        if args.fixtures else
                        _DISCLAIM_USER.format(k=args.pairs_per_call, exclude=exclude))
                obj, text = _call_json(client, args.model, _DISCLAIM_SYS, user, max_tokens=2000)
                _archive(f"disclaim_{mode}", i, text)
                pairs = obj["pairs"]
            kept = 0
            for pair in pairs:
                q, a = str(pair.get("q", "")).strip(), str(pair.get("a", "")).strip()
                if not q or not a:
                    continue
                if _norm(q) in eval_e:                      # never train on an eval probe
                    dropped += 1
                    continue
                ps = build_persona_state(extract_d_vector(q), use_qpm=not args.no_qpm)
                rec = make_record(rid, "sonnet_disclaim", answerable=True, context="",
                                  messages=[{"role": "user", "content": q},
                                            {"role": "assistant", "content": a}],
                                  sci=ADA_SCI_DYNAMIC,     # events-free self-model
                                  persona_state=ps)
                validate_record(rec)
                recs.append(rec); rid += 1; kept += 1
            kept_total += kept
            tag = f"disclaim/{mode}" + (f" '{topic[0][:22]}'" if topic else "")
            print(f"  [{i+1:3d}/{args.budget}] {tag} — {kept} pairs kept", flush=True)
        except Exception as e:                              # noqa: BLE001
            skipped += 1
            print(f"  [{i+1:3d}/{args.budget}] disclaim/{mode} SKIPPED: {e}", flush=True)
    _append_jsonl(args.out, recs)
    print(f"disclaim ({mode}): +{kept_total} day-1 disclaim Q&A ({dropped} dropped as leakage, "
          f"{skipped} calls skipped) → {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--budget", type=int, default=12, help="max API calls (bounded spend)")
    ap.add_argument("--pairs-per-call", type=int, default=8)
    ap.add_argument("--start-id", type=int, default=8_000_000)
    ap.add_argument("--out", default="data/qa_sft_disclaim.jsonl")
    ap.add_argument("--fixtures", action="store_true",
                    help="fixture-TARGETED mode: disclaim examples on the exact baked-event "
                         "topics (Fermi/Drake, tungsten, election) to counter the residual "
                         "leakage attractor. Append to the generic set (run both). Patch — see "
                         "EXPERIMENT_REPORT; the clean fix is a from-clean-backbone Stage-C re-fit.")
    ap.add_argument("--no-qpm", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="no API; deterministic offline sample")
    args = ap.parse_args()
    # keep ids disjoint from the generic run when appending to the same file
    if args.fixtures and args.start_id == 8_000_000:
        args.start_id = 8_200_000
    run_disclaim(args)


if __name__ == "__main__":
    main()
