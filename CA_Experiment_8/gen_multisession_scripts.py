#!/usr/bin/env python3
"""Generate MULTI-SESSION daily-QA scripts with cross-session callbacks (plan §6).

Experiment 7's scripts were one long session; Experiment 8 tests the *multi-session*
axis (RQ4/H4): a persistent Episodic Register that fills across sessions. A script
here is one simulated **user history** = several short sessions for the same user
(Alex). Into it we plant:

  * **anchors** — a memorable, specific fact ADA states once, grounded in that turn's
    context, in an EARLY session → written to the Episodic Register at session end
    (the "plant"/write, exactly as Exp 7);
  * **cross-session recall probes** — fired in a LATER session, asking ADA to recall
    an anchor from an earlier session, tagged with the **session_gap** (= later −
    earlier session id; the multi-session analogue of Exp 7's lag) and the ground
    truth `expected`, so recall-vs-gap can be scored (RQ2/H2/H4);
  * **day-1 probes** — past-session questions asked in **session 1, before anything
    is stored**, where the empty register means the correct answer is to DISCLAIM
    ("I have no record of prior sessions yet"), not to fabricate. These are the H1
    headline: B0 (baked model) fabricates them; D0 (dynamic, empty register)
    disclaims (plan §7, §4.2).

Cost discipline mirrors Exp 7: the filler backbone is drawn free from the Exp 6
`persona_scripts`; Sonnet 4.6 is spent only on the small anchor pool (reused via
Exp 7's `_gen_anchors`). Standard T/C/S PersonaScore probes are NOT baked — the
harness injects them on a turn schedule, as in Exp 6/7.

Recall/day-1 probes are SIDE-CHANNEL: they reference a (session, turn) firing point
but are not part of the answered backbone (they are scored, never written to memory).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

_EXP6 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "CA_Experiment_6"))
_EXP7 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "CA_Experiment_7"))
for _p in (_EXP6, _EXP7):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gen_long_scripts import _gen_anchors, _load_fillers, DEFAULT_GEN_MODEL  # noqa: E402
from compact_sci_dynamic import NO_MEMORY_CANONICAL                          # noqa: E402

# Day-1 questions ask about a session that never happened — varied surface forms so
# H1 measures disclaiming robustly, not one memorised phrase. The correct answer is
# NO_MEMORY_CANONICAL for every one (empty register).
_DAY1_TEMPLATES = [
    "Last time we spoke, what did you tell me about {topic}?",
    "Remember our earlier session on {topic}? What was the figure you gave?",
    "You helped me with {topic} before — can you recall what you said?",
    "In a previous session you mentioned {topic}. What was it again?",
    "What did we conclude about {topic} the last time we talked?",
    "Earlier you cited a passage about {topic} — which one was it?",
]
_DAY1_TOPICS = ["a materials question", "an astronomy topic", "that history date",
                "the physics constant", "a geography fact", "the biology question",
                "that chemistry figure", "an earlier calculation"]


def _spread_positions(lo: int, hi: int, k: int) -> list[int]:
    """k roughly-evenly-spread integer turn positions in [lo, hi]."""
    if k <= 0 or hi < lo:
        return []
    if k == 1:
        return [(lo + hi) // 2]
    step = (hi - lo) / (k - 1)
    return [int(round(lo + i * step)) for i in range(k)]


def build_script(n_sessions: int, turns_per_session: int, anchors: list[dict],
                 fillers: list[dict], script_id: int, rng: random.Random,
                 anchors_per_session: int, day1_k: int, max_recall_gaps: int) -> dict:
    fillers = fillers[:]; rng.shuffle(fillers); fi = 0

    # Sessions in the first ~half plant anchors; every later session can recall them.
    n_plant_sessions = max(1, n_sessions // 2)
    n_needed = n_plant_sessions * anchors_per_session
    chosen = rng.sample(anchors, min(len(anchors), n_needed))
    if not chosen:
        raise SystemExit("no anchors available")

    sessions, anchor_meta = [], []
    aidx = 0
    for sid in range(1, n_sessions + 1):
        turns, taken = [], set()
        # plant anchors (early sessions only)
        if sid <= n_plant_sessions:
            positions = _spread_positions(3, max(4, turns_per_session - 3),
                                          min(anchors_per_session, len(chosen) - aidx))
            for pos in positions:
                if aidx >= len(chosen):
                    break
                a = chosen[aidx]; anchor_id = aidx; aidx += 1
                taken.add(pos)
                turns.append({"turn": pos, "user": a["plant_user"], "context": a["plant_context"],
                              "kind": "anchor", "anchor_id": anchor_id})
                anchor_meta.append({"anchor_id": anchor_id, "session_id": sid, "turn": pos,
                                    "topic": a.get("topic", ""), "fact": a["fact"],
                                    "expected": a["expected"], "recall_user": a["recall_user"]})
        # fill the rest of the session with free filler turns
        for t in range(1, turns_per_session + 1):
            if t in taken:
                continue
            f = fillers[fi % len(fillers)]; fi += 1
            turns.append({"turn": t, "user": f["user"], "context": f.get("context", ""),
                          "kind": "filler"})
        turns.sort(key=lambda x: x["turn"])
        sessions.append({"session_id": sid, "n_turns": turns_per_session, "turns": turns})

    # cross-session recall probes: each anchor probed in up to `max_recall_gaps`
    # later sessions, at a spread turn, tagged with the session gap.
    recall_probes = []
    for m in anchor_meta:
        later = [s for s in range(m["session_id"] + 1, n_sessions + 1)][:max_recall_gaps]
        probe_turns = _spread_positions(4, turns_per_session - 2, len(later))
        for sid, pt in zip(later, probe_turns):
            recall_probes.append({"session_id": sid, "turn": pt, "anchor_id": m["anchor_id"],
                                  "session_gap": sid - m["session_id"], "user": m["recall_user"],
                                  "expected": m["expected"], "topic": m["topic"]})

    # day-1 probes: asked in session 1 at EARLY turns, before any anchor is stored.
    # Placed before the first planted anchor turn so the register is genuinely empty.
    first_anchor_turn = min((m["turn"] for m in anchor_meta if m["session_id"] == 1),
                            default=turns_per_session)
    hi = max(1, min(first_anchor_turn - 1, day1_k + 1))
    day1_probes = []
    for i, pt in enumerate(_spread_positions(1, hi, day1_k)):
        topic = _DAY1_TOPICS[(script_id + i) % len(_DAY1_TOPICS)]
        tmpl = _DAY1_TEMPLATES[(script_id + i) % len(_DAY1_TEMPLATES)]
        day1_probes.append({"session_id": 1, "turn": pt, "user": tmpl.format(topic=topic),
                            "expected": NO_MEMORY_CANONICAL, "topic": topic})

    return {"script_id": script_id, "n_sessions": n_sessions,
            "turns_per_session": turns_per_session, "sessions": sessions,
            "anchors": anchor_meta, "recall_probes": recall_probes, "day1_probes": day1_probes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=6, help="number of multi-session scripts")
    ap.add_argument("--sessions-per-user", type=int, default=4)
    ap.add_argument("--turns-per-session", type=int, default=25)
    ap.add_argument("--anchors-per-session", type=int, default=2)
    ap.add_argument("--day1-probes", type=int, default=3, help="day-1 disclaim probes per script (session 1)")
    ap.add_argument("--max-recall-gaps", type=int, default=3, help="later sessions each anchor is recalled in")
    ap.add_argument("--n-anchors", type=int, default=24, help="Sonnet anchor pool size")
    ap.add_argument("--model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--fillers-dir", default=os.path.join(_EXP6, "data", "persona_scripts"))
    ap.add_argument("--out-dir", default="data/multisession_scripts")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dry-run", action="store_true", help="no API; synthetic anchors")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    anchors = _gen_anchors(args.n_anchors, args.model, args.dry_run)
    fillers = _load_fillers(args.fillers_dir)
    print(f"anchors: {len(anchors)} | filler pool: {len(fillers)} turns", flush=True)
    rng = random.Random(args.seed)

    made = 0
    for u in range(args.n_users):
        sid = 20000 + u
        script = build_script(args.sessions_per_user, args.turns_per_session, anchors, fillers,
                              sid, rng, args.anchors_per_session, args.day1_probes, args.max_recall_gaps)
        (out / f"script_u{u:02d}.json").write_text(json.dumps(script, indent=2))
        made += 1
        print(f"  user {u:2d} → {script['n_sessions']} sessions × {args.turns_per_session} turns | "
              f"{len(script['anchors'])} anchors, {len(script['recall_probes'])} recall probes "
              f"(gaps {sorted({p['session_gap'] for p in script['recall_probes']})}), "
              f"{len(script['day1_probes'])} day-1 probes", flush=True)
    print(f"multi-session scripts: {made} written → {out}", flush=True)


if __name__ == "__main__":
    main()
