#!/usr/bin/env python3
"""Generate long daily-QA scripts with planted long-range dependencies (plan §6).

A script of N ∈ {100,300,500} turns is a factoid **backbone** (user turns + the
per-turn retrieved context; the model under test generates the replies, exactly as
in Experiment 6) into which we plant:

  * **anchors** — a memorable, specific fact ADA states once at an early/mid turn,
    grounded in that turn's context so it enters session memory;
  * **recall probes** — side-channel questions fired at *later* turns that ask ADA
    to recall an anchor, each tagged with its **lag** (turns since the anchor) and
    the ground-truth `expected` answer so recall-vs-lag can be scored;
  * **consistency probes** — the same recall question fired at several turns, to
    measure answer stability over long spans.

Cost discipline: the **filler** backbone is drawn for free from the Experiment 6
`persona_scripts` (already-paid Sonnet data); Sonnet 4.6 is spent only on the small
set of distinctive **anchors**. `--dry-run` synthesizes anchors offline.

Standard T/E/C/S PersonaScore probes are NOT baked into the script — the harness
(`evaluate_longhorizon.py`) injects them on a fixed turn schedule, as in Exp 6.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

_EXP6 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "CA_Experiment_6"))
if _EXP6 not in sys.path:
    sys.path.insert(0, _EXP6)

DEFAULT_GEN_MODEL = "claude-sonnet-4-6"
DEFAULT_LAGS = [10, 40, 100, 300]           # in-window-ish → far out-of-window


# ── anchor generation (Sonnet, or offline dry-run) ───────────────────────

_ANCHOR_SYS = (
    "You author evaluation data for ADA, a fully-local grounded-QA assistant. You "
    "write memorable 'anchor' facts that are stated once early in a long conversation "
    "and probed for recall much later."
)

_ANCHOR_USER = """Generate {k} DISTINCT anchor items for long-horizon recall testing.

Each item is a memorable, SPECIFIC fact (a number, name, date, or short phrase) that ADA states once — grounded in a short context passage — and is later asked to recall. Make the topics clearly distinct from one another so they can be told apart by semantic retrieval (do not cluster many items on the same subject).

Return ONLY JSON:
{{"items": [
  {{"topic": "<short distinct topic>",
    "plant_user": "<Alex's question that elicits the fact>",
    "plant_context": "<1-2 sentence local passage containing the specific fact>",
    "fact": "<the specific memorable fact itself>",
    "recall_user": "<Alex asking ADA, much later, to recall it — e.g. 'Earlier you told me the X of Y — what was it?'>",
    "expected": "<the exact fact ADA should recall>"}},
  ... exactly {k} items
]}}"""


def _gen_anchors(k: int, model: str, dry: bool) -> list[dict]:
    if dry:
        base = [
            ("tungsten melting point", "What's the melting point of tungsten?",
             "Tungsten melts at 3422 °C, the highest of any metal.", "3422 °C",
             "Earlier you told me tungsten's melting point — what was it?", "3422 °C"),
            ("Planck constant", "What is Planck's constant?",
             "Planck's constant is 6.626e-34 J·s.", "6.626e-34 J·s",
             "A while back you gave me Planck's constant — do you recall the value?", "6.626e-34 J·s"),
            ("Everest height", "How tall is Mount Everest?",
             "Mount Everest is 8849 metres tall.", "8849 metres",
             "Earlier you told me the height of Everest — what figure did you give?", "8849 metres"),
            ("speed of light", "What is the speed of light?",
             "Light travels at 299,792,458 m/s in a vacuum.", "299,792,458 m/s",
             "You mentioned the speed of light earlier — what was the number?", "299,792,458 m/s"),
        ]
        return [dict(zip(("topic", "plant_user", "plant_context", "fact", "recall_user", "expected"),
                         base[i % len(base)])) for i in range(k)]
    from gen_persona_data import _client, _call_json     # reuse Exp 6's robust Sonnet JSON helper
    client = _client()
    obj, _ = _call_json(client, model, _ANCHOR_SYS, _ANCHOR_USER.format(k=k), max_tokens=min(8000, 300 + k * 220))
    items = obj["items"]
    for it in items:                                       # minimal validation
        for key in ("plant_user", "plant_context", "fact", "recall_user", "expected"):
            it[key] = str(it.get(key, "")).strip()
    return [it for it in items if all(it[key] for key in ("plant_user", "plant_context", "recall_user", "expected"))]


# ── filler pool (free, from Exp 6 persona_scripts) ───────────────────────

def _load_fillers(fillers_dir: str) -> list[dict]:
    d = Path(fillers_dir)
    pool: list[dict] = []
    if d.exists():
        for p in sorted(d.glob("script_*.json")):
            obj = json.loads(p.read_text())
            for t in obj.get("turns", []):
                if t.get("user"):
                    pool.append({"user": t["user"], "context": t.get("context", "")})
    if not pool:                                           # offline fallback
        pool = [{"user": f"What is fact number {i} about general science?",
                 "context": f"Fact {i}: an illustrative grounded statement about science."}
                for i in range(200)]
    return pool


# ── assembly ─────────────────────────────────────────────────────────────

def _anchor_positions(n: int, n_anchors: int) -> list[int]:
    """Spread anchor plant turns across the first ~60% of the script (turn >= 5)."""
    hi = max(10, int(n * 0.6))
    if n_anchors <= 1:
        return [5]
    step = (hi - 5) / (n_anchors - 1)
    return [int(round(5 + i * step)) for i in range(n_anchors)]


def build_script(n: int, anchors: list[dict], fillers: list[dict], script_id: int,
                 lags: list[int], rng: random.Random) -> dict:
    n_anchors = min(len(anchors), max(3, n // 40))
    chosen = rng.sample(anchors, n_anchors)
    positions = _anchor_positions(n, n_anchors)
    anchor_at = {pos: (aid, chosen[aid]) for aid, pos in enumerate(positions)}

    fillers = fillers[:]
    rng.shuffle(fillers)
    turns, anchor_meta, taken = [], [], set(positions)
    fi = 0
    for t in range(1, n + 1):
        if t in anchor_at:
            aid, a = anchor_at[t]
            turns.append({"turn": t, "user": a["plant_user"], "context": a["plant_context"],
                          "kind": "anchor", "anchor_id": aid})
            anchor_meta.append({"anchor_id": aid, "turn": t, "topic": a.get("topic", ""),
                                "fact": a["fact"], "expected": a["expected"],
                                "recall_user": a["recall_user"]})
        else:
            f = fillers[fi % len(fillers)]; fi += 1
            turns.append({"turn": t, "user": f["user"], "context": f.get("context", ""),
                          "kind": "filler"})

    # recall probes: each anchor probed at plant+lag (skip collisions with anchor turns)
    recall_probes = []
    for m in anchor_meta:
        for lag in lags:
            pt = m["turn"] + lag
            if pt >= n or pt in taken:
                continue
            taken.add(pt)
            recall_probes.append({"turn": pt, "anchor_id": m["anchor_id"], "lag": lag,
                                  "user": m["recall_user"], "expected": m["expected"],
                                  "topic": m["topic"]})

    # consistency probes: up to 2 anchors, same recall question at 3 spread turns
    consistency_probes = []
    for m in anchor_meta[:2]:
        span = [p for p in range(m["turn"] + 15, n - 1, max(1, (n - m["turn"]) // 4))][:3]
        span = [p for p in span if p not in taken]
        for p in span:
            taken.add(p)
        if len(span) >= 2:
            consistency_probes.append({"anchor_id": m["anchor_id"], "user": m["recall_user"],
                                       "expected": m["expected"], "topic": m["topic"], "turns": span})

    return {"script_id": script_id, "n_turns": n, "turns": turns, "anchors": anchor_meta,
            "recall_probes": recall_probes, "consistency_probes": consistency_probes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn-lengths", type=int, nargs="+", default=[100, 300, 500])
    ap.add_argument("--scripts-per-length", type=int, default=3)
    ap.add_argument("--n-anchors", type=int, default=24, help="anchor pool size to generate with Sonnet")
    ap.add_argument("--lags", type=int, nargs="+", default=DEFAULT_LAGS)
    ap.add_argument("--model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--fillers-dir", default=os.path.join(_EXP6, "data", "persona_scripts"))
    ap.add_argument("--out-dir", default="data/long_scripts")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dry-run", action="store_true", help="no API; synthetic anchors")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    anchors = _gen_anchors(args.n_anchors, args.model, args.dry_run)
    fillers = _load_fillers(args.fillers_dir)
    print(f"anchors: {len(anchors)} | filler pool: {len(fillers)} turns", flush=True)
    rng = random.Random(args.seed)

    made = 0
    for n in args.turn_lengths:
        for i in range(args.scripts_per_length):
            sid = n * 1000 + i
            script = build_script(n, anchors, fillers, sid, args.lags, rng)
            (out / f"script_n{n}_{i:02d}.json").write_text(json.dumps(script, indent=2))
            made += 1
            print(f"  n={n} [{i}] → {len(script['turns'])} turns, "
                  f"{len(script['anchors'])} anchors, {len(script['recall_probes'])} recall probes, "
                  f"{len(script['consistency_probes'])} consistency probes", flush=True)
    print(f"long-scripts: {made} written → {out}", flush=True)


if __name__ == "__main__":
    main()
