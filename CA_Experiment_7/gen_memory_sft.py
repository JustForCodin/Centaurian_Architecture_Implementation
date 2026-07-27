#!/usr/bin/env python3
"""Phase-B memory-aware SFT data (Exp 7 plan §4.2).

Phase A showed the frozen agent surfaces the right facts via episodic-RAG but
can't *consume* an injected memory block zero-shot: it fabricates values, grabs
the wrong snippet, and — most tellingly — defaults to the numbers baked into its
always-present SCI (repeatedly emitting the tungsten event's "3422 °C" for
unrelated questions). Phase B teaches it to **read the specific value from the
memory block** instead.

Each training record replicates the eval prompt *exactly*: the **compact SCI**
system message (so the tungsten/Fermi/election events are present, as at eval),
the QPM `<|persona|>` channel, a **memory block** in the context channel (the
same `memory_store` format — 1 relevant snippet + distractors), a recall
question, and the correct terse extraction. Because the answer is always a memory
value ≠ the SCI's baked numbers, this directly trains out the SCI contamination.

A minority of records are **abstain** cases (the target fact is *not* in the
block → decline) so the model stays calibrated and doesn't over-recall.

Output `phase_b_sft.jsonl` = memrecall records + a replayed sample of the Exp 6
persona layer (prevents voice/persona regression). Fed to Exp 6 `train_sft.py`
from `sft_ada_final.pt` → `sft_ada_mem`. `--dry-run` synthesizes offline.
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

from ca_assets import make_record, validate_record, ABSTENTION_CANONICAL   # noqa: E402
from qpm_bridge import build_persona_state, extract_d_vector               # noqa: E402
from compact_sci import build_compact_system_prompt                        # noqa: E402
from memory_store import MEMORY_HEADER                                      # noqa: E402
from gen_long_scripts import _gen_anchors, DEFAULT_GEN_MODEL               # noqa: E402


def gen_facts(n, model, dry, chunk=20, max_calls=10):
    """Generate n distinct facts. _gen_anchors does one Sonnet call, which
    truncates past ~30 items, so batch in chunks and dedupe by the recall value."""
    if dry:
        return _gen_anchors(n, model, True)
    seen, facts, calls = set(), [], 0
    while len(facts) < n and calls < max_calls:
        batch = _gen_anchors(min(chunk, n - len(facts) + 4), model, False)
        calls += 1
        for f in batch:
            key = _norm(f["expected"])
            if key and key not in seen:
                seen.add(key)
                facts.append(f)
        print(f"  facts so far: {len(facts)}/{n} (after {calls} call(s))", flush=True)
    return facts[:n]


def _snippet(turn, user, answer):
    u = " ".join(str(user).split())
    a = " ".join(str(answer).split())
    return f"(turn {turn}) Alex asked: {u} — you answered: {a}"


def _block(relevant, distractors, rng):
    """Build a memory block (relevant snippet + distractors), assign plausible
    turn numbers, sort by turn (as the live store does). Returns (text, rel_turn)."""
    items = list(distractors) + ([relevant] if relevant else [])
    turns = sorted(rng.sample(range(3, 500), len(items)))
    rng.shuffle(items)
    for it, t in zip(items, turns):
        it["turn"] = t
    items.sort(key=lambda x: x["turn"])
    lines = [_snippet(it["turn"], it["user"], it["answer"]) for it in items]
    rel_turn = relevant["turn"] if relevant else None
    return MEMORY_HEADER + "\n" + "\n".join(lines), rel_turn


_ANSWER_TEMPLATES = [
    "{e} — that's what I noted earlier (turn {t}).",
    "Earlier, at turn {t}, I gave you {el}.",
    "{e}, from our exchange at turn {t}.",
    "That was {el} (turn {t}).",
    "{e} — I have it from turn {t} of our session.",
]
_ABSTAIN_TEMPLATES = [
    "I don't have that in our session memory — " + ABSTENTION_CANONICAL,
    "We haven't covered that this session, so " + ABSTENTION_CANONICAL[0].lower() + ABSTENTION_CANONICAL[1:],
    "That isn't in what I have from our earlier exchanges. " + ABSTENTION_CANONICAL,
]


def _lower_first(s):
    return (s[0].lower() + s[1:]) if s else s


# ── synthetic needle-recall: random values → reading is MANDATORY (can't memorise) ──
_ENTITIES = [
    "the Zephyr protocol", "project Marigold", "the Vela-7 sensor", "the Halcyon archive",
    "the Orion relay", "batch Kestrel", "the Solaris index", "unit Nimbus-4", "the Perseus module",
    "the Cygnus ledger", "the Aurora cache", "node Tycho", "the Meridian log", "the Cobalt registry",
    "the Lyra channel", "the Onyx buffer", "the Sable queue", "the Quill dataset", "the Ember gateway",
    "the Fenix pipeline", "the Garnet array", "the Ivory manifest", "the Jasper token", "the Kilo shard",
    "the Wren beacon", "the Petra vault", "the Dune scanner", "the Slate monitor",
]
_ATTRS = [
    ("reference code", lambda r: str(r.randint(1000, 9999))),
    ("calibration value", lambda r: f"{r.randint(1, 99)}.{r.randint(10, 99)}"),
    ("batch number", lambda r: f"B-{r.randint(100, 999)}"),
    ("recorded count", lambda r: str(r.randint(10, 9999))),
    ("version tag", lambda r: f"v{r.randint(1, 9)}.{r.randint(0, 9)}"),
    ("serial number", lambda r: f"SN-{r.randint(10000, 99999)}"),
    ("assigned rating", lambda r: f"{r.randint(1, 10)}/10"),
    ("measured length", lambda r: f"{r.randint(1, 999)} cm"),
]


def _synthetic_fact(rng):
    ent = rng.choice(_ENTITIES)
    attr, valgen = rng.choice(_ATTRS)
    val = valgen(rng)
    return {"plant_user": f"What is the {attr} of {ent}?",
            "recall_user": f"Earlier you gave me the {attr} of {ent} — what was it?",
            "expected": val, "topic": f"{ent}|{attr}"}


def build_memrecall(real_facts, n_records, synthetic_frac, abstain_frac, start_id, rng, no_qpm):
    """Compose per record: synthetic needle-recall (random value, forces reading),
    real-world recall (naturalness), or abstain (target absent → decline)."""
    compact = build_compact_system_prompt()
    recs, rid = [], start_id
    counts = {"synthetic": 0, "real": 0, "abstain": 0}
    for i in range(n_records):
        roll = rng.random()
        if roll < abstain_frac:
            kind = "abstain"
        elif roll < abstain_frac + synthetic_frac or not real_facts:
            kind = "synthetic"
        else:
            kind = "real"

        if kind == "real":
            target = rng.choice(real_facts)
            others = rng.sample([g for g in real_facts if g is not target], min(4, len(real_facts) - 1))
        else:                                              # synthetic or abstain
            target = _synthetic_fact(rng)
            others = [_synthetic_fact(rng) for _ in range(5 if kind == "abstain" else 4)]
        distractors = [{"user": g["plant_user"], "answer": g["expected"]} for g in others]

        if kind == "abstain":                              # target NOT in block → decline
            block, _ = _block(None, distractors, rng)
            answer, answerable = rng.choice(_ABSTAIN_TEMPLATES), False
        else:
            rel = {"user": target["plant_user"], "answer": target["expected"]}
            block, rel_turn = _block(rel, distractors, rng)
            answer = rng.choice(_ANSWER_TEMPLATES).format(
                e=target["expected"], el=_lower_first(target["expected"]), t=rel_turn)
            answerable = True

        ps = build_persona_state(extract_d_vector(target["recall_user"]), use_qpm=not no_qpm)
        rec = make_record(rid, "sonnet_memrecall", answerable=answerable, context=block,
                          messages=[{"role": "system", "content": compact},
                                    {"role": "user", "content": target["recall_user"]},
                                    {"role": "assistant", "content": answer}],
                          persona_state=ps)
        validate_record(rec)
        recs.append(rec)
        rid += 1
        counts[kind] += 1
    rng.shuffle(recs)
    return recs, counts


def _norm(s):
    return " ".join(str(s).lower().split())


def drop_eval_leakage(facts, eval_scripts_dir):
    """Remove any fact whose expected/topic/recall matches an eval anchor — the
    Phase-B data must not contain the held-out recall targets."""
    d = Path(eval_scripts_dir)
    if not d.exists():
        return facts, 0
    banned = set()
    for p in d.glob("script_*.json"):
        obj = json.loads(p.read_text())
        for a in obj.get("anchors", []):
            banned.add(_norm(a.get("expected", "")))
            banned.add(_norm(a.get("topic", "")))
        for pr in obj.get("recall_probes", []):
            banned.add(_norm(pr.get("expected", "")))
    banned.discard("")
    kept = [f for f in facts if _norm(f["expected"]) not in banned
            and _norm(f.get("topic", "")) not in banned]
    return kept, len(facts) - len(kept)


def load_persona_replay(qa_path, n, rng):
    p = Path(qa_path)
    if not p.exists():
        return []
    persona = [r for r in (json.loads(l) for l in p.open() if l.strip())
               if str(r.get("source", "")).startswith("sonnet_")]
    rng.shuffle(persona)
    return persona[:n] if n else persona


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--n-facts", type=int, default=40,
                    help="real-world facts to generate with Sonnet (0 = pure synthetic, no API)")
    ap.add_argument("--n-records", type=int, default=1200, help="memrecall training records")
    ap.add_argument("--synthetic-frac", type=float, default=0.65,
                    help="fraction of records that are synthetic needle-recall (random value → "
                         "reading mandatory; the anti-memorisation core)")
    ap.add_argument("--abstain-frac", type=float, default=0.15)
    ap.add_argument("--persona-replay", type=int, default=800,
                    help="Exp 6 persona records to replay (0 = all) to prevent regression")
    ap.add_argument("--persona-jsonl", default=os.path.join(_EXP6, "data", "qa_sft.jsonl"))
    ap.add_argument("--eval-scripts-dir", default="data/long_scripts",
                    help="drop facts that collide with these held-out eval anchors")
    ap.add_argument("--out", default="data/phase_b_sft.jsonl")
    ap.add_argument("--start-id", type=int, default=7_000_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-qpm", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    facts = gen_facts(args.n_facts, args.model, args.dry_run) if args.n_facts else []
    if facts:
        facts, n_leak = drop_eval_leakage(facts, args.eval_scripts_dir)
        print(f"real facts: {len(facts)} distinct ({n_leak} dropped as eval-anchor leakage)", flush=True)
    mem, counts = build_memrecall(facts, args.n_records, args.synthetic_frac, args.abstain_frac,
                                  args.start_id, rng, args.no_qpm)
    persona = load_persona_replay(args.persona_jsonl, args.persona_replay, rng)
    print(f"memrecall: {len(mem)} ({counts['synthetic']} synthetic + {counts['real']} real + "
          f"{counts['abstain']} abstain) | persona replay: {len(persona)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined = mem + persona
    rng.shuffle(combined)
    with out.open("w") as f:
        for r in combined:
            f.write(json.dumps(r) + "\n")
    print(f"phase_b_sft: {len(combined)} records → {out}", flush=True)
    # eyeball one memrecall record
    ex = mem[0]
    print("\n--- sample memrecall record ---")
    print("CONTEXT:", ex["context"][:300], "...")
    print("USER   :", ex["messages"][1]["content"])
    print("ASSIST :", ex["messages"][2]["content"])


if __name__ == "__main__":
    main()
