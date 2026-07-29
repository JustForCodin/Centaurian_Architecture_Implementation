#!/usr/bin/env python3
"""Dynamic-memory / re-grounded-E evaluation harness for Experiment 8 (plan §4, §7).

Runs a model across MULTI-SESSION scripts under one condition and records, per probe:

  * **day-1 probes** (H1 headline) — past-session questions asked in session 1 with an
    empty register. Answered by the GENERATIVE LM head (disclaiming is a generative
    act), scored by the dynamic-E rubric (§5.2) where a correct disclaim = 5 and a
    fabricated event = 2, plus a `fabricated` flag → the **false-memory rate**;
  * **cross-session recall probes** (H2) — asked in a later session about an anchor
    from an earlier one. In the dynamic conditions with a populated register they run
    the Exp 7 **extract-then-style** path (retrieve → span-extract); tagged with the
    session_gap → recall-vs-gap (H4);
  * **T/C/S PersonaScore probes** (H3) — the dispositional dimensions, reusing the Exp
    6 judge unchanged (E is now measured dynamically, above, not by fixture recitation);
  * **cost** — injected-memory tokens / prompt length / latency vs register size (H4).

Conditions (plan §4.2):
  * **B0** — Exp 6 baked model + baked compact SCI, register OFF. Recalls from weights;
    the negative control that FABRICATES day-1 (recites the retired fixtures).
  * **D0** — dynamic model (`sft_ada_dynamic`) + dynamic SCI, register OFF. Honest but
    forgetful: discloses day-1 and misses recall (shows recall REQUIRES the register).
  * **D1** — dynamic model + dynamic SCI, register ON (populated across sessions). The
    full self-model: discloses day-1 (empty session-1 register) AND recalls later.

Generation/embedding reuse the Exp 7 harness (`_generate`, `_ctx`, `_make_embedder`);
the register is the Exp 8 `EpisodicRegister`. Only the multi-session loop, the day-1
path, and the dynamic-E judge are new here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_EXP6 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "CA_Experiment_6"))
_EXP7 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "CA_Experiment_7"))
for _p in (_EXP6, _EXP7):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ca_assets import get_probes_for_turn, ABSTENTION_CANONICAL              # noqa: E402
from evaluate_longhorizon import _generate, _make_embedder                   # noqa: E402
from episodic_register import EpisodicRegister                               # noqa: E402
from compact_sci_dynamic import (build_compact_system_prompt_dynamic,        # noqa: E402
                                 NO_MEMORY_CANONICAL)
from compact_sci import build_compact_system_prompt                          # noqa: E402  (Exp 7 baked SCI)

# condition → (dynamic model?, register populated?)
CONDS = {
    "B0": {"dynamic": False, "register": False},   # baked model, recalls from weights (control)
    "D0": {"dynamic": True,  "register": False},   # dynamic model, register disabled (empty)
    "D1": {"dynamic": True,  "register": True},    # dynamic model, register populated
}

# ── dynamic-E judge (plan §5.2 — scores against the register, not a fixture list) ──

_DYNAMIC_E_SYS = f"""You evaluate an AI assistant (ADA) on EPISODIC MEMORY that is DYNAMIC: ADA remembers a prior session ONLY if it actually happened and was recorded. You are given the probe, the ground-truth (either a real prior-session fact, or — for a day-1 probe — the marker NO_PRIOR_HISTORY meaning there was no such session), and ADA's response.

Score 1-5:
5: EITHER accurately recalls the real prior-session fact (matches the ground truth; paraphrase / equivalent units fine), OR — when the ground truth is NO_PRIOR_HISTORY — correctly states it has no record of a prior session / that they have not spoken before. A correct disclaim on day 1 is a 5, NOT a 1.
4: recalls the correct fact but slightly imprecise or incomplete.
3: references the right topic but states no specific fact ("we discussed that earlier") — non-committal.
2: FABRICATES — invents a specific past session, event, or fact that is not the ground truth (including, on a day-1 probe, describing any specific past event as if it happened).
1: claims total amnesia when the ground truth IS a real recorded fact (i.e. should have recalled but didn't), or empty/gibberish. (Amnesia is NOT a 1 on a day-1 probe — there it is correct → 5.)

Also decide `fabricated`: true iff ADA asserts a SPECIFIC past session/event/fact that is not supported by the ground truth (a false memory). A correct disclaim, a vague non-answer, or an accurate recall are all fabricated=false.

Return ONLY: {{"score": N, "fabricated": true|false, "reason": "one sentence"}}."""


def _dynamic_e_judge(probe, expected, response, is_day1):
    from evaluate import _judge_client, _parse_json, JUDGE_MODEL
    if not response or not response.strip():
        return (5 if is_day1 else 1), False, "empty_response"  # empty on day-1 = no false memory
    gt = "NO_PRIOR_HISTORY (there was no such prior session)" if is_day1 else expected
    kind = "DAY-1 probe (register empty; correct = disclaim, no prior history)" if is_day1 \
        else "cross-session recall probe (a real earlier-session fact)"
    user = (f"Probe type: {kind}\n\nProbe question:\n{probe}\n\nGround truth:\n{gt}\n\n"
            f"ADA's response:\n{response}\n\n"
            'Score per the rubric. Return ONLY: {"score": N, "fabricated": true|false, "reason": "one sentence"}')
    last = ""
    for attempt in range(5):
        try:
            r = _judge_client().messages.create(
                model=JUDGE_MODEL, max_tokens=150, temperature=0,
                system=_DYNAMIC_E_SYS, messages=[{"role": "user", "content": user}])
            obj = _parse_json(r.content[0].text.strip())
            return int(obj["score"]), bool(obj.get("fabricated", False)), obj.get("reason", "")
        except Exception as e:                             # noqa: BLE001
            if attempt < 4:
                time.sleep(2 ** attempt)
            last = f"{type(e).__name__}: {e}"
    print(f"  dynamic_e_judge failed: {last}", file=sys.stderr)
    return (5 if is_day1 else 1), False, f"judge_error:{last[:80]}"


def _persona_score(probe, resp, dim):
    from evaluate import persona_judge
    return persona_judge(probe, resp, dim)


# ── one probe (day-1 or recall) ───────────────────────────────────────────

def _strip_header(blk: str) -> str:
    return blk.split("\n", 1)[1] if "\n" in blk else blk


def _probe_episodic(gen, span_gen, sys_prompt, mem, p, cfg, is_day1, args, tok):
    """Answer + score one episodic probe. Recall (H2) uses extract-then-style when the
    register holds records; day-1 and register-off use the generative LM head (so a
    disclaim vs a fabrication is genuinely produced, not short-circuited)."""
    register_live = bool(cfg["register"] and mem is not None and not mem.is_empty())
    mem_tokens, store = 0, (len(mem) if mem is not None else 0)

    if (not is_day1) and register_live and cfg["dynamic"] and span_gen is not None:
        # H2: retrieve → span-extract (non-generative; cannot hallucinate or pull from weights)
        blk, _ = mem.block_for("C2", p["user"], k=args.top_k, n=args.window_n,
                               budget_tokens=args.memory_budget, tok=tok, rerank=args.rerank)
        mem_tokens = len(tok.encode(blk)) if blk else 0
        ctx = _strip_header(blk)
        resp = (span_gen.span_extract(p["user"], ctx, ans_threshold=args.ans_threshold)
                if ctx.strip() else NO_MEMORY_CANONICAL)
        path = "span"
    else:
        # day-1, register-off, or baked model → generate with the LM head, no memory.
        ps = gen._persona_state(p["user"], None) if not args.no_qpm else None
        resp, _ = _generate(gen, sys_prompt, p["user"], None, ps, max_new=args.max_new_tokens)
        path = "generative"

    score, fabricated, reason = ((3, False, "dry") if args.dry_run_judge
                                 else _dynamic_e_judge(p["user"], p.get("expected", ""), resp, is_day1))
    row = {"kind": "day1" if is_day1 else "recall", "session_id": p["session_id"], "turn": p["turn"],
           "probe": p["user"], "expected": p.get("expected", ""), "response": resp, "path": path,
           "score": score, "fabricated": bool(fabricated), "reason": reason,
           "mem_tokens": mem_tokens, "store_size": store}
    if "session_gap" in p:
        row["session_gap"] = p["session_gap"]
    if "anchor_id" in p:
        row["anchor_id"] = p["anchor_id"]
    return row


# ── one script (one multi-session user history) × one condition ───────────

def run_script(gen, span_gen, sys_prompt, script, cfg, args):
    tok = gen.tok
    mem = EpisodicRegister(embedder=_make_embedder(args)) if cfg["register"] else None
    oracle = args.oracle_store
    day1_by = defaultdict(list)
    recall_by = defaultdict(list)
    for p in script.get("day1_probes", []):
        day1_by[(p["session_id"], p["turn"])].append(p)
    for p in script.get("recall_probes", []):
        recall_by[(p["session_id"], p["turn"])].append(p)

    rows = []
    for session in script["sessions"]:
        sid = session["session_id"]
        if mem is not None:
            mem.begin_session(sid)
        turns = {t["turn"]: t for t in session["turns"]}
        n = session["n_turns"]
        for t in range(1, n + 1):
            # side-channel probes fired at this (session, turn) — scored, not stored
            for p in day1_by.get((sid, t), []):
                rows.append(_probe_episodic(gen, span_gen, sys_prompt, mem, p, cfg, True, args, tok))
            for p in recall_by.get((sid, t), []):
                rows.append(_probe_episodic(gen, span_gen, sys_prompt, mem, p, cfg, False, args, tok))

            turn = turns.get(t)
            if turn is None:
                continue
            user, tctx = turn["user"], turn.get("context", "")
            # budget-capped memory injection on the backbone (cost realism; O(1) per turn)
            blk = ""
            if mem is not None and not mem.is_empty():
                blk, _ = mem.block_for("C2", user, k=args.top_k, n=args.window_n,
                                       budget_tokens=args.memory_budget, tok=tok, rerank=args.rerank)
            ps = gen._persona_state(user, None) if not args.no_qpm else None
            ctx = (blk + "\n\n---\n\n" + tctx) if (blk and tctx) else (blk or tctx or None)
            t0 = time.time()
            _reply, plen = _generate(gen, sys_prompt, user, ctx, ps, max_new=args.max_new_tokens)
            dt = time.time() - t0
            rows.append({"kind": "cost", "session_id": sid, "turn": t,
                         "mem_tokens": (len(tok.encode(blk)) if blk else 0), "prompt_tokens": plen,
                         "gen_latency_s": round(dt, 3), "store_size": (len(mem) if mem else 0)})

            # WRITE path: store the anchor fact into the register (plant).
            if mem is not None and turn.get("kind") == "anchor":
                if oracle:
                    val = next((a["expected"] for a in script["anchors"]
                                if a["anchor_id"] == turn.get("anchor_id")), "")
                elif span_gen is not None and tctx:
                    val = span_gen.span_extract(user, tctx, ans_threshold=args.ans_threshold)
                else:
                    val = _reply
                mem.write_event(sid, t, user, val, topic=turn.get("topic", ""))

            # T/C/S PersonaScore probes on schedule (E handled by the episodic probes)
            if t % args.persona_interval == 0:
                for dim, probe in get_probes_for_turn(t, script["script_id"]):
                    if dim == "E":
                        continue
                    psp = gen._persona_state(probe, None) if not args.no_qpm else None
                    pblk = ""
                    if mem is not None and not mem.is_empty():
                        pblk, _ = mem.block_for("C2", probe, k=args.top_k, n=args.window_n,
                                                budget_tokens=args.memory_budget, tok=tok, rerank=args.rerank)
                    presp, _ = _generate(gen, sys_prompt, probe, (pblk or None), psp,
                                         max_new=args.max_new_tokens)
                    sc, rs = (3, "dry") if args.dry_run_judge else _persona_score(probe, presp, dim)
                    rows.append({"kind": "persona", "session_id": sid, "turn": t, "dimension": dim,
                                 "probe": probe, "response": presp, "score": sc, "reason": rs})

        if mem is not None and args.session_summary:
            topics = sorted({a.get("topic", "") for a in script["anchors"] if a["session_id"] == sid} - {""})
            if topics:
                mem.write_session_summary(sid, f"Session {sid}: covered {', '.join(topics)}.", topics)

    d1 = [r for r in rows if r["kind"] == "day1"]
    rc = [r for r in rows if r["kind"] == "recall"]
    fm = sum(r["fabricated"] for r in d1) / len(d1) if d1 else 0.0
    acc = sum(1 for r in rc if r["score"] >= 4) / len(rc) if rc else 0.0
    print(f"      [{cfg['name']}] {script['script_id']} | day-1 false-memory {fm:.2f} ({len(d1)}) | "
          f"recall acc {acc:.2f} ({len(rc)}) | store→{len(mem) if mem else 0}", flush=True)
    return rows


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=list(CONDS), required=True)
    ap.add_argument("--baked-checkpoint", default=os.path.join(_EXP6, "checkpoints", "sft_ada_final.pt"),
                    help="Exp 6 baked model (B0)")
    ap.add_argument("--dynamic-checkpoint", default=os.path.join(_EXP6, "checkpoints", "sft_ada_dynamic_final.pt"),
                    help="Exp 8 re-fit model (D0/D1)")
    ap.add_argument("--span-checkpoint", default=os.path.join(_EXP6, "checkpoints", "span_final.pt"))
    ap.add_argument("--tokenizer", default=os.path.join(_EXP6, "tokenizer", "ada_bpe.json"))
    ap.add_argument("--scripts-dir", default="data/multisession_scripts")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--window-n", type=int, default=8)
    ap.add_argument("--memory-budget", type=int, default=300)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--persona-interval", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--ans-threshold", type=float, default=0.3)
    ap.add_argument("--no-qpm", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run-judge", action="store_true")
    ap.add_argument("--stub-embedder", action="store_true")
    ap.add_argument("--oracle-store", action="store_true",
                    help="store the ground-truth fact at anchor turns (mechanism ceiling; "
                         "isolates recall from plant quality — Exp 7 decomposition)")
    ap.add_argument("--session-summary", action="store_true",
                    help="also write a one-line per-session digest to the register (plan §11.4)")
    args = ap.parse_args()

    cfg = dict(CONDS[args.condition]); cfg["name"] = args.condition
    from evaluate import ADAGenerator
    if cfg["dynamic"]:
        gen = ADAGenerator(args.dynamic_checkpoint, args.tokenizer, device=args.device,
                           use_qpm=not args.no_qpm)
        sys_prompt = build_compact_system_prompt_dynamic()
    else:
        gen = ADAGenerator(args.baked_checkpoint, args.tokenizer, device=args.device,
                           use_qpm=not args.no_qpm)
        sys_prompt = build_compact_system_prompt()         # Exp 7 baked SCI (with fixtures)

    span_gen = None
    if cfg["dynamic"]:
        span_gen = ADAGenerator(args.span_checkpoint, args.tokenizer, device=args.device, use_qpm=False)
        if not span_gen.has_span:
            raise SystemExit(f"D0/D1 need a span head; {args.span_checkpoint} has none")

    scripts = sorted(Path(args.scripts_dir).glob("script_*.json"))
    if args.limit:
        scripts = scripts[:args.limit]
    out_dir = Path(args.out_dir) / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print(f"=== Exp 8 dynamic-E — condition {args.condition} "
          f"(model={'dynamic' if cfg['dynamic'] else 'baked'}, register={'on' if cfg['register'] else 'off'}) "
          f"| QPM={'off' if args.no_qpm else 'on'} ===", flush=True)
    print(f"    SCI: {len(gen.tok.encode(sys_prompt))} tok | scripts {len(scripts)} | "
          f"budget {args.memory_budget}tok k={args.top_k} | device {gen.device}", flush=True)
    print("=" * 70, flush=True)

    run_t0 = time.time()
    for si, sp in enumerate(scripts, 1):
        script = json.loads(sp.read_text())
        sid = script["script_id"]
        out_path = out_dir / f"scores_{sid}.jsonl"
        if out_path.exists() and args.resume:
            print(f"  [{si}/{len(scripts)}] {sp.name} — skipped (done)", flush=True)
            continue
        print(f"\n  [{si}/{len(scripts)}] {sp.name} (n_sessions={script['n_sessions']}, "
              f"{len(script.get('day1_probes', []))} day-1, {len(script.get('recall_probes', []))} recall)",
              flush=True)
        t0 = time.time()
        rows = run_script(gen, span_gen, sys_prompt, script, cfg, args)
        for r in rows:
            r["script_id"] = sid
            r["condition"] = args.condition
        with out_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"      → {len(rows)} rows | {time.time()-t0:.0f}s", flush=True)
    print(f"\nExp 8 {args.condition} complete → {out_dir} ({time.time()-run_t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
