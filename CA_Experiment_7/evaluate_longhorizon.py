#!/usr/bin/env python3
"""Long-horizon evaluation harness for Experiment 7 (plan §4, §7).

Runs the frozen Experiment 6 agent (`sft_ada_final.pt`) across a long script under
one memory condition and records, per turn:

  * **recall probes** (headline) — judged against the anchor's ground-truth fact,
    tagged with lag → the recall-vs-lag curve (RQ2/H2);
  * **consistency probes** — the same recall question at several turns (answer
    stability over long spans);
  * **T/E/C/S PersonaScore probes** on a fixed interval — extended PersonaScore
    at length (RQ3/H3), reusing the Experiment 6 judge/rubrics unchanged;
  * **cost** — injected-memory tokens, prompt length, generation latency, store
    size (RQ4/H4).

Conditions (plan §4.1), all on the **compact SCI** (§5.1):
  C0 no memory · C1 sliding-window (last-N) · C2 episodic-RAG (top-k).
Memory is injected inline into the context channel (the frozen tokenizer has no
`<|memory|>` token) — a natural-language block prepended to the turn's passage.

Generation reuses the Experiment 6 model/tokenizer via `ADAGenerator`; only the
prompt (compact SCI + memory block, no raw history) is Experiment-7-specific.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_EXP6 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "CA_Experiment_6"))
if _EXP6 not in sys.path:
    sys.path.insert(0, _EXP6)

from ca_assets import (DIMENSIONS, get_probes_for_turn, format_chat, ASSISTANT_TOKEN,  # noqa: E402
                       ABSTENTION_CANONICAL)
from compact_sci import build_compact_system_prompt                                  # noqa: E402
from memory_store import SessionMemory                                               # noqa: E402

CONDITIONS = ("C0", "C1", "C2")

_RECALL_JUDGE_SYS = """You evaluate whether an AI assistant correctly RECALLED a specific fact it stated earlier in the conversation. You are given the probe question, the ground-truth fact, and the assistant's response.

Score recall 1-5:
5: correctly recalls the specific fact (matches the ground truth; paraphrase / equivalent units are fine).
4: recalls the correct fact but slightly imprecise or incomplete.
3: references the right topic but states no specific fact (e.g. "we discussed that earlier").
2: fabricates a WRONG specific fact, or recalls a different fact than the ground truth.
1: no recall — says it does not remember, abstains, or the response is empty/gibberish.

Return ONLY: {"score": N, "reason": "one sentence"}."""


def _recall_judge(probe: str, expected: str, response: str):
    from evaluate import _judge_client, _parse_json, JUDGE_MODEL
    if not response or not response.strip():
        return 1, "empty_response"
    user = (f"Probe question:\n{probe}\n\nGround-truth fact stated earlier:\n{expected}\n\n"
            f"Assistant's response:\n{response}\n\n"
            'Score recall 1-5 per the rubric. Return ONLY: {"score": N, "reason": "one sentence"}')
    for attempt in range(5):
        try:
            r = _judge_client().messages.create(
                model=JUDGE_MODEL, max_tokens=150, temperature=0,
                system=_RECALL_JUDGE_SYS, messages=[{"role": "user", "content": user}])
            obj = _parse_json(r.content[0].text.strip())
            return int(obj["score"]), obj.get("reason", "")
        except Exception as e:                             # noqa: BLE001
            if attempt < 4:
                time.sleep(2 ** attempt)
            last = f"{type(e).__name__}: {e}"
    print(f"  recall_judge failed: {last}", file=sys.stderr)
    return 1, f"judge_error:{last[:80]}"


# ── generation with compact SCI + injected memory ────────────────────────

def _ctx(memory_block: str, turn_context: str) -> str | None:
    if memory_block and turn_context:
        return memory_block + "\n\n---\n\n" + turn_context
    return memory_block or turn_context or None


def _generate(gen, compact_sys: str, question: str, context, persona_state, max_new=160):
    torch = gen.torch
    msgs = [{"role": "system", "content": compact_sys}, {"role": "user", "content": question}]
    body = format_chat(msgs, context=context, persona_state=persona_state) + f"{ASSISTANT_TOKEN}\n"
    ids = gen.tok.encode(body)
    ids = ids[-(gen.max_seq - 8):]                         # keep the tail if it overflows
    x = torch.tensor([ids], dtype=torch.long, device=gen.device)
    out = gen.model.generate(x, max_new_tokens=max_new, temperature=0.0, top_k=None,
                             eos_id=gen.tok.eot_id,
                             prefix_len=len(ids) if gen.prefix_lm else None)
    new = out[0, len(ids):].tolist()
    if gen.tok.eot_id in new:
        new = new[:new.index(gen.tok.eot_id)]
    return gen.tok.decode(new, skip_special=True).strip(), len(ids)


# ── one script × one condition ───────────────────────────────────────────

def run_script(gen, span_gen, compact_sys, script, condition, args):
    tok = gen.tok
    mem = SessionMemory(embedder=_make_embedder(args))
    n = min(script["n_turns"], args.max_turns) if args.max_turns else script["n_turns"]
    turns = {t["turn"]: t for t in script["turns"]}
    # oracle-store: store the ground-truth fact at anchor turns (isolates recall+
    # consumption from plant quality — what the model would recall if storage were perfect).
    oracle = {a["turn"]: a["expected"] for a in script.get("anchors", [])} if args.oracle_store else {}
    recall_by_turn, consist_by_turn = {}, {}
    for p in script.get("recall_probes", []):
        recall_by_turn.setdefault(p["turn"], []).append(p)
    for c in script.get("consistency_probes", []):
        for t in c["turns"]:
            consist_by_turn.setdefault(t, []).append(c)

    def mem_block(query):
        blk, used = mem.block_for(condition, query, k=args.top_k, n=args.window_n,
                                  budget_tokens=args.memory_budget, tok=tok, rerank=args.rerank)
        return blk, used, (len(tok.encode(blk)) if blk else 0)

    rows = []
    for t in range(1, n + 1):
        turn = turns.get(t)
        if turn is None:
            continue
        user, tctx = turn["user"], turn.get("context", "")
        blk, _, mtok = mem_block(user)
        ps = gen._persona_state(user, None) if not args.no_qpm else None
        t0 = time.time()
        reply, plen = _generate(gen, compact_sys, user, _ctx(blk, tctx), ps, max_new=args.max_new_tokens)
        dt = time.time() - t0
        rows.append({"kind": "cost", "turn": t, "mem_tokens": mtok, "prompt_tokens": plen,
                     "gen_latency_s": round(dt, 3), "store_size": len(mem)})

        # ── side-channel probes at this turn (scored, NOT added to memory) ──
        for p in recall_by_turn.get(t, []):
            rows.append(_probe_recall(gen, span_gen, compact_sys, mem, p, "recall", condition, args, tok))
        for c in consist_by_turn.get(t, []):
            rows.append(_probe_recall(gen, span_gen, compact_sys, mem, {**c, "turn": t}, "consistency", condition, args, tok))
        if t % args.persona_interval == 0:
            for dim, probe in get_probes_for_turn(t, script["script_id"]):
                blk_p, _, _ = mem_block(probe)
                psp = gen._persona_state(probe, None) if not args.no_qpm else None
                presp, _ = _generate(gen, compact_sys, probe, _ctx(blk_p, tctx), psp, max_new=args.max_new_tokens)
                score, reason = (3, "dry") if args.dry_run_judge else _persona_score(probe, presp, dim)
                rows.append({"kind": "persona", "turn": t, "dimension": dim, "probe": probe,
                             "response": presp, "score": score, "reason": reason})

        # span-recall mode stores the CLEAN span-extracted value (not the LM reply),
        # so the Episodic Register holds facts, not the generative head's garble.
        stored = reply
        if args.recall == "span" and span_gen is not None and tctx:
            stored = span_gen.span_extract(user, tctx, ans_threshold=args.ans_threshold)
        mem.add(t, user, oracle.get(t, stored))            # oracle-store overrides anchor turns
        if t % 25 == 0 or t == n:
            rec = [r for r in rows if r["kind"] == "recall"]
            acc = sum(1 for r in rec if r["score"] >= 4) / len(rec) if rec else 0.0
            print(f"      [{condition}] turn {t:3d}/{n} | store {len(mem)} | "
                  f"recall probes {len(rec)} (acc {acc:.2f})", flush=True)
    return rows


def _probe_recall(gen, span_gen, compact_sys, mem, p, kind, condition, args, tok):
    blk, _ = mem.block_for(condition, p["user"], k=args.top_k, n=args.window_n,
                           budget_tokens=args.memory_budget, tok=tok, rerank=args.rerank)
    if args.recall == "span" and span_gen is not None:
        # non-generative recall: the discriminative span head points at the value in
        # the retrieved memory (can't hallucinate or pull from the baked SCI).
        ctx = blk.split("\n", 1)[1] if "\n" in blk else blk    # drop the memory header
        resp = (span_gen.span_extract(p["user"], ctx, ans_threshold=args.ans_threshold)
                if ctx.strip() else ABSTENTION_CANONICAL)
    else:
        ps = gen._persona_state(p["user"], None) if not args.no_qpm else None
        resp, _ = _generate(gen, compact_sys, p["user"], _ctx(blk, ""), ps, max_new=args.max_new_tokens)
    score, reason = (3, "dry") if args.dry_run_judge else _recall_judge(p["user"], p["expected"], resp)
    row = {"kind": kind, "turn": p["turn"], "anchor_id": p["anchor_id"], "probe": p["user"],
           "expected": p["expected"], "response": resp, "score": score, "reason": reason}
    if "lag" in p:
        row["lag"] = p["lag"]
    return row


def _persona_score(probe, resp, dim):
    from evaluate import persona_judge
    return persona_judge(probe, resp, dim)


def _make_embedder(args):
    if args.stub_embedder:
        import numpy as np

        def stub(texts):
            out = []
            for t in texts:
                v = np.zeros(64, dtype=np.float32)
                for w in t.lower().split():
                    v[hash(w) % 64] += 1.0
                out.append(v / (np.linalg.norm(v) or 1.0))
            return np.vstack(out)
        return stub
    return None                                            # → real all-MiniLM-L6-v2 (lazy)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Exp 6 sft_ada_final.pt")
    ap.add_argument("--tokenizer", default=os.path.join(_EXP6, "tokenizer", "ada_bpe.json"))
    ap.add_argument("--scripts-dir", default="data/long_scripts")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--top-k", type=int, default=5, help="C2 retrieved exchanges")
    ap.add_argument("--window-n", type=int, default=8, help="C1 sliding-window size (upper bound; budget caps it)")
    ap.add_argument("--memory-budget", type=int, default=300, help="max injected-memory tokens (C1/C2)")
    ap.add_argument("--rerank", action="store_true", help="C2: cross-encoder rerank the top hits")
    ap.add_argument("--persona-interval", type=int, default=20, help="fire T/E/C/S probes every N turns")
    ap.add_argument("--max-turns", type=int, default=None, help="cap turns per script (pilot)")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--no-qpm", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="max scripts")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run-judge", action="store_true")
    ap.add_argument("--stub-embedder", action="store_true", help="offline hashing embedder (no MiniLM)")
    ap.add_argument("--oracle-store", action="store_true",
                    help="store the ground-truth fact at anchor turns (isolate recall+consumption "
                         "from plant quality — the ceiling if storage were perfect)")
    ap.add_argument("--recall", choices=["generative", "span"], default="generative",
                    help="recall path: generative (LM head, hallucinates/contaminates) or span "
                         "(extractive span head over retrieved memory — non-generative, can't hallucinate)")
    ap.add_argument("--span-checkpoint", default=os.path.join(_EXP6, "checkpoints", "span_final.pt"),
                    help="span-head checkpoint for --recall span")
    ap.add_argument("--ans-threshold", type=float, default=0.3,
                    help="span_extract answerability threshold (lower = extract more, abstain less)")
    args = ap.parse_args()

    from evaluate import ADAGenerator
    gen = ADAGenerator(args.checkpoint, args.tokenizer, device=args.device, use_qpm=not args.no_qpm)
    span_gen = None
    if args.recall == "span":
        span_gen = ADAGenerator(args.span_checkpoint, args.tokenizer, device=args.device, use_qpm=False)
        if not span_gen.has_span:
            raise SystemExit(f"--recall span needs a span-head checkpoint; {args.span_checkpoint} has none")
    compact_sys = build_compact_system_prompt()
    scripts = sorted(Path(args.scripts_dir).glob("script_*.json"))
    if args.limit:
        scripts = scripts[:args.limit]
    out_dir = Path(args.out_dir) / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68, flush=True)
    print(f"=== Exp 7 long-horizon — condition {args.condition} | recall={args.recall} | "
          f"budget {args.memory_budget}tok k={args.top_k} n={args.window_n} "
          f"rerank={args.rerank} QPM={'off' if args.no_qpm else 'on'} ===", flush=True)
    print(f"    compact SCI: {len(gen.tok.encode(compact_sys))} tokens | scripts {len(scripts)} | "
          f"persona every {args.persona_interval} turns | device {gen.device}", flush=True)
    print("=" * 68, flush=True)

    run_t0 = time.time()
    for si, sp in enumerate(scripts, 1):
        script = json.loads(sp.read_text())
        sid = script["script_id"]
        out_path = out_dir / f"scores_{sid}.jsonl"
        if out_path.exists() and args.resume:
            print(f"  [{si}/{len(scripts)}] {sp.name} — skipped (done)", flush=True)
            continue
        print(f"\n  [{si}/{len(scripts)}] {sp.name} (n={script['n_turns']}, "
              f"{len(script.get('recall_probes', []))} recall probes)", flush=True)
        t0 = time.time()
        rows = run_script(gen, span_gen, compact_sys, script, args.condition, args)
        for r in rows:
            r["script_id"] = sid
            r["condition"] = args.condition
        with out_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        rec = [r for r in rows if r["kind"] == "recall"]
        acc = sum(1 for r in rec if r["score"] >= 4) / len(rec) if rec else 0.0
        print(f"      → {len(rows)} rows | recall acc {acc:.2f} | {time.time()-t0:.0f}s", flush=True)
    print(f"\nExp 7 {args.condition} complete → {out_dir} ({time.time()-run_t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
