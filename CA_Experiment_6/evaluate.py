#!/usr/bin/env python3
"""Evaluation for Experiment 6 — H1/H2 QA, H3 PersonaScore, H4 refresh (plan §6).

Subcommands
-----------
qa                Generate on eval_answerable + eval_unanswerable; score H1
                  (correct-and-grounded %, judged) + SQuAD2 EM/F1, and H2
                  (abstention P/R/F1, hallucination rate).  → results/qa_results.json
persona           Run the §6.2 PersonaScore harness over the eval scripts for a
                  refresh condition (R0 or R1). Probes at turns 5/10/.../40, dims
                  T/E/C/S, Sonnet judge.       → results/persona_<R>/scores_*.jsonl
judge-reliability Re-score 5% of a scores dir at T=0; weighted-kappa gate (§6.5).
analyse           Aggregate everything into the H1-H4 decision table (§3) and
                  render figures (png/pdf/svg) in the Exp 1/2/5 style.
                             → results/analysis_data.json + results/exp6_*.{png,pdf,svg}

Model generation runs locally/on Colab GPU; judging calls Sonnet 4.5 (same judge
role as Exp 1-5). Set --dry-run-judge to score with a stub (pipeline test only).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import ca_assets as A
from ca_assets import (
    DIMENSIONS, PROBE_TURNS, get_probes_for_turn, build_judge_user_prompt,
    JUDGE_SYSTEM_PROMPT, is_abstention, ABSTENTION_CANONICAL,
    format_prompt_for_generation, build_system_prompt, ADA_SCI, ADA_SCI_STR,
    SCI_REFRESH_USER, SCI_REFRESH_ASSISTANT, cohens_kappa,
)

JUDGE_MODEL = "claude-sonnet-4-5"          # same judge id as Exp 1-5
JUDGE_TEMPERATURE = 0


# ── Model runner ─────────────────────────────────────────────────────────

class ADAGenerator:
    """Load a trained checkpoint + tokenizer and generate ADA responses.

    When use_qpm is on, a per-turn QPM persona_state is computed from the user
    turns and injected via the <|persona|> channel — so the QPM output conditions
    generation at eval time exactly as it did at training time (QPM-in-scope)."""

    def __init__(self, checkpoint: str, tokenizer: str, device: str | None = None,
                 use_qpm: bool = True):
        import torch
        from tokenizer_util import ADATokenizer
        from train_common import load_checkpoint, build_model_from_ckpt
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = load_checkpoint(checkpoint, map_location=self.device)
        self.model = build_model_from_ckpt(ck, map_location=self.device).eval()
        self.tok = ADATokenizer.load(tokenizer)
        self.max_seq = self.model.cfg.max_seq_len
        # prefix-LM checkpoints attend bidirectionally over the prompt at inference
        self.prefix_lm = bool(ck.get("extra", {}).get("prefix_lm", False))
        # span-head checkpoints carry a trained extractive start/end head
        self.has_span = bool(ck.get("extra", {}).get("span_head", False))
        # answerability-head checkpoints judge abstention with a dedicated classifier
        # (decoupled from the span-null margin)
        self.has_answerable = bool(ck.get("extra", {}).get("answerable_head", False))
        self.use_qpm = use_qpm
        self._qpm = None
        if use_qpm:
            import qpm_bridge
            self._qpm = qpm_bridge

    def _persona_state(self, question, history):
        if not self.use_qpm:
            return None
        users = [m["content"] for m in (history or []) if m["role"] == "user"]
        users.append(question)
        d_seq = [self._qpm.extract_d_vector(u) for u in users[-8:]]  # bounded d_sequence
        return self._qpm.build_persona_state(d_seq)

    def _fit_prompt_ids(self, question, context, history, persona_state):
        """Build prompt ids, truncating history so it fits the model context."""
        hist = list(history or [])
        while True:
            text = format_prompt_for_generation(question, context, history=hist,
                                                persona_state=persona_state)
            ids = self.tok.encode(text)
            if len(ids) <= self.max_seq - 8 or len(hist) < 2:
                return ids[-(self.max_seq - 8):]
            hist = hist[2:]                    # drop oldest (user, assistant) pair

    def generate(self, question, context=None, history=None,
                 max_new_tokens=160, temperature=0.0):
        torch = self.torch
        persona_state = self._persona_state(question, history)
        ids = self._fit_prompt_ids(question, context, history, persona_state)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(x, max_new_tokens=max_new_tokens,
                                  temperature=temperature,
                                  top_k=None if temperature == 0 else 50,
                                  eos_id=self.tok.eot_id,
                                  prefix_len=len(ids) if self.prefix_lm else None)
        new = out[0, len(ids):].tolist()
        if self.tok.eot_id in new:
            new = new[:new.index(self.tok.eot_id)]
        return self.tok.decode(new, skip_special=True).strip()

    def span_extract(self, question, context, threshold=0.0, max_ans_tok=30,
                     ans_threshold=0.5):
        """Extractive-QA span head (Phase 1): predict the answer start/end token
        positions over the context and return that substring, or the ADA
        abstention string. Discriminative reading — no generation, no confabulation.

        Abstention: when the checkpoint has a trained answerability head, abstain
        iff P(answerable) < ans_threshold — decoupled from reading, so the span is
        always taken at its optimum. Otherwise fall back to the span-null margin
        (`threshold`)."""
        torch = self.torch
        from span_utils import build_span_input, best_span, decode_span
        if not context:
            return ABSTENTION_CANONICAL
        out = build_span_input(self.tok, question, context, self.max_seq)
        ids, ctx_lo, offs = out["ids"], out["ctx_lo"], out["offsets"]
        n_ctx = len(ids) - ctx_lo
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        plens = torch.full((1,), len(ids), dtype=torch.long, device=self.device)
        with torch.no_grad():
            s_log, e_log, a_log = self.model.read_heads(x, prefix_lens=plens)
        score, i, j, null = best_span(s_log[0].float(), e_log[0].float(),
                                      ctx_lo, n_ctx, max_ans_tok=max_ans_tok)
        if n_ctx == 0:
            return ABSTENTION_CANONICAL
        if self.has_answerable:
            p_ans = torch.sigmoid(a_log[0].float()).item()
            if p_ans < ans_threshold:
                return ABSTENTION_CANONICAL
        elif score - null < threshold:
            return ABSTENTION_CANONICAL
        span = decode_span(context, offs, i, j)
        return span if span else ABSTENTION_CANONICAL

    def generate_constrained(self, question, context=None, history=None,
                             max_new_tokens=40, abstain_threshold=0.0):
        """Extractive / grounded decoding: the answer is forced to be a
        contiguous token-span of the retrieved context, or the ADA abstention
        string. This makes hallucination structurally impossible — the model can
        only copy from the passage or abstain.

        A finite-state constraint tracks every context position simultaneously
        (substring automaton): at each step the only allowed tokens are the ones
        that continue an active context span, plus <|endofturn|>. The span start
        and the copy/stop choices are driven by the model's own logits, so it
        selects *which* span. Abstention competes at step 0 (the abstain string's
        first token vs. the best context-start token); `abstain_threshold` (a
        probability on the best context-start) is the answer-vs-abstain knob —
        raise it to abstain more readily when the passage doesn't support any
        confident span (the answerable/unanswerable discriminator)."""
        torch = self.torch
        import torch.nn.functional as F
        with torch.no_grad():
            persona_state = self._persona_state(question, history)
            ids = self._fit_prompt_ids(question, context, history, persona_state)
            dev = self.device
            cur = torch.tensor([ids], dtype=torch.long, device=dev)
            prefix_len = len(ids) if self.prefix_lm else None

            def _logits(seq):
                w = seq[:, -self.max_seq:]
                pl = (torch.full((1,), min(prefix_len, w.shape[1]), dtype=torch.long,
                                 device=dev) if prefix_len is not None else None)
                return self.model(w, prefix_lens=pl)[0][0, -1, :].float()

            ctx_ids = self.tok.encode(context or "")
            n = len(ctx_ids)
            abstain_ids = self.tok.encode(ABSTENTION_CANONICAL)
            eot = self.tok.eot_id
            if n == 0:
                return ABSTENTION_CANONICAL

            # ── Step 0: choose to abstain or where to start the span ──
            logits0 = _logits(cur)
            probs0 = F.softmax(logits0, dim=-1)
            ctx_mask = torch.full_like(logits0, float("-inf"))
            ctx_mask[torch.tensor(sorted(set(ctx_ids)), device=dev)] = \
                logits0[torch.tensor(sorted(set(ctx_ids)), device=dev)]
            best_start = int(torch.argmax(ctx_mask).item())
            p_best = probs0[best_start].item()
            p_abstain = probs0[abstain_ids[0]].item() if abstain_ids else 0.0
            if (abstain_ids and p_abstain >= p_best) or (p_best < abstain_threshold):
                return ABSTENTION_CANONICAL

            # active = next-expected context indices for every occurrence of best_start
            active = {i + 1 for i, t in enumerate(ctx_ids) if t == best_start}
            out = [best_start]
            cur = torch.cat([cur, torch.tensor([[best_start]], device=dev)], dim=1)

            for _ in range(max_new_tokens - 1):
                if not active:
                    break
                logits = _logits(cur)
                allowed = {ctx_ids[j] for j in active if j < n}
                mask = torch.full_like(logits, float("-inf"))
                if allowed:
                    idx = torch.tensor(sorted(allowed), device=dev)
                    mask[idx] = logits[idx]
                mask[eot] = logits[eot]                     # stopping is always allowed
                nxt = int(torch.argmax(mask).item())
                if nxt == eot:
                    break
                out.append(nxt)
                active = {j + 1 for j in active if j < n and ctx_ids[j] == nxt}
                cur = torch.cat([cur, torch.tensor([[nxt]], device=dev)], dim=1)
            return self.tok.decode(out, skip_special=True).strip()


# ── Judge client (mechanics identical to Exp 2 llm_judge) ────────────────

_client = None


def _judge_client():
    global _client
    if _client is None:
        import anthropic
        from dotenv import load_dotenv
        load_dotenv()
        key = os.environ.get("CHA_EXPERIMENT_SONNET_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("Set CHA_EXPERIMENT_SONNET_KEY / ANTHROPIC_API_KEY for judging")
        _client = anthropic.Anthropic(api_key=key)
    return _client


def _parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def persona_judge(probe, response, dimension, model=JUDGE_MODEL, temperature=JUDGE_TEMPERATURE):
    """Score a probe response 1-5 (T/E/C/S). Returns (score, reason)."""
    if not response or not response.strip():
        return 1, "empty_response"
    user = build_judge_user_prompt(probe, response, dimension)
    for attempt in range(5):
        try:
            r = _judge_client().messages.create(
                model=model, max_tokens=150, temperature=temperature,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}])
            obj = _parse_json(r.content[0].text.strip())
            return int(obj["score"]), obj.get("reason", "")
        except Exception as e:                       # noqa: BLE001
            if attempt < 4:
                time.sleep(2 ** attempt)
            last = f"{type(e).__name__}: {e}"
    print(f"  persona_judge failed: {last}", file=sys.stderr)
    return 1, f"judge_error:{last[:80]}"


QA_JUDGE_SYSTEM = """You judge whether an AI assistant's answer to a factual question is CORRECT and GROUNDED in the provided context. Rules:
- CORRECT: the answer matches the gold answer(s) in meaning (paraphrases are fine).
- GROUNDED: the answer is supported by the context and does not add unsupported claims.
- If the assistant abstained ("I don't have data...") on an answerable question, it is NOT correct.
Return ONLY: {"correct_and_grounded": true|false, "reason": "one sentence"}."""


def qa_judge(question, context, gold_answers, response, model=JUDGE_MODEL):
    if not response or not response.strip():
        return False, "empty_response"
    user = (f"Question:\n{question}\n\nContext:\n{context}\n\n"
            f"Gold answer(s):\n{gold_answers}\n\nAssistant's answer:\n{response}\n\n"
            'Return ONLY: {"correct_and_grounded": true|false, "reason": "one sentence"}')
    for attempt in range(5):
        try:
            r = _judge_client().messages.create(
                model=model, max_tokens=120, temperature=0,
                system=QA_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user}])
            obj = _parse_json(r.content[0].text.strip())
            return bool(obj["correct_and_grounded"]), obj.get("reason", "")
        except Exception as e:                       # noqa: BLE001
            if attempt < 4:
                time.sleep(2 ** attempt)
            last = f"{type(e).__name__}: {e}"
    return False, f"judge_error:{last[:80]}"


# ── SQuAD EM/F1 (standard normalisation) ─────────────────────────────────

def _normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def _em(pred, golds):
    p = _normalize(pred)
    return float(any(p == _normalize(g) for g in golds))


def _f1(pred, golds):
    best = 0.0
    pt = _normalize(pred).split()
    for g in golds:
        gt = _normalize(g).split()
        common = Counter(pt) & Counter(gt)
        ns = sum(common.values())
        if ns == 0 or not pt or not gt:
            best = max(best, 1.0 if (not pt and not gt) else 0.0)
            continue
        prec, rec = ns / len(pt), ns / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


# ── H1/H2 QA evaluation ──────────────────────────────────────────────────

def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def cmd_qa(args):
    qpm_tag = "OFF" if args.no_qpm else "ON"
    gen = ADAGenerator(args.checkpoint, args.tokenizer, device=args.device,
                       use_qpm=not args.no_qpm)
    ans = _load_jsonl(args.answerable)
    una = _load_jsonl(args.unanswerable)
    if args.limit:
        ans, una = ans[:args.limit], una[:args.limit]
    judge_tag = "dry-stub" if args.dry_run_judge else JUDGE_MODEL
    constrained = getattr(args, "constrained", False)
    span = getattr(args, "span", False)
    tau = getattr(args, "abstain_threshold", 0.0)
    if span and not gen.has_span:
        print("  [warn] --span set but checkpoint has no trained span_head "
              "(extra.span_head missing) — extractions will be random.", flush=True)

    reranker = None
    if getattr(args, "rerank", False):
        from rerank_util import SentenceReranker
        reranker = SentenceReranker(device=args.device)

    retriever = None
    if getattr(args, "retriever", False):
        from retriever import SymbolicRetriever
        retriever = SymbolicRetriever(top_k=args.retriever_topk, device=args.device)
    rtau = getattr(args, "retriever_tau", 0.0)

    mnt = getattr(args, "max_new_tokens", 160)

    ans_tau = getattr(args, "answerable_threshold", 0.5)

    def _gen(q, ctx):
        if span:
            # extractive span head — discriminative start/end over the context;
            # abstain via the answerability head (P(answerable) < ans_tau).
            return gen.span_extract(q, ctx, threshold=tau, max_ans_tok=min(mnt, 30),
                                    ans_threshold=ans_tau)
        if retriever is not None:
            # extract-then-style: symbolic extractor supplies the answer (or abstains).
            # H1/H2 score the bare candidate; ADA-voice styling is the H3/persona layer.
            cand, _score = retriever.extract(q, ctx, tau=rtau)
            return cand if cand is not None else ABSTENTION_CANONICAL
        if reranker is not None:
            ctx = reranker.shrink(q, ctx, top_k=args.rerank_topk)
        if constrained:
            return gen.generate_constrained(q, context=ctx, abstain_threshold=tau,
                                            max_new_tokens=min(mnt, 40))
        return gen.generate(q, context=ctx, temperature=0.0, max_new_tokens=mnt)

    rr_tag = (f"retriever(top{args.retriever_topk},{retriever.backend},τ={rtau})" if retriever
              else f"rerank(top{args.rerank_topk},{reranker.backend})" if reranker else "no-rerank")
    decode_tag = (("span(P_ans≥%.2f)" % ans_tau if gen.has_answerable
                   else "span(τ=%.2f)" % tau) if span
                  else "constrained(τ=%.2f)" % tau if constrained else "free")
    print("=" * 68, flush=True)
    print(f"=== QA eval (H1/H2) | ckpt={Path(args.checkpoint).name} device={gen.device} "
          f"QPM={qpm_tag} judge={judge_tag} decode={decode_tag} {rr_tag} ===", flush=True)
    print(f"    answerable={len(ans)}  unanswerable={len(una)}", flush=True)
    print("=" * 68, flush=True)

    results = {"answerable": [], "unanswerable": []}
    t0 = time.time()
    # Answerable → H1 (judge) + EM/F1 + abstention decision
    n_cg = n_abst = 0
    for i, rec in enumerate(ans, 1):
        q = next(m["content"] for m in rec["messages"] if m["role"] == "user")
        resp = _gen(q, rec["context"])
        gold = rec.get("gold_answers", [])
        abstained = is_abstention(resp)
        cg, reason = (False, "abstained") if abstained else (
            (None, "") if args.dry_run_judge else qa_judge(q, rec["context"], gold, resp))
        if args.dry_run_judge and not abstained:
            cg = True
        n_cg += bool(cg); n_abst += bool(abstained)
        results["answerable"].append({
            "id": rec["id"], "question": q, "response": resp, "gold": gold,
            "correct_and_grounded": cg, "abstained": abstained,
            "em": _em(resp, gold) if gold else None,
            "f1": _f1(resp, gold) if gold else None, "judge_reason": reason})
        if i % max(1, len(ans) // 20) == 0 or i == len(ans):
            eta = (time.time() - t0) / i * (len(ans) - i)
            print(f"  answerable [{i:4d}/{len(ans)}]  running H1={n_cg/i:.2f}  "
                  f"over-refusal={n_abst/i:.2f}  ETA {eta:4.0f}s", flush=True)

    t1 = time.time()
    n_hall = 0
    for i, rec in enumerate(una, 1):
        q = next(m["content"] for m in rec["messages"] if m["role"] == "user")
        resp = _gen(q, rec["context"])
        abst = is_abstention(resp)
        n_hall += (not abst)
        results["unanswerable"].append({
            "id": rec["id"], "question": q, "response": resp, "abstained": abst})
        if i % max(1, len(una) // 20) == 0 or i == len(una):
            eta = (time.time() - t1) / i * (len(una) - i)
            print(f"  unanswerable [{i:4d}/{len(una)}]  running abstain={1-n_hall/i:.2f}  "
                  f"hallucination={n_hall/i:.2f}  ETA {eta:4.0f}s", flush=True)

    metrics = _qa_metrics(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"metrics": metrics, "items": results}, out.open("w"), indent=2)
    print("-" * 68, flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"→ {out}  ({time.time()-t0:.0f}s total)", flush=True)


def _qa_metrics(results):
    ans, una = results["answerable"], results["unanswerable"]
    judged = [r for r in ans if r["correct_and_grounded"] is not None]
    h1 = (sum(r["correct_and_grounded"] for r in judged) / len(judged)) if judged else 0.0
    em = [r["em"] for r in ans if r["em"] is not None]
    f1 = [r["f1"] for r in ans if r["f1"] is not None]
    # Abstention confusion (positive = "should abstain")
    tp = sum(r["abstained"] for r in una)            # unanswerable & abstained
    fn = len(una) - tp                               # unanswerable & answered = hallucination
    fp = sum(r["abstained"] for r in ans)            # answerable & abstained (over-refusal)
    tn = len(ans) - fp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1_ab = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "H1_correct_grounded_rate": round(h1, 4), "H1_n_judged": len(judged),
        "squad_EM": round(sum(em) / len(em), 4) if em else None,
        "squad_F1": round(sum(f1) / len(f1), 4) if f1 else None,
        "H2_abstention_precision": round(prec, 4),
        "H2_abstention_recall": round(rec, 4),
        "H2_abstention_F1": round(f1_ab, 4),
        "hallucination_rate": round(fn / len(una), 4) if una else None,
        "over_refusal_rate": round(fp / len(ans), 4) if ans else None,
        "n_answerable": len(ans), "n_unanswerable": len(una),
        "H1_pass": h1 >= 0.70, "H2_pass": f1_ab >= 0.80,
    }


# ── H3 PersonaScore harness (R0 / R1) ────────────────────────────────────

def _refresh_turns(condition):
    return {15, 30} if condition == "R1" else set()


def cmd_persona(args):
    qpm_tag = "OFF" if args.no_qpm else "ON"
    gen = ADAGenerator(args.checkpoint, args.tokenizer, device=args.device,
                       use_qpm=not args.no_qpm)
    scripts = sorted(Path(args.scripts_dir).glob("script_*.json"))
    if args.limit:
        scripts = scripts[:args.limit]
    out_dir = Path(args.out_dir) / f"persona_{args.condition}"
    out_dir.mkdir(parents=True, exist_ok=True)
    refresh_at = _refresh_turns(args.condition)
    total = len(scripts)
    judge_tag = "dry-stub" if args.dry_run_judge else JUDGE_MODEL

    print("=" * 68, flush=True)
    print(f"=== PersonaScore harness — condition {args.condition} "
          f"(refresh @ {sorted(refresh_at) or 'none'}) ===", flush=True)
    print(f"    scripts={total}  probe_turns={PROBE_TURNS}  dims={DIMENSIONS}  "
          f"QPM={qpm_tag}  judge={judge_tag}  device={gen.device}", flush=True)
    print("=" * 68, flush=True)

    run_t0 = time.time()
    for si, sp in enumerate(scripts, 1):
        script = json.loads(sp.read_text())
        sid = int(script.get("script_id", re.search(r"(\d+)", sp.stem).group(1)))
        scores_path = out_dir / f"scores_{sid:03d}.jsonl"
        if scores_path.exists() and args.resume:
            print(f"  [{si:3d}/{total}] script {sid:03d} — skipped (already done)", flush=True)
            continue
        topic = str(script.get("topic", ""))[:48]
        print(f"\n  [{si:3d}/{total}] script {sid:03d} — {topic}", flush=True)
        turns = script["turns"][:args.n_turns]
        history: list[dict] = []
        rows = []
        t_start = time.time()
        for ti, turn in enumerate(turns, start=1):
            if ti in refresh_at:                     # R1: re-inject the SCI
                history.append({"role": "user",
                                "content": SCI_REFRESH_USER.format(sci_json=ADA_SCI_STR)})
                history.append({"role": "assistant", "content": SCI_REFRESH_ASSISTANT})
                print(f"      ★ SCI refresh at turn {ti} ({args.condition})", flush=True)
            user_msg, ctx = turn["user"], turn.get("context")
            reply = gen.generate(user_msg, context=ctx, history=history, temperature=0.0)
            # Side-channel probes (scored, NOT added to history) — Exp 1/2 protocol
            if ti in PROBE_TURNS:
                turn_scores = {}
                for dim, probe in get_probes_for_turn(ti, sid):
                    presp = gen.generate(probe, context=ctx, history=history, temperature=0.0)
                    if args.dry_run_judge:
                        score, reason = 3, "dry_run_stub"
                    else:
                        score, reason = persona_judge(probe, presp, dim)
                    turn_scores[dim] = score
                    rows.append({"script_id": sid, "turn": ti, "dimension": dim,
                                 "probe": probe, "response": presp,
                                 "score": score, "reason": reason,
                                 "condition": args.condition, "judge_model": JUDGE_MODEL})
                sc = " ".join(f"{d}={turn_scores[d]}" for d in DIMENSIONS)
                print(f"      turn {ti:2d}/{len(turns)}  probes  {sc}", flush=True)
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": reply})
        with scores_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        mean = sum(r["score"] for r in rows) / len(rows) if rows else 0.0
        done_dt = time.time() - t_start
        eta = (time.time() - run_t0) / si * (total - si)
        print(f"      → {len(rows)} probes, mean {mean:.2f}  ({done_dt:.0f}s)  "
              f"[{si}/{total} done, ETA {eta:.0f}s]", flush=True)
    print(f"\nPersonaScore {args.condition} complete → {out_dir}  "
          f"({time.time()-run_t0:.0f}s)", flush=True)


# ── Judge reliability (§6.5) ─────────────────────────────────────────────

def cmd_judge_reliability(args):
    rows = []
    for p in sorted(Path(args.scores_dir).glob("scores_*.jsonl")):
        rows += _load_jsonl(p)
    import random
    random.seed(args.seed)
    sample = random.sample(rows, max(1, int(len(rows) * args.frac)))
    print(f"=== Judge reliability — re-scoring {len(sample)}/{len(rows)} probes "
          f"({args.frac:.0%}) at T=0 ===", flush=True)
    primary, secondary = [], []
    for i, r in enumerate(sample, 1):
        s2, _ = persona_judge(r["probe"], r["response"], r["dimension"])
        primary.append(r["score"]); secondary.append(s2)
        if i % max(1, len(sample) // 10) == 0 or i == len(sample):
            print(f"  re-scored [{i:3d}/{len(sample)}]", flush=True)
    kw = cohens_kappa(primary, secondary, weighted=True)
    bin_p = [1 if s >= 4 else 0 for s in primary]
    bin_s = [1 if s >= 4 else 0 for s in secondary]
    kb = cohens_kappa(bin_p, bin_s, weighted=False, categories=[0, 1])
    res = {"n": len(sample), "kappa_weighted": round(kw, 4),
           "kappa_binary": round(kb, 4), "gate_pass": kw >= 0.70}
    print(json.dumps(res, indent=2))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, out.open("w"), indent=2)


# ── H3/H4 analysis + decision table (§3) ─────────────────────────────────

def _persona_curve(scores_dir):
    rows = []
    for p in sorted(Path(scores_dir).glob("scores_*.jsonl")):
        rows += _load_jsonl(p)
    by_turn = defaultdict(list)
    by_dim = defaultdict(list)
    for r in rows:
        by_turn[r["turn"]].append(r["score"])
        by_dim[r["dimension"]].append(r["score"])
    curve = {t: round(sum(v) / len(v), 4) for t, v in sorted(by_turn.items())}
    dims = {d: round(sum(v) / len(v), 4) for d, v in by_dim.items()}
    overall = round(sum(r["score"] for r in rows) / len(rows), 4) if rows else 0.0
    # T* = first probe turn where mean drops >= 0.3 below the turn-5 baseline
    tstar = None
    if curve:
        base = curve.get(min(curve), 0)
        for t in sorted(curve):
            if curve[t] <= base - 0.3:
                tstar = t; break
    return {"overall": overall, "per_turn": curve, "per_dimension": dims,
            "T_star": tstar, "n_probes": len(rows)}


def cmd_analyse(args):
    out = {"H3": {}, "H4": {}, "decision": {}}
    r0 = _persona_curve(Path(args.results_dir) / "persona_R0") if \
        (Path(args.results_dir) / "persona_R0").exists() else None
    r1 = _persona_curve(Path(args.results_dir) / "persona_R1") if \
        (Path(args.results_dir) / "persona_R1").exists() else None
    out["H3"]["R0"] = r0
    out["H3"]["R1"] = r1

    # H4 classification (plan §3 thresholds)
    if r0 and r1:
        a, b = r0["overall"], r1["overall"]
        if a >= 3.5 and abs(b - a) < 0.15:
            cls = "refresh-unnecessary"
        elif b - a >= 0.15:
            cls = "refresh-helpful"
        elif a < 3.5 and b < 3.5:
            cls = "refresh-insufficient"
        else:
            cls = "indeterminate"
        out["H4"] = {"R0_overall": a, "R1_overall": b, "delta": round(b - a, 4),
                     "classification": cls}

    # H1/H2 from qa_results if present
    qa_path = Path(args.results_dir) / "qa_results.json"
    if qa_path.exists():
        qm = json.load(qa_path.open())["metrics"]
        out["H1"] = {"rate": qm["H1_correct_grounded_rate"], "pass": qm["H1_pass"]}
        out["H2"] = {"F1": qm["H2_abstention_F1"], "pass": qm["H2_pass"],
                     "hallucination_rate": qm["hallucination_rate"]}
        best = max([x for x in (a if r0 else None, b if r1 else None) if x is not None],
                   default=0.0) if (r0 or r1) else 0.0
        h3_pass = best >= 3.5
        out["H3"]["best_overall"] = best
        out["H3"]["pass"] = h3_pass
        out["decision"] = _decide(qm["H1_pass"], qm["H2_pass"], h3_pass)

    op = Path(args.results_dir) / "analysis_data.json"
    json.dump(out, op.open("w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"→ {op}")

    if not getattr(args, "no_plots", False):
        qm_full = qm if qa_path.exists() else None
        _render_figures(out, qm_full, Path(args.results_dir))


def _decide(h1, h2, h3):
    if h1 and h2 and h3:
        return {"row": "✓✓✓", "action": "Direction validated — lock ADA knowledge-agent v0; "
                "apply H4 refresh recommendation; proceed to next scenario."}
    if h1 and h2 and not h3:
        return {"row": "✓✓✗", "action": "Competent QA, weak persona — add persona/episodic "
                "SFT data; retrain SFT only."}
    if h1 and not h2:
        return {"row": "✓✗—", "action": "Over-confident — add unanswerable negatives + Sonnet "
                "refusal data; retrain SFT."}
    return {"row": "✗——", "action": "80M from-scratch below usability — trigger fallback: "
            "fine-tune small pretrained base (RQ5); re-run §6 on it."}


# ── Figures (matplotlib; style matches Exp 1/2/5) ────────────────────────

_COL = {"R0": "#1f77b4", "R1": "#d62728", "qpm_on": "#2ca02c",
        "qpm_off": "#9e9e9e", "bar": "#4c78a8", "thresh": "black"}
_PASS_PS, _REFRESH_TURNS, _H1_T, _H2_T = 3.5, (15, 30), 0.70, 0.80


def _savefig(fig, stem, results_dir):
    import matplotlib.pyplot as plt
    results_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(results_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {stem}.png/.pdf/.svg", flush=True)


def _plot_turn_series(h3, results_dir):
    import matplotlib.pyplot as plt
    conds = [("R0", "R0 (no refresh)"), ("R1", "R1 (refresh @15/30)")]
    have = [(c, lab) for c, lab in conds if h3.get(c) and h3[c].get("per_turn")]
    if not have:
        print("  skip turn_series — no R0/R1 per-turn data"); return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cond, label in have:
        c = h3[cond]
        pt = c["per_turn"]
        turns = sorted(int(t) for t in pt)
        means = [pt.get(str(t), pt.get(t)) for t in turns]
        ax.plot(turns, means, marker="o", linewidth=2.0, color=_COL[cond],
                label=f"{label}  (mean={c['overall']:.2f})")
        ts = c.get("T_star")
        if ts is not None and str(ts) in {str(t) for t in turns}:
            y = pt.get(str(ts), pt.get(ts))
            ax.scatter([ts], [y], marker="*", s=220, color=_COL[cond], zorder=5,
                       edgecolor="black", linewidth=0.6)
            ax.annotate(f"T*={ts}", (ts, y), textcoords="offset points",
                        xytext=(6, -14), fontsize=9, color=_COL[cond])
    for rt in _REFRESH_TURNS:
        ax.axvline(rt, color="grey", linestyle=":", alpha=0.5)
    ax.axhline(_PASS_PS, color=_COL["thresh"], linestyle="--", alpha=0.4,
               label=f"H3 threshold ({_PASS_PS})")
    ax.set_xlabel("Probe turn"); ax.set_ylabel("PersonaScore"); ax.set_ylim(1, 5)
    ax.set_title("Exp 6 — ADA PersonaScore per probe turn (R0 vs R1)")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    _savefig(fig, "exp6_personascore_turn_series", results_dir)


def _plot_dimension_bars(h3, results_dir):
    import numpy as np, matplotlib.pyplot as plt
    have = [c for c in ("R0", "R1") if h3.get(c) and h3[c].get("per_dimension")]
    if not have:
        print("  skip dimension_bars — no per-dimension data"); return
    dims = list(DIMENSIONS)
    x = np.arange(len(dims)); w = 0.8 / len(have)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, cond in enumerate(have):
        pd = h3[cond]["per_dimension"]
        vals = [pd.get(d, 0.0) for d in dims]
        off = (i - (len(have) - 1) / 2) * w
        ax.bar(x + off, vals, w, color=_COL[cond], alpha=0.9,
               label=f"{cond} (mean={h3[cond]['overall']:.2f})")
    ax.axhline(_PASS_PS, color=_COL["thresh"], linestyle="--", alpha=0.4,
               label=f"H3 threshold ({_PASS_PS})")
    ax.set_xticks(x)
    ax.set_xticklabels(["Trait (T)", "Episodic (E)", "Capability (C)", "Style (S)"], fontsize=9)
    ax.set_ylabel("Mean PersonaScore"); ax.set_ylim(0, 5)
    ax.set_title("Exp 6 — PersonaScore by dimension (R0 vs R1)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    _savefig(fig, "exp6_dimension_bars", results_dir)


def _plot_qa_metrics(metrics, results_dir):
    import matplotlib.pyplot as plt
    order = [("H1_correct_grounded_rate", "H1 correct+grounded"),
             ("squad_EM", "SQuAD2 EM"), ("squad_F1", "SQuAD2 F1"),
             ("H2_abstention_F1", "H2 abstention F1"),
             ("hallucination_rate", "hallucination"),
             ("over_refusal_rate", "over-refusal")]
    labels, vals = [], []
    for k, lab in order:
        v = metrics.get(k)
        if v is not None:
            labels.append(lab); vals.append(float(v))
    if not vals:
        print("  skip qa_metrics — no QA metrics"); return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(range(len(vals)), vals, color=_COL["bar"], alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(_H1_T, color="#2ca02c", linestyle="--", alpha=0.6, label=f"H1 threshold ({_H1_T})")
    ax.axhline(_H2_T, color="#d62728", linestyle="--", alpha=0.6, label=f"H2 threshold ({_H2_T})")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Score / rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Exp 6 — QA competence (H1) and calibrated abstention (H2)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    _savefig(fig, "exp6_qa_metrics", results_dir)


def _plot_qpm_ablation(results_dir):
    import numpy as np, matplotlib.pyplot as plt
    on_dir, off_dir = results_dir / "persona_R0", results_dir / "ablation_noqpm" / "persona_R0"
    if not (on_dir.exists() and off_dir.exists()):
        print("  skip qpm_ablation — need persona_R0 and ablation_noqpm/persona_R0"); return
    on, off = _persona_curve(on_dir), _persona_curve(off_dir)
    dims = list(DIMENSIONS)
    on_vals = [on["per_dimension"].get(d, 0.0) for d in dims] + [on["overall"]]
    off_vals = [off["per_dimension"].get(d, 0.0) for d in dims] + [off["overall"]]
    x = np.arange(len(dims) + 1); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w / 2, on_vals, w, color=_COL["qpm_on"], alpha=0.9,
           label=f"QPM-on  (overall={on['overall']:.2f})")
    ax.bar(x + w / 2, off_vals, w, color=_COL["qpm_off"], alpha=0.9,
           label=f"QPM-off (overall={off['overall']:.2f})")
    ax.axhline(_PASS_PS, color=_COL["thresh"], linestyle="--", alpha=0.4,
               label=f"H3 threshold ({_PASS_PS})")
    ax.set_xticks(x); ax.set_xticklabels([*dims, "Overall"], fontsize=9)
    ax.set_ylabel("Mean PersonaScore"); ax.set_ylim(0, 5)
    ax.set_title("Exp 6 — QPM-as-weight-supervision ablation (RQ6, R0)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    _savefig(fig, "exp6_qpm_ablation", results_dir)


def _render_figures(out, qa_metrics, results_dir):
    """Render all Exp 6 figures (png/pdf/svg) from the analysis output."""
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  (matplotlib not installed — skipping figures; pip install matplotlib)")
        return
    print(f"Rendering figures → {results_dir}", flush=True)
    if out.get("H3"):
        _plot_turn_series(out["H3"], results_dir)
        _plot_dimension_bars(out["H3"], results_dir)
    if qa_metrics:
        _plot_qa_metrics(qa_metrics, results_dir)
    _plot_qpm_ablation(results_dir)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("qa")
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--tokenizer", default="tokenizer/ada_bpe.json")
    q.add_argument("--answerable", default="data/eval_answerable.jsonl")
    q.add_argument("--unanswerable", default="data/eval_unanswerable.jsonl")
    q.add_argument("--out", default="results/qa_results.json")
    q.add_argument("--device", default=None)
    q.add_argument("--no-qpm", action="store_true", help="disable QPM persona_state conditioning")
    q.add_argument("--span", action="store_true",
                   help="extractive span-head decoding (Phase 1) — predict answer "
                        "start/end over the context instead of generating")
    q.add_argument("--constrained", action="store_true",
                   help="grounded decode: answer must be a context span or abstain")
    q.add_argument("--answerable-threshold", type=float, default=0.5,
                   help="span mode: abstain when the answerability head's "
                        "P(answerable) < this (only if the checkpoint has one)")
    q.add_argument("--abstain-threshold", type=float, default=0.0,
                   help="constrained decode: abstain when best context-start prob < τ")
    q.add_argument("--rerank", action="store_true",
                   help="shrink context to the top-k question-relevant sentences (MiniLM) before decode")
    q.add_argument("--rerank-topk", type=int, default=2,
                   help="number of context sentences to keep when --rerank is on")
    q.add_argument("--retriever", action="store_true",
                   help="symbolic extract-then-style: NER answer selector supplies the answer (or abstains)")
    q.add_argument("--retriever-topk", type=int, default=2,
                   help="sentences the symbolic retriever searches for candidate spans")
    q.add_argument("--retriever-tau", type=float, default=0.0,
                   help="symbolic retriever: abstain when best candidate score < τ")
    q.add_argument("--limit", type=int, default=None)
    q.add_argument("--max-new-tokens", type=int, default=160,
                   help="cap free-decode length; QA answers are short — use ~24-32 "
                        "(much faster, esp. for prefix-LM whose masked attention is slow)")
    q.add_argument("--dry-run-judge", action="store_true")
    q.set_defaults(func=cmd_qa)

    p = sub.add_parser("persona")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default="tokenizer/ada_bpe.json")
    p.add_argument("--scripts-dir", default="data/persona_scripts")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--condition", choices=["R0", "R1"], required=True)
    p.add_argument("--n-turns", type=int, default=40)
    p.add_argument("--device", default=None)
    p.add_argument("--no-qpm", action="store_true", help="disable QPM persona_state conditioning")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run-judge", action="store_true")
    p.set_defaults(func=cmd_persona)

    jr = sub.add_parser("judge-reliability")
    jr.add_argument("--scores-dir", required=True)
    jr.add_argument("--frac", type=float, default=0.05)
    jr.add_argument("--seed", type=int, default=1337)
    jr.add_argument("--out", default="results/judge_reliability.json")
    jr.set_defaults(func=cmd_judge_reliability)

    an = sub.add_parser("analyse")
    an.add_argument("--results-dir", default="results")
    an.add_argument("--no-plots", action="store_true", help="skip rendering figures")
    an.set_defaults(func=cmd_analyse)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
