#!/usr/bin/env python3
"""Aggregate Experiment 7 results into the RQ1–RQ4 decision table + figures (plan §3, §7).

Reads results/{C0,C1,C2}/scores_*.jsonl and produces:
  * recall-vs-lag curves per condition (headline, RQ2/H2);
  * extended PersonaScore (overall + per-dimension + per-turn), RQ3/H3;
  * self-consistency (agreement across repeated recall probes),
  * cost profile (injected-memory tokens / prompt length / latency vs store size), RQ4/H4;
  * the pre-registered decision table (§3).

Recall "accuracy" = fraction of recall probes scoring ≥ 4 on the 1–5 recall rubric.
"Out-of-window" anchors are recall probes with lag ≥ --oow-lag (default 40) — beyond
what the C1 sliding window can hold — the set H2 is judged on.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

CONDITIONS = ("C0", "C1", "C2")
EXP6_PERSONA = 3.80              # 40-turn reference (H1 flatness band)
_COL = {"C0": "#9e9e9e", "C1": "#1f77b4", "C2": "#2ca02c", "thresh": "black"}


def _load(results_dir, condition):
    rows = []
    d = Path(results_dir) / condition
    for p in sorted(d.glob("scores_*.jsonl")):
        rows += [json.loads(l) for l in p.open() if l.strip()]
    return rows


def _acc(rows):
    return (sum(1 for r in rows if r["score"] >= 4) / len(rows)) if rows else 0.0


def analyse(results_dir, oow_lag=40):
    data = {c: _load(results_dir, c) for c in CONDITIONS}
    out = {"recall": {}, "persona": {}, "consistency": {}, "cost": {}, "params": {"oow_lag": oow_lag}}

    for c in CONDITIONS:
        rows = data[c]
        recall = [r for r in rows if r["kind"] == "recall"]
        persona = [r for r in rows if r["kind"] == "persona"]
        consist = [r for r in rows if r["kind"] == "consistency"]
        cost = [r for r in rows if r["kind"] == "cost"]
        if not rows:
            continue

        # ── recall by lag ──
        by_lag = defaultdict(list)
        for r in recall:
            by_lag[r.get("lag", -1)].append(r)
        out["recall"][c] = {
            "overall_acc": round(_acc(recall), 4),
            "overall_mean": round(mean(r["score"] for r in recall), 4) if recall else None,
            "n": len(recall),
            "by_lag": {str(l): {"acc": round(_acc(v), 4), "mean": round(mean(x["score"] for x in v), 3),
                                "n": len(v)} for l, v in sorted(by_lag.items())},
            "out_of_window_acc": round(_acc([r for r in recall if r.get("lag", 0) >= oow_lag]), 4),
            "out_of_window_n": sum(1 for r in recall if r.get("lag", 0) >= oow_lag),
        }

        # ── persona (overall / per-dim / per-turn) ──
        if persona:
            by_dim, by_turn = defaultdict(list), defaultdict(list)
            for r in persona:
                by_dim[r["dimension"]].append(r["score"]); by_turn[r["turn"]].append(r["score"])
            out["persona"][c] = {
                "overall": round(mean(r["score"] for r in persona), 4),
                "per_dimension": {d: round(mean(v), 3) for d, v in by_dim.items()},
                "per_turn": {str(t): round(mean(v), 3) for t, v in sorted(by_turn.items())},
                "n": len(persona),
            }

        # ── self-consistency: agreement across repeated probes of one anchor ──
        groups = defaultdict(list)
        for r in consist:
            groups[(r["script_id"], r["anchor_id"])].append(r["score"])
        stds = [pstdev(v) for v in groups.values() if len(v) >= 2]
        out["consistency"][c] = {
            "n_groups": len(groups),
            "mean_within_group_std": round(mean(stds), 3) if stds else None,
            "stable_frac": round(sum(1 for s in stds if s <= 0.5) / len(stds), 3) if stds else None,
        }

        # ── cost vs store size (O(1) check) ──
        if cost:
            lo = [r for r in cost if r["store_size"] <= 50]
            hi = [r for r in cost if r["store_size"] >= 200]
            out["cost"][c] = {
                "mean_mem_tokens": round(mean(r["mem_tokens"] for r in cost), 1),
                "mean_prompt_tokens": round(mean(r["prompt_tokens"] for r in cost), 1),
                "mean_latency_s": round(mean(r["gen_latency_s"] for r in cost), 3),
                "prompt_tokens_lo_store": round(mean(r["prompt_tokens"] for r in lo), 1) if lo else None,
                "prompt_tokens_hi_store": round(mean(r["prompt_tokens"] for r in hi), 1) if hi else None,
                "latency_lo_store": round(mean(r["gen_latency_s"] for r in lo), 3) if lo else None,
                "latency_hi_store": round(mean(r["gen_latency_s"] for r in hi), 3) if hi else None,
            }

    out["decision"] = _decide(out, oow_lag)
    return out


def _decide(out, oow_lag):
    r, p, cost = out["recall"], out["persona"], out["cost"]
    d = {}
    # H1 — statelessness floor (C0): persona flat near 3.80, recall at fabrication baseline
    if "C0" in p and "C0" in r:
        turns = p["C0"]["per_turn"].values()
        d["H1_persona_flat"] = bool(turns and abs(p["C0"]["overall"] - EXP6_PERSONA) <= 0.2 + 1e-9
                                    and (max(turns) - min(turns)) <= 0.6)
        d["H1_recall_floor"] = bool(r["C0"]["overall_mean"] is not None and r["C0"]["overall_mean"] <= 2.0)
    # H2 — episodic-RAG recall on out-of-window anchors ≥ 0.70 and > C0, > C1
    if all(c in r for c in CONDITIONS):
        c2, c1, c0 = (r["C2"]["out_of_window_acc"], r["C1"]["out_of_window_acc"], r["C0"]["out_of_window_acc"])
        d["H2_c2_oow_acc"] = c2
        d["H2_pass"] = bool(c2 >= 0.70 and c2 > c0 and c2 > c1)
    # H3 — persona holds at length under C2
    if "C2" in p:
        turns = p["C2"]["per_turn"]
        d["H3_min_turn_score"] = round(min(turns.values()), 3) if turns else None
        d["H3_pass"] = bool(p["C2"]["overall"] >= 3.5 and all(v >= 3.5 for v in turns.values()))
    # H4 — bounded cost (C2 prompt tokens & latency at high store within 1.5x of low store)
    if "C2" in cost and cost["C2"].get("prompt_tokens_hi_store") and cost["C2"].get("prompt_tokens_lo_store"):
        cc = cost["C2"]
        ratio_ctx = cc["prompt_tokens_hi_store"] / max(cc["prompt_tokens_lo_store"], 1)
        d["H4_ctx_ratio"] = round(ratio_ctx, 3)
        d["H4_pass"] = bool(ratio_ctx <= 1.5)
    # overall action (plan §3 decision rule)
    if d.get("H2_pass") and d.get("H3_pass"):
        d["action"] = "Adopt episodic-RAG as the long-running agent's memory architecture."
    elif d.get("H2_pass") and d.get("H3_pass") is False:
        d["action"] = "Recall works but persona degrades — tune injection format; re-run persona."
    elif d.get("H2_pass") is False:
        d["action"] = "Frozen model cannot consume retrieved memory zero-shot — trigger Phase B (memory-aware SFT)."
    return d


# ── figures ──────────────────────────────────────────────────────────────

def _savefig(fig, stem, out_dir):
    import matplotlib.pyplot as plt
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(Path(out_dir) / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {stem}.png/.pdf/.svg", flush=True)


def render_figures(out, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping figures)"); return

    # recall vs lag
    if any(c in out["recall"] for c in CONDITIONS):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for c in CONDITIONS:
            rc = out["recall"].get(c)
            if not rc:
                continue
            lags = sorted(int(l) for l in rc["by_lag"] if int(l) >= 0)
            ax.plot(lags, [rc["by_lag"][str(l)]["acc"] for l in lags], marker="o", color=_COL[c],
                    linewidth=2, label=f"{c} (oow acc {rc['out_of_window_acc']:.2f})")
        ax.axhline(0.70, color=_COL["thresh"], ls="--", alpha=0.4, label="H2 threshold (0.70)")
        ax.set_xlabel("lag (turns since anchor planted)"); ax.set_ylabel("recall accuracy (score ≥ 4)")
        ax.set_ylim(0, 1.02); ax.set_title("Exp 7 — long-range recall vs lag (C0 / C1 / C2)")
        ax.legend(loc="upper right", fontsize=9); _savefig(fig, "exp7_recall_vs_lag", out_dir)

    # persona per turn
    if any(c in out["persona"] for c in CONDITIONS):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for c in CONDITIONS:
            pc = out["persona"].get(c)
            if not pc:
                continue
            turns = sorted(int(t) for t in pc["per_turn"])
            ax.plot(turns, [pc["per_turn"][str(t)] for t in turns], marker=".", color=_COL[c],
                    linewidth=1.6, label=f"{c} (mean {pc['overall']:.2f})")
        ax.axhline(3.5, color=_COL["thresh"], ls="--", alpha=0.4, label="H3 threshold (3.5)")
        ax.set_xlabel("probe turn"); ax.set_ylabel("PersonaScore"); ax.set_ylim(1, 5)
        ax.set_title("Exp 7 — PersonaScore over long horizons"); ax.legend(loc="lower left", fontsize=9)
        _savefig(fig, "exp7_persona_turn_series", out_dir)

    # cost vs store
    if any(c in out["cost"] for c in CONDITIONS):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        labels = ["mem_tokens", "prompt_tokens"]
        import numpy as np
        x = np.arange(len(labels)); w = 0.8 / max(len(out["cost"]), 1)
        for i, c in enumerate([c for c in CONDITIONS if c in out["cost"]]):
            cc = out["cost"][c]
            ax.bar(x + (i - 0.5) * w, [cc["mean_mem_tokens"], cc["mean_prompt_tokens"]], w,
                   color=_COL[c], label=c)
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("mean tokens / turn")
        ax.set_title("Exp 7 — injected-context cost by condition (RQ4)"); ax.legend(fontsize=9)
        _savefig(fig, "exp7_cost", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--oow-lag", type=int, default=40, help="lag threshold for 'out-of-window' (H2)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out = analyse(args.results_dir, oow_lag=args.oow_lag)
    op = Path(args.results_dir) / "analysis_data.json"
    json.dump(out, op.open("w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"→ {op}")
    if not args.no_plots:
        render_figures(out, args.results_dir)


if __name__ == "__main__":
    main()
