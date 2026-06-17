#!/usr/bin/env python3
"""
Analysis for CA Experiment 5: Logits-Level QPM→SLM Steering via Residual
Stream Injection.

Reads per-condition JSONL logs and produces:

  • Per-condition mean PersonaScore + by-dimension and by-turn breakdowns
  • Continuity check (Condition A vs Experiment 4 Condition A: 4.4385)
  • Primary — H_logits: paired t-test B vs A  (plan §6.1)
  • Secondary — H_coherence: paired t-test D vs B  (plan §6.2)
  • Exploratory — H_channel: C vs A and C vs B with interaction classification  (plan §6.3)
  • Dimension analysis — T/E/C/S breakdowns with Episodic degradation check  (plan §6.4)
  • Turn-level PersonaScore plot — 4 conditions × 8 probe turns  (plan §6.5)
  • Composite vector norm plot — B/C/D per turn  (§8.5 diagnostic)
  • Updated effect-size ladder (Exp 3–5 progression)  (plan §6.6)
  • Decision-rule outcome JSON  (plan §6.7)

Usage:
  python3 analyse_results.py
  python3 analyse_results.py --conditions A,B
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ca_assets as A

# ── Paths ─────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
LOGS_BASE   = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# ── Reference anchors (frozen from Exp 3 / 4) ────────────────────────────

EXP3_H1_COHENS_D = 21.51   # Order effects (JSD, QPM vs CMG)
EXP3_H2_COHENS_D = 2.59    # Ambivalence (entropy, QPM vs CMG)
EXP3_H3_COHENS_D = 0.032   # PersonaScore (QPM vs CMG, JSON marginals)
EXP4_BEST_DZ = -0.074      # Exp 4 best condition D vs A (bivariate coactivations)
EXP4_COND_A_MEAN = 4.4385  # Exp 4 Condition A mean PersonaScore (continuity anchor)

CONTINUITY_TOLERANCE = 0.05     # plan §6.1

# Hypothesis thresholds (plan §6.1–6.2)
H_LOGITS_P = 0.05
H_LOGITS_D = 0.20      # primary: B vs A
H_COHERENCE_P = 0.05
H_COHERENCE_D = 0.10   # secondary: D vs B (relaxed)
H_CHANNEL_ADDITIVE_MARGIN = 0.05    # C > max(A,B) by ≥ 0.05 → additive
H_CHANNEL_DOMINANT_MARGIN = 0.05   # |C − X| < 0.05 → dominant


# ── Helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_condition_scores(condition: str, profile: str = "psychotherapy") -> list[dict]:
    cond_dir = LOGS_BASE / f"condition_{condition.lower()}_{profile}"
    if not cond_dir.exists():
        return []
    rows = []
    for path in sorted(cond_dir.glob(f"scores_condition_{condition.lower()}_*.jsonl")):
        rows.extend(load_jsonl(path))
    return rows


def load_condition_context(condition: str, profile: str = "psychotherapy") -> list[dict]:
    cond_dir = LOGS_BASE / f"condition_{condition.lower()}_{profile}"
    if not cond_dir.exists():
        return []
    rows = []
    for path in sorted(cond_dir.glob(f"context_condition_{condition.lower()}_*.jsonl")):
        rows.extend(load_jsonl(path))
    return rows


def _score_key(rec: dict) -> tuple:
    return (rec["script_id"], rec["turn"], rec["dimension"])


def paired_ttest(a: list[float], b: list[float]) -> dict:
    """Paired t-test: a − b.  Positive d_z means a > b."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diffs = a_arr - b_arr
    n = len(diffs)
    if n == 0:
        return {"n": 0}
    t_stat, p_val = stats.ttest_rel(a_arr, b_arr)
    sd = float(np.std(diffs, ddof=1))
    d_z = float(np.mean(diffs) / (sd + 1e-12))
    if n > 1 and sd > 0:
        ci_lo, ci_hi = stats.t.interval(
            0.95, df=n - 1,
            loc=float(np.mean(diffs)),
            scale=stats.sem(diffs),
        )
    else:
        ci_lo = ci_hi = float(np.mean(diffs))
    return {
        "n":         n,
        "mean_a":    round(float(np.mean(a_arr)), 4),
        "mean_b":    round(float(np.mean(b_arr)), 4),
        "mean_diff": round(float(np.mean(diffs)), 4),
        "ci_95":     [round(float(ci_lo), 4), round(float(ci_hi), 4)],
        "t":         round(float(t_stat), 4),
        "p":         float(p_val),
        "cohens_d":  round(d_z, 4),
        "sd_diff":   round(sd, 4),
    }


def savefig(fig, stem: str, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(results_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Main analysis ────────────────────────────────────────────────────────

def analyse(conditions: list[str], profile: str = "psychotherapy") -> dict:
    print(f"\n=== Experiment 5 analysis — profile: {profile} ===", flush=True)

    # 1. Load scores by condition
    data: dict[str, dict[tuple, float]] = {}
    for cond in conditions:
        rows = load_condition_scores(cond, profile=profile)
        per_key: dict[tuple, float] = {}
        for rec in rows:
            if "score" not in rec:
                continue
            per_key[_score_key(rec)] = float(rec["score"])
        data[cond] = per_key
        print(f"  Condition {cond}: {len(per_key):>4} scored probes", flush=True)

    # 2. Per-condition summary
    per_condition_summary: dict[str, dict] = {}
    for cond, per_key in data.items():
        all_scores = list(per_key.values())
        if not all_scores:
            per_condition_summary[cond] = {"n": 0}
            continue
        per_dim = defaultdict(list)
        per_turn = defaultdict(list)
        for (sid, turn, dim), score in per_key.items():
            per_dim[dim].append(score)
            per_turn[turn].append(score)
        per_condition_summary[cond] = {
            "n":       len(all_scores),
            "mean":    round(float(np.mean(all_scores)), 4),
            "sd":      round(float(np.std(all_scores, ddof=1)), 4),
            "by_dim":  {d: round(float(np.mean(v)), 4) for d, v in sorted(per_dim.items())},
            "by_turn": {int(t): round(float(np.mean(v)), 4) for t, v in sorted(per_turn.items())},
        }

    # 3. Continuity check — Condition A vs Exp 4 Condition A (plan §6.1)
    cond_a_mean = per_condition_summary.get("A", {}).get("mean")
    continuity = None
    if cond_a_mean is not None:
        deviation = round(cond_a_mean - EXP4_COND_A_MEAN, 4)
        continuity = {
            "exp4_cond_a_mean":     EXP4_COND_A_MEAN,
            "exp5_cond_a_mean":     cond_a_mean,
            "deviation":            deviation,
            "tolerance":            CONTINUITY_TOLERANCE,
            "within_tolerance":     abs(deviation) <= CONTINUITY_TOLERANCE,
        }
        verdict = "OK" if continuity["within_tolerance"] else "FAIL — investigate confound"
        print(f"\n  Continuity check: Cond A mean = {cond_a_mean:.4f}, "
              f"Exp 4 Cond A = {EXP4_COND_A_MEAN}, "
              f"Δ = {deviation:+.4f} → {verdict}", flush=True)

    # 4. Hypothesis tests
    # H_logits: B vs A (primary, plan §6.1)
    h_logits_result = {}
    if "A" in data and "B" in data:
        shared = sorted(set(data["A"]) & set(data["B"]))
        b_scores = [data["B"][k] for k in shared]
        a_scores = [data["A"][k] for k in shared]
        res = paired_ttest(b_scores, a_scores)  # B − A
        res["passes"] = (
            res.get("mean_diff", 0) > 0
            and res.get("p", 1.0) < H_LOGITS_P
            and res.get("cohens_d", 0.0) >= H_LOGITS_D
        )
        h_logits_result = res
        print(f"\n  H_logits (B vs A)  n={res['n']}")
        print(f"    mean_B={res['mean_a']}  mean_A={res['mean_b']}  "
              f"Δ={res['mean_diff']:+}  t={res['t']}  "
              f"p={res['p']:.4g}  d_z={res['cohens_d']:+.4f}  "
              f"95%CI {res['ci_95']}")
        print(f"    H_logits PASS? {res['passes']}", flush=True)

    # H_coherence: D vs B (secondary, plan §6.2)
    h_coherence_result = {}
    if "B" in data and "D" in data:
        shared = sorted(set(data["B"]) & set(data["D"]))
        d_scores = [data["D"][k] for k in shared]
        b_scores2 = [data["B"][k] for k in shared]
        res = paired_ttest(d_scores, b_scores2)  # D − B
        res["passes"] = (
            res.get("mean_diff", 0) > 0
            and res.get("p", 1.0) < H_COHERENCE_P
            and res.get("cohens_d", 0.0) >= H_COHERENCE_D
        )
        h_coherence_result = res
        print(f"\n  H_coherence (D vs B)  n={res['n']}")
        print(f"    mean_D={res['mean_a']}  mean_B={res['mean_b']}  "
              f"Δ={res['mean_diff']:+}  t={res['t']}  "
              f"p={res['p']:.4g}  d_z={res['cohens_d']:+.4f}")
        print(f"    H_coherence PASS? {res['passes']}", flush=True)

    # H_channel: C vs A and C vs B (exploratory, plan §6.3)
    h_channel_result = {}
    if "A" in data and "B" in data and "C" in data:
        shared_ca = sorted(set(data["C"]) & set(data["A"]))
        shared_cb = sorted(set(data["C"]) & set(data["B"]))
        c_vs_a = paired_ttest(
            [data["C"][k] for k in shared_ca],
            [data["A"][k] for k in shared_ca],
        )
        c_vs_b = paired_ttest(
            [data["C"][k] for k in shared_cb],
            [data["B"][k] for k in shared_cb],
        )
        # Classify channel interaction (plan §6.3)
        mean_a = per_condition_summary.get("A", {}).get("mean")
        mean_b = per_condition_summary.get("B", {}).get("mean")
        mean_c = per_condition_summary.get("C", {}).get("mean")
        interaction = _classify_channel_interaction(mean_a, mean_b, mean_c)
        h_channel_result = {
            "c_vs_a":    c_vs_a,
            "c_vs_b":    c_vs_b,
            "interaction": interaction,
        }
        print(f"\n  H_channel (C vs A)  Δ={c_vs_a.get('mean_diff', 0):+.4f}  "
              f"p={c_vs_a.get('p', 1):.4g}")
        print(f"  H_channel (C vs B)  Δ={c_vs_b.get('mean_diff', 0):+.4f}  "
              f"p={c_vs_b.get('p', 1):.4g}")
        print(f"  Channel interaction: {interaction}", flush=True)

    # 5. Dimension analysis (plan §6.4)
    dim_deltas: dict[str, dict[str, float]] = {}
    if "A" in per_condition_summary:
        a_by_dim = per_condition_summary["A"].get("by_dim", {})
        for cond in ("B", "C", "D"):
            if cond not in per_condition_summary:
                continue
            x_by_dim = per_condition_summary[cond].get("by_dim", {})
            dim_deltas[cond] = {
                d: round(x_by_dim.get(d, 0) - a_by_dim.get(d, 0), 4)
                for d in sorted(set(a_by_dim) | set(x_by_dim))
            }
        print("\n  Dimension deltas vs Condition A:")
        for cond, deltas in sorted(dim_deltas.items()):
            print(f"    Cond {cond}: " + "  ".join(f"{d}:{v:+.4f}" for d, v in sorted(deltas.items())))

    # 6. Effect-size ladder (plan §6.6)
    b_vs_a_d = h_logits_result.get("cohens_d", float("nan"))
    d_vs_a_d = float("nan")
    if "A" in data and "D" in data:
        shared_da = sorted(set(data["D"]) & set(data["A"]))
        if shared_da:
            d_vs_a = paired_ttest(
                [data["D"][k] for k in shared_da],
                [data["A"][k] for k in shared_da],
            )
            d_vs_a_d = d_vs_a.get("cohens_d", float("nan"))

    ladder = [
        {"experiment": "Exp 3 H1",   "metric": "Order effects (JSD QPM vs CMG)",               "d": EXP3_H1_COHENS_D},
        {"experiment": "Exp 3 H2",   "metric": "Ambivalence (entropy QPM vs CMG)",              "d": EXP3_H2_COHENS_D},
        {"experiment": "Exp 3 H3",   "metric": "PersonaScore (QPM vs CMG, JSON marginals)",     "d": EXP3_H3_COHENS_D},
        {"experiment": "Exp 4 best", "metric": "PersonaScore (D vs A, bivariate coactivations)","d": EXP4_BEST_DZ},
        {"experiment": "Exp 5 B",    "metric": "PersonaScore (B vs A, diagonal steering)",      "d": round(b_vs_a_d, 4) if not np.isnan(b_vs_a_d) else None},
        {"experiment": "Exp 5 D",    "metric": "PersonaScore (D vs A, diagonal+coherence)",     "d": round(d_vs_a_d, 4) if not np.isnan(d_vs_a_d) else None},
    ]

    # 7. Decision rule (plan §6.7)
    h_logits_pass = h_logits_result.get("passes", False)
    h_coherence_pass = h_coherence_result.get("passes", False)

    if continuity and not continuity["within_tolerance"]:
        decision_rule = (
            "Condition A deviates > ±0.05 from Exp 4 Cond A — pause and "
            "investigate confound before interpreting B/C/D."
        )
        paper_update = "Methodological finding only — no §5.7 update yet."
    elif h_logits_pass and h_coherence_pass:
        decision_rule = (
            "H_logits PASSED and H_coherence PASSED — logits-level interface works; "
            f"QPM coherence contributes independently (d_z(D−B) = "
            f"{h_coherence_result.get('cohens_d', 'N/A')})."
        )
        paper_update = (
            "Formalise Condition D as production interface in §5.7. "
            "Add Exp 5 as §15.4.5. Update Conclusions: QPM coherence reaches "
            "SLM behaviour via residual stream injection."
        )
    elif h_logits_pass and not h_coherence_pass:
        decision_rule = (
            "H_logits PASSED; H_coherence FAILED — logits-level works; "
            "marginals alone sufficient; coherence adds no independent signal."
        )
        paper_update = (
            "Formalise Condition B as production interface in §5.7. "
            "Add coherence-null note in §5.7.4. Add Exp 5 as §15.4.5."
        )
    else:
        decision_rule = (
            "H_logits FAILED — logits-level interface did not recover QPM advantage "
            f"(d_z(B−A) = {h_logits_result.get('cohens_d', 'N/A')}). "
            "QPM advantage unrecoverable at 7B output level via this mechanism."
        )
        paper_update = (
            "Add §5.7.5 'Fundamental Interface Limitations'. "
            "Exp 6 recommendation: smaller (3B) model or task domain shift."
        )

    # H_channel addendum
    channel_note = ""
    interaction = h_channel_result.get("interaction", "")
    if interaction == "Interfering":
        channel_note = ("H_channel = Interfering: dual-channel operation is contraindicated "
                        "at 7B scale — add §8.6 implementation note.")
    elif interaction == "Additive":
        channel_note = ("H_channel = Additive: production deployments should enable both channels.")

    summary = {
        "profile":              profile,
        "conditions":           conditions,
        "per_condition":        per_condition_summary,
        "continuity_check":     continuity,
        "h_logits":             h_logits_result,
        "h_coherence":          h_coherence_result,
        "h_channel":            h_channel_result,
        "dimension_deltas":     dim_deltas,
        "hypotheses": {
            "H_logits":    h_logits_pass,
            "H_coherence": h_coherence_pass,
            "H_channel":   h_channel_result.get("interaction", "unknown"),
        },
        "effect_size_ladder":   ladder,
        "decision_rule_outcome": decision_rule,
        "h_channel_note":       channel_note,
        "paper_update":         paper_update,
    }

    # 8. Plots
    _maybe_plot_turn_series(per_condition_summary, conditions)
    _maybe_plot_dimension_bars(per_condition_summary, conditions)
    _plot_effect_size_ladder(ladder)
    context_by_cond = {
        cond: load_condition_context(cond, profile=profile) for cond in conditions
    }
    _maybe_plot_vector_norms(context_by_cond)

    summary_path = RESULTS_DIR / "analysis_data.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\n  Summary written to {summary_path}", flush=True)
    print(f"\n  Decision-rule outcome:\n    {decision_rule}", flush=True)
    if channel_note:
        print(f"  H_channel note: {channel_note}", flush=True)
    return summary


def _classify_channel_interaction(
    mean_a: float | None,
    mean_b: float | None,
    mean_c: float | None,
) -> str:
    """Classify C vs max(A,B) and C vs min(A,B) per plan §6.3."""
    if mean_a is None or mean_b is None or mean_c is None:
        return "insufficient_data"
    max_ab = max(mean_a, mean_b)
    min_ab = min(mean_a, mean_b)
    if mean_c > max_ab + H_CHANNEL_ADDITIVE_MARGIN:
        return "Additive"
    elif mean_c < min_ab - H_CHANNEL_ADDITIVE_MARGIN:
        return "Interfering"
    elif abs(mean_c - mean_a) < H_CHANNEL_DOMINANT_MARGIN:
        return "Dominant_A"
    elif abs(mean_c - mean_b) < H_CHANNEL_DOMINANT_MARGIN:
        return "Dominant_B"
    return "Ambiguous"


# ── Plots ─────────────────────────────────────────────────────────────────

_COND_COLOURS = {
    "A": "#1f77b4",
    "B": "#2ca02c",
    "C": "#d62728",
    "D": "#9467bd",
}


def _maybe_plot_turn_series(summary: dict, conditions: list[str]):
    if not all(summary.get(c, {}).get("by_turn") for c in conditions):
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cond in conditions:
        by_turn = summary.get(cond, {}).get("by_turn", {})
        if not by_turn:
            continue
        turns = sorted(int(t) for t in by_turn)
        means = [by_turn[t] for t in turns]
        ax.plot(turns, means, marker="o", linewidth=2.0,
                color=_COND_COLOURS.get(cond, "black"),
                label=f"Condition {cond} (mean={summary[cond]['mean']:.3f})")
    for rt in (15, 30):
        ax.axvline(rt, color="grey", linestyle=":", alpha=0.5)
    ax.axhline(EXP4_COND_A_MEAN, color="black", linestyle="--", alpha=0.4,
               label=f"Exp 4 Cond A ({EXP4_COND_A_MEAN})")
    ax.set_xlabel("Probe turn")
    ax.set_ylabel("PersonaScore")
    ax.set_title("Experiment 5 — PersonaScore by probe turn × condition")
    ax.set_xticks([5, 10, 15, 20, 25, 30, 35, 40])
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    savefig(fig, "exp5_turn_series_psychotherapy", RESULTS_DIR)


def _maybe_plot_dimension_bars(summary: dict, conditions: list[str]):
    if not all(summary.get(c, {}).get("by_dim") for c in conditions):
        return
    dims = list(A.DIMENSIONS)
    x = np.arange(len(dims))
    width = 0.8 / max(len(conditions), 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k, cond in enumerate(conditions):
        by_dim = summary.get(cond, {}).get("by_dim", {})
        if not by_dim:
            continue
        means = [by_dim.get(d, 0) for d in dims]
        ax.bar(x + k * width, means, width=width * 0.95,
               color=_COND_COLOURS.get(cond, "grey"),
               label=f"Condition {cond}")
    ax.set_xticks(x + (len(conditions) - 1) * width / 2)
    ax.set_xticklabels(dims)
    ax.set_ylabel("Mean PersonaScore")
    ax.set_title("Experiment 5 — by-dimension PersonaScore × condition")
    ax.set_ylim(3.5, 5.0)
    ax.axhline(EXP4_COND_A_MEAN, color="black", linestyle="--", alpha=0.4,
               label=f"Exp 4 Cond A ({EXP4_COND_A_MEAN})")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, "exp5_dimension_bars_psychotherapy", RESULTS_DIR)


def _plot_effect_size_ladder(ladder: list[dict]):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [f"{r['experiment']}\n{r['metric']}" for r in ladder]
    values = [r["d"] for r in ladder]
    pos = np.arange(len(labels))
    abs_vals = [max(abs(v) if v is not None else 1e-4, 1e-4) for v in values]
    colours = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
    bars = ax.barh(pos, abs_vals,
                   color=colours[:len(labels)], alpha=0.9)
    ax.set_xscale("log")
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Cohen's d (log scale)")
    ax.set_title("CA program effect-size ladder — QPM internal vs downstream")
    for bar, val in zip(bars, values):
        label = f"d = {val:.3f}" if val is not None else "TBD"
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9)
    ax.axvline(H_LOGITS_D, color="grey", linestyle="--", alpha=0.6,
               label=f"d = {H_LOGITS_D} (H_logits threshold)")
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, which="both")
    savefig(fig, "exp5_effect_size_ladder", RESULTS_DIR)


def _maybe_plot_vector_norms(context_by_cond: dict[str, list[dict]]):
    """Composite vector norm by probe turn for conditions B, C, D (plan §8.5)."""
    steering_conds = [c for c in ("B", "C", "D") if context_by_cond.get(c)]
    if not steering_conds:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cond in steering_conds:
        rows = context_by_cond[cond]
        by_turn: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            norm = r.get("composite_vector_norm")
            t = r.get("turn")
            if norm is not None and t is not None:
                by_turn[int(t)].append(float(norm))
        if not by_turn:
            continue
        turns = sorted(by_turn)
        means = [float(np.mean(by_turn[t])) for t in turns]
        ax.plot(turns, means, marker="o", linewidth=2.0,
                color=_COND_COLOURS.get(cond, "black"),
                label=f"Condition {cond}")

    ax.set_xlabel("Conversation turn")
    ax.set_ylabel("‖v_composite‖")
    ax.set_title("Experiment 5 — composite steering vector norm by turn × condition\n"
                 "(near-constant norms suggest QPM marginals not varying; plan §8.5 diagnostic)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    savefig(fig, "exp5_vector_norms", RESULTS_DIR)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CA Experiment 5 analysis")
    p.add_argument("--conditions", type=str, default="A,B,C,D")
    p.add_argument("--profile",    type=str, default="psychotherapy")
    args = p.parse_args()
    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    analyse(conditions, profile=args.profile)


if __name__ == "__main__":
    main()
