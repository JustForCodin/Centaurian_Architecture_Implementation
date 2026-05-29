#!/usr/bin/env python3
"""
Analysis for CA Experiment 4: QPM→SLM Interface Richness Ablation.

Reads per-condition JSONL logs from experiment_runner.py and produces:

  • Per-condition mean PersonaScore + by-dimension means (T/E/C/S)
  • Paired t-tests B vs A, C vs A, D vs A at (script, turn, dimension)
  • Continuity check vs Experiment 3 QPM mean (4.410)
  • Turn-level PersonaScore plot (4 conditions × 8 probe turns)
  • Dimension breakdown plot (4 conditions × 4 dimensions)
  • Effect-size ladder plot (Exp 3 H1/H2/H3 anchors + Exp 4 best)
  • Decision-rule outcome JSON (plan §6.5)

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

BASE_DIR     = Path(__file__).parent
LOGS_BASE    = BASE_DIR / "logs"
RESULTS_DIR  = BASE_DIR / "results"

# ── Experiment 3 anchors (frozen reference values) ───────────────────────
# Source: CA_Experiment_3/EXPERIMENT_REPORT.md (psychotherapy profile)

EXP3_QPM_MEAN_PERSONA  = 4.410   # Cond-A continuity check target (±0.05)
EXP3_CMG_MEAN_PERSONA  = 4.387
EXP3_H1_COHENS_D       = 21.51
EXP3_H2_COHENS_D       = 2.59
EXP3_H3_COHENS_D       = 0.032

CONTINUITY_TOLERANCE = 0.05      # plan §6.1

# Decision-rule effect-size threshold (plan §3 / §6.5)
H_INTERFACE_D_THRESHOLD = 0.20
H_INTERFACE_P_THRESHOLD = 0.05


# ── Shared helpers ────────────────────────────────────────────────────────

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


def load_condition_scores(condition: str,
                          profile: str = "psychotherapy") -> list[dict]:
    """Return all scored probes for a given condition."""
    cond_dir = LOGS_BASE / f"condition_{condition.lower()}_{profile}"
    if not cond_dir.exists():
        return []
    rows = []
    for path in sorted(cond_dir.glob(f"scores_condition_{condition.lower()}_*.jsonl")):
        rows.extend(load_jsonl(path))
    return rows


def load_condition_context(condition: str,
                           profile: str = "psychotherapy") -> list[dict]:
    """Return per-turn context-log records for a condition (one row per turn).

    Context rows carry ambivalence + c_modifier for every conversational turn
    (not just probe turns), which is what we want for ambivalence histograms
    and C firing-rate plots.
    """
    cond_dir = LOGS_BASE / f"condition_{condition.lower()}_{profile}"
    if not cond_dir.exists():
        return []
    rows = []
    for path in sorted(cond_dir.glob(f"context_condition_{condition.lower()}_*.jsonl")):
        rows.extend(load_jsonl(path))
    return rows


def paired_ttest(a: list[float], b: list[float]) -> dict:
    """Paired t-test with Cohen's d_z and 95 % CI on the mean difference."""
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
        "n":          n,
        "mean_a":     round(float(np.mean(a_arr)), 4),
        "mean_b":     round(float(np.mean(b_arr)), 4),
        "mean_diff":  round(float(np.mean(diffs)), 4),
        "ci_95":      [round(float(ci_lo), 4), round(float(ci_hi), 4)],
        "t":          round(float(t_stat), 4),
        "p":          float(p_val),
        "cohens_d":   round(d_z, 4),
        "sd_diff":    round(sd, 4),
    }


def savefig(fig, stem: str, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(results_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _score_key(rec: dict) -> tuple:
    """Pairing key for matched t-tests: (script_id, turn, dimension)."""
    return (rec["script_id"], rec["turn"], rec["dimension"])


# ── Main analysis ────────────────────────────────────────────────────────

def analyse(conditions: list[str], profile: str = "psychotherapy") -> dict:
    print(f"\n=== Experiment 4 analysis — profile: {profile} ===", flush=True)

    # 1.  Load scores by condition --------------------------------------
    data: dict[str, dict[tuple, float]] = {}     # condition -> key -> score
    for cond in conditions:
        rows = load_condition_scores(cond, profile=profile)
        per_key: dict[tuple, float] = {}
        for rec in rows:
            if "score" not in rec:
                continue
            per_key[_score_key(rec)] = float(rec["score"])
        data[cond] = per_key
        print(f"  Condition {cond}: {len(per_key):>4} scored probes", flush=True)

    # 2.  Per-condition means + by-dimension means ----------------------
    per_condition_summary: dict[str, dict] = {}
    for cond, per_key in data.items():
        all_scores = list(per_key.values())
        if not all_scores:
            per_condition_summary[cond] = {"n": 0}
            continue
        per_dim = defaultdict(list)
        for (sid, turn, dim), score in per_key.items():
            per_dim[dim].append(score)
        per_turn = defaultdict(list)
        for (sid, turn, dim), score in per_key.items():
            per_turn[turn].append(score)
        per_condition_summary[cond] = {
            "n":          len(all_scores),
            "mean":       round(float(np.mean(all_scores)), 4),
            "sd":         round(float(np.std(all_scores, ddof=1)), 4),
            "by_dim":     {d: round(float(np.mean(v)), 4)
                           for d, v in sorted(per_dim.items())},
            "by_turn":    {int(t): round(float(np.mean(v)), 4)
                           for t, v in sorted(per_turn.items())},
        }

    # 3.  Continuity check (plan §6.1) ----------------------------------
    cond_a_mean = per_condition_summary.get("A", {}).get("mean")
    continuity = None
    if cond_a_mean is not None:
        deviation = round(cond_a_mean - EXP3_QPM_MEAN_PERSONA, 4)
        continuity = {
            "exp3_qpm_mean":        EXP3_QPM_MEAN_PERSONA,
            "exp4_cond_a_mean":     cond_a_mean,
            "deviation":            deviation,
            "tolerance":            CONTINUITY_TOLERANCE,
            "within_tolerance":     abs(deviation) <= CONTINUITY_TOLERANCE,
        }
        verdict = "OK" if continuity["within_tolerance"] else "FAIL — investigate confound"
        print(f"\n  Continuity check: Cond A mean = {cond_a_mean:.4f}, "
              f"Exp 3 QPM = {EXP3_QPM_MEAN_PERSONA}, "
              f"Δ = {deviation:+.4f} → {verdict}", flush=True)

    # 4.  Paired t-tests B/C/D vs A (H_interface) -----------------------
    pairwise: dict[str, dict] = {}
    if "A" in data:
        key_a = data["A"]
        for cond in ("B", "C", "D"):
            if cond not in data:
                continue
            key_x = data[cond]
            shared_keys = sorted(set(key_a) & set(key_x))
            a_scores = [key_a[k] for k in shared_keys]
            x_scores = [key_x[k] for k in shared_keys]
            # Paired t-test: x − a (positive d_z = condition improved over A)
            res = paired_ttest(x_scores, a_scores)
            res["passes"] = (
                res.get("p", 1.0) < H_INTERFACE_P_THRESHOLD
                and res.get("cohens_d", 0.0) >= H_INTERFACE_D_THRESHOLD
            )
            pairwise[cond] = res
            sign = "+" if res.get("cohens_d", 0) > 0 else ""
            print(f"\n  Paired t-test  Condition {cond} vs A   (n={res['n']})")
            print(f"    mean_{cond}={res['mean_a']}  mean_A={res['mean_b']}  "
                  f"Δ={res['mean_diff']:+}")
            print(f"    t={res['t']}  p={res['p']:.4g}  d_z={sign}{res['cohens_d']}  "
                  f"95%CI {res['ci_95']}")
            print(f"    H_interface pass? {res['passes']}")

    # 5.  Hypothesis-level summary --------------------------------------
    h_interface_pass = any(p.get("passes") for p in pairwise.values())
    h_c_wins = False
    if pairwise.get("C") and per_condition_summary:
        means = {c: per_condition_summary.get(c, {}).get("mean")
                 for c in ("A", "B", "C", "D")}
        means = {c: m for c, m in means.items() if m is not None}
        if means and max(means, key=means.get) == "C":
            h_c_wins = True

    # H_capability: largest condition-vs-A improvement on dim C (Capability)
    h_capability_pass = False
    dim_deltas: dict[str, dict[str, float]] = {}
    if "A" in per_condition_summary:
        a_by_dim = per_condition_summary["A"].get("by_dim", {})
        for cond, summary in per_condition_summary.items():
            if cond == "A":
                continue
            x_by_dim = summary.get("by_dim", {})
            dim_deltas[cond] = {
                d: round(x_by_dim.get(d, 0) - a_by_dim.get(d, 0), 4)
                for d in sorted(set(a_by_dim) | set(x_by_dim))
            }
        # Per-condition: which dimension improved most?  Did C dim win?
        c_wins_dim = 0
        considered = 0
        for cond, deltas in dim_deltas.items():
            if not deltas:
                continue
            considered += 1
            top_dim = max(deltas, key=deltas.get)
            if top_dim == "C":
                c_wins_dim += 1
        h_capability_pass = considered > 0 and c_wins_dim == considered

    # H_purity_episodic: B and D show E-dimension gain that A and C do not
    h_purity_episodic_pass = False
    if {"B", "D"}.issubset(dim_deltas):
        bd_e_gain = (dim_deltas["B"].get("E", 0) > 0
                     and dim_deltas["D"].get("E", 0) > 0)
        c_e_gain = dim_deltas.get("C", {}).get("E", 0) > 0
        h_purity_episodic_pass = bd_e_gain and not c_e_gain

    # 6.  Effect-size ladder (plan §6.4) --------------------------------
    best_d = 0.0
    best_cond = None
    for cond, res in pairwise.items():
        d = res.get("cohens_d", 0.0)
        if d > best_d:
            best_d = d
            best_cond = cond
    ladder = [
        {"experiment": "Exp 3 H1", "metric": "Order effects (JSD QPM vs CMG)",     "d": EXP3_H1_COHENS_D},
        {"experiment": "Exp 3 H2", "metric": "Ambivalence (entropy QPM vs CMG)",   "d": EXP3_H2_COHENS_D},
        {"experiment": "Exp 3 H3", "metric": "PersonaScore (QPM vs CMG, A)",       "d": EXP3_H3_COHENS_D},
        {"experiment": "Exp 4 best",
         "metric": f"PersonaScore ({best_cond or '—'} vs A)",
         "d": round(best_d, 4)},
    ]

    # 7.  Decision-rule outcome (plan §6.5) ------------------------------
    if continuity and not continuity["within_tolerance"]:
        decision_rule = (
            "Condition A deviates > ±0.05 from Exp 3 — pause, investigate "
            "confound before interpreting any condition comparisons."
        )
        paper_update = "Methodological finding only — no Section 5.7 update yet."
    elif h_interface_pass and h_c_wins:
        decision_rule = (
            "H_interface PASSED with Condition C winning — "
            "coherence-conditional speech-act is the mechanism."
        )
        paper_update = (
            "Formalise Condition C as the production QPM→SLM interface in "
            "Section 5.7.3.  Add Experiment 4 as Section 15.4.4.  Update "
            "Section 18 conclusions to note that downstream behavioural "
            "advantage is detectable under the coherence-conditional "
            "interface."
        )
    elif h_interface_pass:
        decision_rule = (
            f"H_interface PASSED — best condition: {best_cond} "
            f"(d_z={best_d:.4f}).  Numeric / coactivation exposure sufficient."
        )
        paper_update = (
            f"Formalise Condition {best_cond} as the production QPM→SLM "
            f"interface.  Add Experiment 4 as Section 15.4.4.  Update "
            f"Section 18 conclusions."
        )
    else:
        decision_rule = (
            "H_interface FAILED — no enriched interface reaches d_z ≥ 0.2.  "
            "JSON-mediated interface appears unable to transmit QPM advantage "
            "at 7B model scale."
        )
        paper_update = (
            "Strengthen Section 2.3 scope note.  Add Section 5.7.5 "
            "'Interface Limitations'.  Add logits-level QPM→SLM steering as "
            "Experiment 5 in Section 18 Future Work."
        )

    summary = {
        "profile":               profile,
        "conditions":            conditions,
        "per_condition":         per_condition_summary,
        "continuity_check":      continuity,
        "pairwise_vs_A":         pairwise,
        "dimension_deltas":      dim_deltas,
        "hypotheses": {
            "H_interface":        h_interface_pass,
            "H_C_wins":           h_c_wins,
            "H_capability":       h_capability_pass,
            "H_purity_episodic":  h_purity_episodic_pass,
        },
        "best_condition":        best_cond,
        "best_cohens_d":         round(best_d, 4),
        "effect_size_ladder":    ladder,
        "decision_rule_outcome": decision_rule,
        "paper_update":          paper_update,
    }

    # 8.  Plots --------------------------------------------------------
    if all(per_condition_summary.get(c, {}).get("by_turn") for c in conditions):
        _plot_turn_series(per_condition_summary, conditions)
    if all(per_condition_summary.get(c, {}).get("by_dim") for c in conditions):
        _plot_dimension_bars(per_condition_summary, conditions)
    _plot_effect_size_ladder(ladder)

    # Per-turn ambivalence + Condition C firing rate — depend on context logs
    context_by_cond: dict[str, list[dict]] = {
        cond: load_condition_context(cond, profile=profile) for cond in conditions
    }
    if any(context_by_cond.values()):
        _plot_ambivalence_distribution(context_by_cond)
    if context_by_cond.get("C"):
        _plot_condition_c_firing_rate(context_by_cond["C"])
        summary["c_firing_rate"] = _condition_c_firing_summary(context_by_cond["C"])

    summary_path = RESULTS_DIR / "analysis_data.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n  Summary written to {summary_path}", flush=True)
    print(f"\n  Decision-rule outcome:\n    {decision_rule}", flush=True)
    return summary


# ── Plots ─────────────────────────────────────────────────────────────────

_COND_COLOURS = {
    "A": "#1f77b4",
    "B": "#2ca02c",
    "C": "#d62728",
    "D": "#9467bd",
}


def _plot_turn_series(summary: dict, conditions: list[str]):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cond in conditions:
        by_turn = summary.get(cond, {}).get("by_turn", {})
        if not by_turn:
            continue
        turns = sorted(int(t) for t in by_turn.keys())
        means = [by_turn[t] for t in turns]
        ax.plot(turns, means, marker="o", linewidth=2.0,
                color=_COND_COLOURS.get(cond, "black"),
                label=f"Condition {cond} (mean={summary[cond]['mean']:.3f})")
    for refresh_t in (15, 30):
        ax.axvline(refresh_t, color="grey", linestyle=":", alpha=0.5)
    ax.axhline(EXP3_QPM_MEAN_PERSONA, color="black", linestyle="--", alpha=0.4,
               label=f"Exp 3 QPM mean ({EXP3_QPM_MEAN_PERSONA})")
    ax.set_xlabel("Probe turn")
    ax.set_ylabel("PersonaScore")
    ax.set_title("Experiment 4 — PersonaScore by probe turn × condition")
    ax.set_xticks([5, 10, 15, 20, 25, 30, 35, 40])
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    savefig(fig, "exp4_turn_series_psychotherapy", RESULTS_DIR)


def _plot_dimension_bars(summary: dict, conditions: list[str]):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    dims = list(A.DIMENSIONS)
    x = np.arange(len(dims))
    width = 0.8 / max(len(conditions), 1)
    for k, cond in enumerate(conditions):
        by_dim = summary.get(cond, {}).get("by_dim", {})
        if not by_dim:
            continue
        means = [by_dim.get(d, 0) for d in dims]
        ax.bar(x + k * width, means, width=width * 0.95,
               color=_COND_COLOURS.get(cond, "grey"),
               label=f"Condition {cond}")
    ax.set_xticks(x + (len(conditions) - 1) * width / 2)
    ax.set_xticklabels([f"{d}" for d in dims])
    ax.set_ylabel("Mean PersonaScore")
    ax.set_title("Experiment 4 — by-dimension PersonaScore × condition")
    ax.set_ylim(3.5, 5.0)
    ax.axhline(EXP3_QPM_MEAN_PERSONA, color="black", linestyle="--", alpha=0.4,
               label=f"Exp 3 QPM mean ({EXP3_QPM_MEAN_PERSONA})")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, "exp4_dimension_bars_psychotherapy", RESULTS_DIR)


def _plot_ambivalence_distribution(context_by_cond: dict[str, list[dict]]):
    """Histogram of per-turn ambivalence across all Battery C turns, overlaid
    by condition, with the Condition C decision thresholds drawn as vlines.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0.0, 1.0, 31)  # 30 bins of width ~0.033
    plotted = False
    for cond, rows in context_by_cond.items():
        amb = [r.get("ambivalence") for r in rows if r.get("ambivalence") is not None]
        if not amb:
            continue
        ax.hist(
            amb, bins=bins, histtype="step", linewidth=2.0,
            color=_COND_COLOURS.get(cond, "black"),
            label=f"Condition {cond} (n={len(amb)}, mean={np.mean(amb):.3f})",
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return

    # Condition C decision thresholds (plan Appendix).  Shade the three
    # behavioural bands and label them inside the panel; the per-condition
    # legend is moved outside the axes to avoid covering the right-hand
    # 'expressed uncertainty' band label.
    ymax_now = ax.get_ylim()[1]
    ax.set_ylim(0, ymax_now * 1.18)
    label_y = ymax_now * 1.08
    ax.axvspan(0.00, 0.15, color="#9ecae1", alpha=0.10)
    ax.axvspan(0.15, 0.45, color="#bdbdbd", alpha=0.08)
    ax.axvspan(0.45, 1.00, color="#fdae6b", alpha=0.12)
    ax.axvline(0.15, color="grey", linestyle="--", alpha=0.7)
    ax.axvline(0.45, color="grey", linestyle="--", alpha=0.7)
    ax.text(0.075, label_y, "C: 'grounded'",
            ha="center", fontsize=9, color="dimgrey")
    ax.text(0.30, label_y, "C: no directive\n(moderate band)",
            ha="center", fontsize=9, color="dimgrey")
    ax.text(0.725, label_y,
            "C: 'with expressed\nuncertainty'",
            ha="center", fontsize=9, color="dimgrey")

    ax.set_xlabel("QPM ambivalence (= 1 − purity_proxy)")
    ax.set_ylabel("Turns")
    ax.set_title("Experiment 4 — per-turn ambivalence distribution, "
                 "by condition")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    savefig(fig, "exp4_ambivalence_distribution", RESULTS_DIR)


def _plot_condition_c_firing_rate(c_context_rows: list[dict]):
    """Per-script bar chart: % of conversational turns where Condition C's
    speech-act modifier fired (i.e. c_modifier was non-null).
    """
    by_script: dict[int, list[int]] = defaultdict(list)
    for row in c_context_rows:
        sid = row.get("script_id")
        modifier = row.get("c_modifier")
        if sid is None:
            continue
        by_script[sid].append(1 if modifier else 0)
    if not by_script:
        return

    sids = sorted(by_script.keys())
    rates = [100.0 * np.mean(by_script[s]) for s in sids]
    overall_rate = 100.0 * np.mean(
        [v for vals in by_script.values() for v in vals]
    )

    fig, ax = plt.subplots(figsize=(11, 5.0))
    colours = [
        "#d62728" if s >= 81 else "#1f77b4"   # adversarial scripts red
        for s in sids
    ]
    ax.bar(range(len(sids)), rates, color=colours, edgecolor="white", linewidth=0.6)
    ax.axhline(overall_rate, color="black", linestyle="--", alpha=0.7,
               label=f"overall: {overall_rate:.1f}%")
    ax.set_xticks(range(len(sids)))
    ax.set_xticklabels([f"{s:03d}" for s in sids], rotation=60, fontsize=8)
    ax.set_xlabel("Script ID  (blue = naturalistic, red = adversarial)")
    ax.set_ylabel("% turns where speech-act modifier fired")
    ax.set_title("Experiment 4 — Condition C firing rate, by script")
    ax.set_ylim(0, max(max(rates), 1.0) * 1.15)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, "exp4_condition_c_firing_rate", RESULTS_DIR)


def _condition_c_firing_summary(c_context_rows: list[dict]) -> dict:
    """Aggregate Condition C firing statistics for the summary JSON."""
    by_script: dict[int, list[int]] = defaultdict(list)
    high = low = 0
    for row in c_context_rows:
        sid = row.get("script_id")
        modifier = row.get("c_modifier")
        if sid is not None:
            by_script[sid].append(1 if modifier else 0)
        if modifier == "with_expressed_uncertainty":
            high += 1
        elif modifier == "grounded":
            low += 1
    total_turns = sum(len(v) for v in by_script.values())
    total_fires = sum(sum(v) for v in by_script.values())
    return {
        "total_turns":            total_turns,
        "total_fires":            total_fires,
        "fire_rate":              round(total_fires / max(total_turns, 1), 4),
        "with_expressed_uncertainty": high,
        "grounded":               low,
        "per_script_fire_rate":   {
            int(s): round(float(np.mean(v)), 4) for s, v in sorted(by_script.items())
        },
    }


def _plot_effect_size_ladder(ladder: list[dict]):
    fig, ax = plt.subplots(figsize=(9, 5.0))
    labels = [f"{row['experiment']}\n{row['metric']}" for row in ladder]
    values = [row["d"] for row in ladder]
    # Log scale because H1 (~21) and H3 (~0.03) differ by ~3 orders of magnitude.
    pos = np.arange(len(labels))
    abs_values = [max(abs(v), 1e-4) for v in values]
    colours = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]
    bars = ax.barh(pos, abs_values, color=colours, alpha=0.9)
    ax.set_xscale("log")
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Cohen's d (log scale)")
    ax.set_title("CA program effect-size ladder — QPM internal vs downstream")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                f"d = {val:.3f}", va="center", fontsize=9)
    ax.axvline(H_INTERFACE_D_THRESHOLD, color="grey", linestyle="--", alpha=0.6,
               label=f"d = {H_INTERFACE_D_THRESHOLD} (decision threshold)")
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, which="both")
    savefig(fig, "exp4_effect_size_ladder", RESULTS_DIR)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CA Experiment 4 analysis")
    p.add_argument(
        "--conditions", type=str, default="A,B,C,D",
        help="Comma-separated list of conditions to analyse (default: A,B,C,D)",
    )
    p.add_argument(
        "--profile", type=str, default="psychotherapy",
        help="Personality profile name (default: psychotherapy)",
    )
    args = p.parse_args()
    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    analyse(conditions, profile=args.profile)


if __name__ == "__main__":
    main()
