#!/usr/bin/env python3
"""
Analysis for CA Experiment 3: QPM vs. CMG-CDK ablation.

Reads JSONL logs from experiment_runner.py and produces:
  Battery A: JSD distribution plots + paired t-test (H1)
  Battery B: Entropy comparison + coherence proxy validation (H2)
  Battery C: PersonaScore time series + paired t-test (H3)
  Decision rule outcome summary

Usage:
  python3 analyse_results.py --battery A --profile psychotherapy
  python3 analyse_results.py --battery B
  python3 analyse_results.py --battery C
  python3 analyse_results.py --all           # run all batteries + decision rules
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# Matplotlib with non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
LOGS_BASE = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# ── Shared helpers ────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def paired_ttest(a: list[float], b: list[float]) -> dict:
    """Paired t-test with Cohen's d_z."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    diffs = a_arr - b_arr
    n = len(diffs)
    t_stat, p_val = stats.ttest_rel(a_arr, b_arr)
    d_z = float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-12))
    ci_lo, ci_hi = stats.t.interval(0.95, df=n - 1,
                                     loc=np.mean(diffs),
                                     scale=stats.sem(diffs))
    return {
        "n": n,
        "mean_a": round(float(np.mean(a_arr)), 4),
        "mean_b": round(float(np.mean(b_arr)), 4),
        "mean_diff": round(float(np.mean(diffs)), 4),
        "ci_95": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
        "t": round(float(t_stat), 4),
        "p": float(p_val),
        "cohens_d": round(d_z, 4),
    }


def savefig(fig, stem: str, results_dir: Path):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(results_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Battery A analysis ────────────────────────────────────────────────────

def analyse_battery_a(profile: str, results_dir: Path) -> dict:
    log_path = LOGS_BASE / f"battery_a_{profile}" / "battery_a_results.jsonl"
    if not log_path.exists():
        print(f"Battery A log not found: {log_path}")
        return {}

    records = load_jsonl(log_path)
    if not records:
        print("Battery A: no records found.")
        return {}

    jsd_qpm = [r["jsd_qpm"] for r in records]
    jsd_cmg = [r["jsd_cmg"] for r in records]
    categories = [r["category"] for r in records]

    stats_out = paired_ttest(jsd_qpm, jsd_cmg)
    h1_pass = stats_out["p"] < 0.05 and stats_out["mean_a"] > stats_out["mean_b"]

    # Per-category means
    cat_means: dict[str, dict] = defaultdict(lambda: {"qpm": [], "cmg": []})
    for r in records:
        cat_means[r["category"]]["qpm"].append(r["jsd_qpm"])
        cat_means[r["category"]]["cmg"].append(r["jsd_cmg"])
    cat_summary = {
        cat: {
            "mean_jsd_qpm": round(float(np.mean(v["qpm"])), 4),
            "mean_jsd_cmg": round(float(np.mean(v["cmg"])), 4),
            "n": len(v["qpm"]),
        }
        for cat, v in cat_means.items()
    }

    # Plot: JSD distributions side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Battery A — Order Effects (H1) | profile={profile}", fontsize=13)

    ax = axes[0]
    ax.scatter(range(len(jsd_qpm)), sorted(jsd_qpm), label="QPM", color="#2196F3", s=40)
    ax.scatter(range(len(jsd_cmg)), sorted(jsd_cmg), label="CMG-CDK", color="#FF9800", s=40)
    ax.set_xlabel("Pair rank (sorted by QPM JSD)")
    ax.set_ylabel("JSD")
    ax.set_title(f"Sorted JSD per pair\np={stats_out['p']:.4f}  d={stats_out['cohens_d']:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    cat_names = list(cat_summary.keys())
    x = np.arange(len(cat_names))
    w = 0.35
    ax.bar(x - w/2, [cat_summary[c]["mean_jsd_qpm"] for c in cat_names],
           w, label="QPM", color="#2196F3")
    ax.bar(x + w/2, [cat_summary[c]["mean_jsd_cmg"] for c in cat_names],
           w, label="CMG-CDK", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in cat_names], fontsize=8)
    ax.set_ylabel("Mean JSD")
    ax.set_title("JSD by input category")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    savefig(fig, f"battery_a_{profile}", results_dir)

    result = {
        "battery": "A",
        "profile": profile,
        "hypothesis": "H1",
        "n_pairs": len(records),
        "mean_jsd_qpm": stats_out["mean_a"],
        "mean_jsd_cmg": stats_out["mean_b"],
        "paired_ttest": stats_out,
        "h1_pass": h1_pass,
        "per_category": cat_summary,
    }

    print(f"\n=== Battery A ({profile}) ===")
    print(f"  n pairs         : {len(records)}")
    print(f"  Mean JSD QPM    : {stats_out['mean_a']:.4f}")
    print(f"  Mean JSD CMG    : {stats_out['mean_b']:.4f}")
    print(f"  Mean diff       : {stats_out['mean_diff']:.4f}  95% CI {stats_out['ci_95']}")
    print(f"  t={stats_out['t']:.3f}  p={stats_out['p']:.4f}  d={stats_out['cohens_d']:.3f}")
    print(f"  H1 {'PASSED ✓' if h1_pass else 'FAILED ✗'}")
    return result


# ── Battery B analysis ────────────────────────────────────────────────────

def analyse_battery_b(profile: str, results_dir: Path) -> dict:
    log_path = LOGS_BASE / f"battery_b_{profile}" / "battery_b_results.jsonl"
    if not log_path.exists():
        print(f"Battery B log not found: {log_path}")
        return {}

    records = load_jsonl(log_path)
    if not records:
        print("Battery B: no records found.")
        return {}

    ent_qpm = [r["entropy_qpm"] for r in records]
    ent_cmg = [r["entropy_cmg"] for r in records]
    purity_qpm = [r["purity_approx_qpm"] for r in records]

    stats_out = paired_ttest(ent_qpm, ent_cmg)
    h2_pass = stats_out["p"] < 0.05 and stats_out["mean_a"] > stats_out["mean_b"]

    # Purity under conflict vs control (last 4 = task_constraint type)
    conflict_purity = [r["purity_approx_qpm"]
                       for r in records if r["scenario_type"] != "task_constraint"]
    control_purity  = [r["purity_approx_qpm"]
                       for r in records if r["scenario_type"] == "task_constraint"]

    # Per-type means
    type_means: dict[str, dict] = defaultdict(lambda: {"qpm": [], "cmg": [], "purity": []})
    for r in records:
        type_means[r["scenario_type"]]["qpm"].append(r["entropy_qpm"])
        type_means[r["scenario_type"]]["cmg"].append(r["entropy_cmg"])
        type_means[r["scenario_type"]]["purity"].append(r["purity_approx_qpm"])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Battery B — Ambivalence (H2) | profile={profile}", fontsize=13)

    ax = axes[0]
    scenario_types = [r["scenario_type"] for r in records]
    type_colors = {
        "warm_pressure": "#E91E63",
        "formal_distress": "#9C27B0",
        "engaged_ambiguity": "#2196F3",
        "calm_urgency": "#FF9800",
        "task_constraint": "#9E9E9E",
    }
    colors = [type_colors.get(t, "#333333") for t in scenario_types]
    ax.scatter(ent_qpm, ent_cmg, c=colors, s=60, alpha=0.8)
    lo = min(min(ent_qpm), min(ent_cmg)) - 0.002
    hi = max(max(ent_qpm), max(ent_cmg)) + 0.002
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="QPM = CMG")
    ax.set_xlabel("QPM entropy")
    ax.set_ylabel("CMG-CDK entropy")
    ax.set_title(f"Entropy QPM vs CMG-CDK\np={stats_out['p']:.4f}  d={stats_out['cohens_d']:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.boxplot(
        [conflict_purity, control_purity],
        labels=["Conflict scenarios\n(n=16)", "Control\n(n=4)"],
        patch_artist=True,
        boxprops={"facecolor": "#E3F2FD"},
    )
    ax.set_ylabel("QPM purity_approx (coherence proxy)")
    ax.set_title("Purity proxy: conflict vs control")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    savefig(fig, f"battery_b_{profile}", results_dir)

    result = {
        "battery": "B",
        "profile": profile,
        "hypothesis": "H2",
        "n_scenarios": len(records),
        "mean_entropy_qpm": stats_out["mean_a"],
        "mean_entropy_cmg": stats_out["mean_b"],
        "paired_ttest": stats_out,
        "h2_pass": h2_pass,
        "mean_purity_conflict": round(float(np.mean(conflict_purity)), 4) if conflict_purity else None,
        "mean_purity_control": round(float(np.mean(control_purity)), 4) if control_purity else None,
    }

    print(f"\n=== Battery B ({profile}) ===")
    print(f"  n scenarios       : {len(records)}")
    print(f"  Mean entropy QPM  : {stats_out['mean_a']:.4f}")
    print(f"  Mean entropy CMG  : {stats_out['mean_b']:.4f}")
    print(f"  t={stats_out['t']:.3f}  p={stats_out['p']:.4f}  d={stats_out['cohens_d']:.3f}")
    print(f"  Purity (conflict) : {result['mean_purity_conflict']}")
    print(f"  Purity (control)  : {result['mean_purity_control']}")
    print(f"  H2 {'PASSED ✓' if h2_pass else 'FAILED ✗'}")
    return result


# ── Battery C analysis ────────────────────────────────────────────────────

def _load_battery_c_scores(logs_dir: Path, model_label: str) -> list[dict]:
    records = []
    for p in sorted(logs_dir.glob(f"scores_{model_label}_*.jsonl")):
        records.extend(load_jsonl(p))
    return records


def analyse_battery_c(profile: str, results_dir: Path) -> dict:
    logs_dir = LOGS_BASE / f"battery_c_{profile}"
    if not logs_dir.exists():
        print(f"Battery C logs not found: {logs_dir}")
        return {}

    qpm_records = _load_battery_c_scores(logs_dir, "qpm")
    cmg_records = _load_battery_c_scores(logs_dir, "cmg")

    if not qpm_records or not cmg_records:
        print("Battery C: insufficient records.")
        return {}

    def mean_score(records):
        return float(np.mean([r["score"] for r in records]))

    def score_by_turn(records):
        turn_scores: dict[int, list] = defaultdict(list)
        for r in records:
            turn_scores[r["turn"]].append(r["score"])
        return {t: round(float(np.mean(v)), 3)
                for t, v in sorted(turn_scores.items())}

    def score_by_dim(records):
        dim_scores: dict[str, list] = defaultdict(list)
        for r in records:
            dim_scores[r["dimension"]].append(r["score"])
        return {d: round(float(np.mean(v)), 3) for d, v in dim_scores.items()}

    # Build paired observations: (qpm_score, cmg_score) per (script_id, turn, dim)
    qpm_idx = {(r["script_id"], r["turn"], r["dimension"]): r["score"]
               for r in qpm_records}
    cmg_idx = {(r["script_id"], r["turn"], r["dimension"]): r["score"]
               for r in cmg_records}
    common = sorted(set(qpm_idx) & set(cmg_idx))
    paired_qpm = [qpm_idx[k] for k in common]
    paired_cmg = [cmg_idx[k] for k in common]

    stats_out = paired_ttest(paired_qpm, paired_cmg)
    h3_pass = stats_out["p"] < 0.05 and stats_out["mean_a"] > stats_out["mean_b"]

    qpm_turns = score_by_turn(qpm_records)
    cmg_turns = score_by_turn(cmg_records)
    qpm_dims  = score_by_dim(qpm_records)
    cmg_dims  = score_by_dim(cmg_records)

    # Time-series plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Battery C — PersonaScore (H3) | profile={profile}", fontsize=13)

    ax = axes[0]
    turns = sorted(set(qpm_turns) | set(cmg_turns))
    ax.plot(turns, [qpm_turns.get(t, np.nan) for t in turns],
            "o-", label="QPM", color="#2196F3", linewidth=2)
    ax.plot(turns, [cmg_turns.get(t, np.nan) for t in turns],
            "s--", label="CMG-CDK", color="#FF9800", linewidth=2)
    ax.axhline(3.5, color="red", linestyle=":", linewidth=1.5, label="threshold 3.5")
    ax.set_xlabel("Probe turn")
    ax.set_ylabel("Mean PersonaScore")
    ax.set_title(
        f"PersonaScore over time\n"
        f"QPM={stats_out['mean_a']:.3f}  CMG={stats_out['mean_b']:.3f}  "
        f"p={stats_out['p']:.4f}"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    dims = ["T", "E", "C", "S"]
    x = np.arange(len(dims))
    w = 0.35
    ax.bar(x - w/2, [qpm_dims.get(d, 0) for d in dims], w, label="QPM", color="#2196F3")
    ax.bar(x + w/2, [cmg_dims.get(d, 0) for d in dims], w, label="CMG-CDK", color="#FF9800")
    ax.axhline(3.5, color="red", linestyle=":", linewidth=1.5, label="threshold 3.5")
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.set_ylabel("Mean PersonaScore")
    ax.set_title("PersonaScore by dimension")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    savefig(fig, f"battery_c_{profile}", results_dir)

    result = {
        "battery": "C",
        "profile": profile,
        "hypothesis": "H3",
        "n_paired": len(common),
        "mean_score_qpm": stats_out["mean_a"],
        "mean_score_cmg": stats_out["mean_b"],
        "paired_ttest": stats_out,
        "h3_pass": h3_pass,
        "by_turn_qpm": qpm_turns,
        "by_turn_cmg": cmg_turns,
        "by_dim_qpm": qpm_dims,
        "by_dim_cmg": cmg_dims,
    }

    print(f"\n=== Battery C ({profile}) ===")
    print(f"  n paired obs      : {len(common)}")
    print(f"  Mean PersonaScore QPM : {stats_out['mean_a']:.3f}")
    print(f"  Mean PersonaScore CMG : {stats_out['mean_b']:.3f}")
    print(f"  Δ                     : {stats_out['mean_diff']:+.3f}")
    print(f"  t={stats_out['t']:.3f}  p={stats_out['p']:.4f}  d={stats_out['cohens_d']:.3f}")
    print(f"  H3 {'PASSED ✓' if h3_pass else 'FAILED ✗'}")

    print("\n  Per-dimension:")
    for d in dims:
        print(f"    {d}: QPM={qpm_dims.get(d,'?')}  CMG={cmg_dims.get(d,'?')}")
    return result


# ── Decision-rule outcome ─────────────────────────────────────────────────

_DECISION_RULES = {
    (True,  True,  True):  (
        "H1✓ H2✓ H3✓",
        "QPM earns its keep on all three dimensions. "
        "Framing bulletproof — add Experiment 3 as Section 15.4.3, "
        "update Table 41 ablation with CMG-CDK as correct control.",
    ),
    (True,  True,  False): (
        "H1✓ H2✓ H3✗",
        "Quantum advantage confirmed at internal state level; "
        "downstream behavioural advantage not detected at current sample size. "
        "Add scope note to §2.3.",
    ),
    (True,  False, True):  (
        "H1✓ H2✗ H3✓",
        "Ambivalence advantage absent; order effects + behaviour hold. "
        "Revise §2.3 to de-emphasise ambivalence claim.",
    ),
    (False, True,  True):  (
        "H1✗ H2✓ H3✓",
        "Non-commutativity not distinguishable at n=30; "
        "ambivalence + behaviour advantages hold. "
        "Revise §2.3 to de-emphasise order-effect claim.",
    ),
    (False, False, True):  (
        "H1✗ H2✗ H3✓",
        "Classical model produces equivalent internal states but QPM behaviour is better. "
        "Investigate: QPM stochasticity may be helping PersonaScore incidentally. "
        "Redesign QPM to increase Tr(ρ²) divergence from classical.",
    ),
    (True,  False, False): (
        "H1✓ H2✗ H3✗",
        "Order effects present but do not propagate to ambivalence or behaviour. "
        "Scope claim to non-commutativity only.",
    ),
    (False, True,  False): (
        "H1✗ H2✓ H3✗",
        "Ambivalence present internally but does not improve behaviour. "
        "Scope claim to ambivalence representation only.",
    ),
    (False, False, False): (
        "H1✗ H2✗ H3✗",
        "Classical model matches QPM on all dimensions. "
        "Revise framing: QPM offers equivalent expressiveness with superior "
        "interpretability and auditability — not a performance claim.",
    ),
}


def decision_rule_outcome(
    h1: bool | None,
    h2: bool | None,
    h3: bool | None,
) -> str:
    if any(x is None for x in (h1, h2, h3)):
        return "Incomplete — some batteries not yet run."
    label, text = _DECISION_RULES.get((h1, h2, h3), ("Unknown", ""))
    return f"{label}\n→ {text}"


# ── Summary report ────────────────────────────────────────────────────────

def write_summary(
    results: dict,
    profile: str,
    results_dir: Path,
):
    a = results.get("A", {})
    b = results.get("B", {})
    c = results.get("C", {})

    h1 = a.get("h1_pass")
    h2 = b.get("h2_pass")
    h3 = c.get("h3_pass")

    outcome = decision_rule_outcome(h1, h2, h3)

    lines = [
        f"# CA Experiment 3 — Analysis Summary",
        f"**Profile:** {profile}",
        "",
        "## Hypothesis Results",
        "",
        f"| Hypothesis | Metric | QPM | CMG-CDK | p | d | Pass? |",
        f"|---|---|---|---|---|---|---|",
    ]
    if a:
        t = a["paired_ttest"]
        lines.append(
            f"| H1 (order effects) | JSD | {a['mean_jsd_qpm']:.4f} | "
            f"{a['mean_jsd_cmg']:.4f} | {t['p']:.4f} | {t['cohens_d']:.3f} | "
            f"{'✓' if h1 else '✗'} |"
        )
    if b:
        t = b["paired_ttest"]
        lines.append(
            f"| H2 (ambivalence) | Shannon H | {b['mean_entropy_qpm']:.4f} | "
            f"{b['mean_entropy_cmg']:.4f} | {t['p']:.4f} | {t['cohens_d']:.3f} | "
            f"{'✓' if h2 else '✗'} |"
        )
    if c:
        t = c["paired_ttest"]
        lines.append(
            f"| H3 (PersonaScore) | Mean score | {c['mean_score_qpm']:.3f} | "
            f"{c['mean_score_cmg']:.3f} | {t['p']:.4f} | {t['cohens_d']:.3f} | "
            f"{'✓' if h3 else '✗'} |"
        )

    lines += [
        "",
        "## Decision Rule Outcome",
        "",
        f"```",
        outcome,
        f"```",
        "",
    ]

    if b and b.get("mean_purity_conflict") is not None:
        lines += [
            "## QPM Coherence Proxy (Battery B)",
            "",
            f"- Mean purity_approx (conflict scenarios): {b['mean_purity_conflict']}",
            f"- Mean purity_approx (control scenarios):  {b['mean_purity_control']}",
            "",
        ]

    if c and c.get("by_dim_qpm"):
        lines += ["## PersonaScore by Dimension (Battery C)", ""]
        lines.append("| Dim | QPM | CMG-CDK | Δ |")
        lines.append("|---|---|---|---|")
        for d in ["T", "E", "C", "S"]:
            qpm_v = c["by_dim_qpm"].get(d, "?")
            cmg_v = c["by_dim_cmg"].get(d, "?")
            delta = (
                round(float(qpm_v) - float(cmg_v), 3)
                if isinstance(qpm_v, (int, float)) and isinstance(cmg_v, (int, float))
                else "?"
            )
            lines.append(f"| {d} | {qpm_v} | {cmg_v} | {delta:+} |")
        lines.append("")

    report_path = results_dir / "summary_report.md"
    report_path.write_text("\n".join(lines))
    print(f"\nSummary written to {report_path}")

    # Save JSON
    analysis_data = {
        "profile": profile,
        "battery_a": a,
        "battery_b": b,
        "battery_c": c,
        "decision_outcome": outcome,
    }
    (results_dir / "analysis_data.json").write_text(
        json.dumps(analysis_data, indent=2)
    )
    return report_path


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CA Experiment 3 Analysis")
    p.add_argument(
        "--battery", choices=["A", "B", "C", "all"], default="all",
        help="Which battery to analyse (default: all)",
    )
    p.add_argument(
        "--profile", default="psychotherapy",
        choices=["psychotherapy", "software_eng"],
    )
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    run_all = args.battery == "all"

    if run_all or args.battery == "A":
        results["A"] = analyse_battery_a(args.profile, RESULTS_DIR)
    if run_all or args.battery == "B":
        results["B"] = analyse_battery_b(args.profile, RESULTS_DIR)
    if run_all or args.battery == "C":
        results["C"] = analyse_battery_c(args.profile, RESULTS_DIR)

    if run_all:
        write_summary(results, args.profile, RESULTS_DIR)
        h1 = results.get("A", {}).get("h1_pass")
        h2 = results.get("B", {}).get("h2_pass")
        h3 = results.get("C", {}).get("h3_pass")
        print(f"\n{'='*60}")
        print("DECISION RULE OUTCOME:")
        print(decision_rule_outcome(h1, h2, h3))
        print("=" * 60)


if __name__ == "__main__":
    main()
