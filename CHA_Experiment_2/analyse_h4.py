#!/usr/bin/env python3
"""
Analysis for the CHA H4 base-capability test.

Reads logs/h4_base/responses.jsonl and logs/h4_lora/responses.jsonl,
pairs them by probe_id, and reports whether the LoRA-10K adapter
causes catastrophic forgetting on out-of-domain capability.

Per plan §5.3:
  PASS if the LoRA condition is within 5% of the base condition's
  mean score (i.e., (base - lora) / base < 0.05).

Statistical tests:
  - Paired t-test (lora score vs base score, paired by probe_id).
  - Cohen's d_z for paired samples.
  - Per-category verdicts using the same 5% rule.

Outputs to results/:
  h4_summary_report.md      - text summary + verdict
  h4_analysis_data.json     - raw stats for downstream figures
  h4_category_comparison.png- bar chart, base vs lora per category
  h4_score_distribution.png - score histograms per condition
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


BASE_DIR = Path(__file__).parent
LOGS_BASE = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
PROBES_PATH = BASE_DIR / "h4_probes.json"

CONDITIONS = ("base", "lora")
CONDITION_LABELS = {
    "base": "Base Qwen2.5-7B",
    "lora": "Base + LoRA-10K",
}
CONDITION_COLORS = {
    "base": "#9e9e9e",
    "lora": "#4caf50",
}

CATEGORIES = (
    "general_knowledge",
    "code_reasoning",
    "math",
    "instruction_following",
    "structured_intent",
)
CATEGORY_LABELS = {
    "general_knowledge": "General Knowledge",
    "code_reasoning": "Code Reasoning",
    "math": "Math",
    "instruction_following": "Instruction Following",
    "structured_intent": "Structured Intent JSON",
}

DEGRADATION_THRESHOLD_PCT = 5.0


# ── Data loading ─────────────────────────────────────────────────────────

def load_condition(cond: str) -> dict[str, dict]:
    """Return {probe_id: record} for one condition."""
    path = LOGS_BASE / f"h4_{cond}" / "responses.jsonl"
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("probe_id")
            if pid is None:
                continue
            out[pid] = rec
    return out


# ── Stats helpers ────────────────────────────────────────────────────────

def cohen_d_paired(diffs: np.ndarray) -> float:
    """Paired Cohen's d_z = mean(diff) / sd(diff)."""
    if len(diffs) < 2 or float(diffs.std(ddof=1)) == 0.0:
        return 0.0
    return float(diffs.mean() / diffs.std(ddof=1))


def degradation_pct(base_mean: float, lora_mean: float) -> float:
    """Positive number = LoRA worse than base. Can be negative (improvement)."""
    if base_mean == 0:
        return 0.0
    return 100.0 * (base_mean - lora_mean) / base_mean


def verdict_label(deg_pct: float) -> str:
    if deg_pct < DEGRADATION_THRESHOLD_PCT:
        return "PASS"
    return "FAIL"


# ── Main analysis ────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    base = load_condition("base")
    lora = load_condition("lora")

    if not base or not lora:
        print("Missing logs for one or both conditions. Run run_h4.py first.",
              file=sys.stderr)
        print(f"  logs/h4_base/responses.jsonl  : {len(base)} rows")
        print(f"  logs/h4_lora/responses.jsonl  : {len(lora)} rows")
        sys.exit(1)

    paired_ids = sorted(set(base.keys()) & set(lora.keys()))
    if not paired_ids:
        print("No overlapping probe_ids between conditions.", file=sys.stderr)
        sys.exit(1)

    base_scores = np.array([base[pid]["score"] for pid in paired_ids], dtype=float)
    lora_scores = np.array([lora[pid]["score"] for pid in paired_ids], dtype=float)
    diffs = lora_scores - base_scores

    base_mean, base_std = float(base_scores.mean()), float(base_scores.std(ddof=1))
    lora_mean, lora_std = float(lora_scores.mean()), float(lora_scores.std(ddof=1))

    t_stat, p_value = stats.ttest_rel(lora_scores, base_scores)
    t_stat, p_value = float(t_stat), float(p_value)
    cohens_d = cohen_d_paired(diffs)
    overall_deg = degradation_pct(base_mean, lora_mean)
    overall_verdict = verdict_label(overall_deg)

    # Per-category breakdown
    per_cat = {}
    cat_to_pids: dict[str, list[str]] = defaultdict(list)
    for pid in paired_ids:
        cat = base[pid]["category"]
        cat_to_pids[cat].append(pid)

    for cat in CATEGORIES:
        pids = cat_to_pids.get(cat, [])
        if not pids:
            per_cat[cat] = None
            continue
        b = np.array([base[p]["score"] for p in pids], dtype=float)
        l = np.array([lora[p]["score"] for p in pids], dtype=float)
        d = l - b
        b_mean, l_mean = float(b.mean()), float(l.mean())
        deg = degradation_pct(b_mean, l_mean)
        if len(pids) >= 2 and float(b.std(ddof=1)) > 0 or float(l.std(ddof=1)) > 0:
            ts, pv = stats.ttest_rel(l, b)
            ts, pv = float(ts), float(pv)
        else:
            ts, pv = 0.0, 1.0
        per_cat[cat] = {
            "n": len(pids),
            "base_mean": b_mean,
            "base_std": float(b.std(ddof=1)) if len(pids) > 1 else 0.0,
            "lora_mean": l_mean,
            "lora_std": float(l.std(ddof=1)) if len(pids) > 1 else 0.0,
            "delta_mean": float(d.mean()),
            "degradation_pct": deg,
            "verdict": verdict_label(deg),
            "t_stat": ts,
            "p_value": pv,
            "cohens_d_paired": cohen_d_paired(d),
        }

    # Score-bucket distributions per condition
    score_bins = [1, 2, 3, 4, 5]
    base_hist = {s: int((base_scores == s).sum()) for s in score_bins}
    lora_hist = {s: int((lora_scores == s).sum()) for s in score_bins}

    # Identify regressions: probes where LoRA scored materially lower
    regressions = []
    for pid in paired_ids:
        diff = lora[pid]["score"] - base[pid]["score"]
        if diff <= -2:
            regressions.append({
                "probe_id": pid,
                "category": base[pid]["category"],
                "base_score": base[pid]["score"],
                "lora_score": lora[pid]["score"],
                "prompt": base[pid]["prompt"][:120],
                "base_response": base[pid]["response"][:200],
                "lora_response": lora[pid]["response"][:200],
            })

    # ── Save JSON ────────────────────────────────────────────────────────
    payload = {
        "n_probes": len(paired_ids),
        "overall": {
            "base_mean": base_mean,
            "base_std": base_std,
            "lora_mean": lora_mean,
            "lora_std": lora_std,
            "delta_mean": float(diffs.mean()),
            "degradation_pct": overall_deg,
            "threshold_pct": DEGRADATION_THRESHOLD_PCT,
            "verdict": overall_verdict,
            "t_stat_paired": t_stat,
            "p_value_paired": p_value,
            "cohens_d_paired": cohens_d,
        },
        "per_category": per_cat,
        "score_histograms": {
            "base": base_hist,
            "lora": lora_hist,
        },
        "regressions_2plus": regressions,
    }
    (RESULTS_DIR / "h4_analysis_data.json").write_text(
        json.dumps(payload, indent=2)
    )

    # ── Plot: per-category bar chart ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cats_present = [c for c in CATEGORIES if per_cat.get(c)]
    x = np.arange(len(cats_present))
    w = 0.36

    b_means = [per_cat[c]["base_mean"] for c in cats_present]
    l_means = [per_cat[c]["lora_mean"] for c in cats_present]
    b_stds = [per_cat[c]["base_std"] for c in cats_present]
    l_stds = [per_cat[c]["lora_std"] for c in cats_present]

    ax.bar(x - w/2, b_means, w, yerr=b_stds, capsize=4,
           color=CONDITION_COLORS["base"], label=CONDITION_LABELS["base"])
    ax.bar(x + w/2, l_means, w, yerr=l_stds, capsize=4,
           color=CONDITION_COLORS["lora"], label=CONDITION_LABELS["lora"])

    # Degradation labels
    for i, c in enumerate(cats_present):
        deg = per_cat[c]["degradation_pct"]
        sign = "-" if deg > 0 else "+"
        color = "tab:red" if deg >= DEGRADATION_THRESHOLD_PCT else "tab:green"
        ax.text(x[i], max(b_means[i], l_means[i]) + 0.25,
                f"Δ {sign}{abs(deg):.1f}%", ha="center",
                fontsize=9, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats_present],
                       rotation=15, ha="right")
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Mean score (1-5)")
    ax.set_title("H4 Base-Capability Test — Base vs LoRA-10K (per category)")
    ax.axhline(5.0, color="gray", linewidth=0.5, linestyle=":")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "h4_category_comparison.png", dpi=150)
    plt.close(fig)

    # ── Plot: score histograms ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(score_bins))
    w = 0.36
    ax.bar(xs - w/2, [base_hist[s] for s in score_bins], w,
           color=CONDITION_COLORS["base"], label=CONDITION_LABELS["base"])
    ax.bar(xs + w/2, [lora_hist[s] for s in score_bins], w,
           color=CONDITION_COLORS["lora"], label=CONDITION_LABELS["lora"])
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in score_bins])
    ax.set_xlabel("Score")
    ax.set_ylabel("Probe count")
    ax.set_title(f"H4 Score Distribution (n={len(paired_ids)} probes per condition)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "h4_score_distribution.png", dpi=150)
    plt.close(fig)

    # ── Markdown report ──────────────────────────────────────────────────
    lines = []
    lines.append("# H4 Base-Capability Test — Summary Report\n")
    lines.append(f"**n probes (paired):** {len(paired_ids)}  ")
    lines.append(f"**Judge:** Sonnet 4.5  ")
    lines.append(f"**Pass threshold:** < {DEGRADATION_THRESHOLD_PCT:.0f}% mean degradation (lora vs base)\n")

    lines.append("## Overall verdict\n")
    lines.append(f"**Verdict: {overall_verdict}**")
    lines.append("")
    lines.append("| | Mean | SD |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Base Qwen2.5-7B | {base_mean:.3f} | {base_std:.3f} |")
    lines.append(f"| Base + LoRA-10K | {lora_mean:.3f} | {lora_std:.3f} |")
    lines.append(f"| ΔMean (lora − base) | {diffs.mean():+.3f} | — |")
    lines.append("")
    lines.append(f"- Degradation: **{overall_deg:+.2f}%** "
                 f"(threshold = {DEGRADATION_THRESHOLD_PCT:.0f}% → "
                 f"{'PASS' if overall_deg < DEGRADATION_THRESHOLD_PCT else 'FAIL'})")
    lines.append(f"- Paired t-test: t = {t_stat:+.3f}, p = {p_value:.3g}")
    lines.append(f"- Cohen's d (paired): {cohens_d:+.3f}")
    lines.append("")

    lines.append("## Per-category breakdown\n")
    lines.append("| Category | n | Base mean | LoRA mean | Δ | Deg % | Verdict | p |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|---:|")
    for c in CATEGORIES:
        d = per_cat.get(c)
        if d is None:
            continue
        lines.append(
            f"| {CATEGORY_LABELS[c]} | {d['n']} | "
            f"{d['base_mean']:.3f} | {d['lora_mean']:.3f} | "
            f"{d['delta_mean']:+.3f} | {d['degradation_pct']:+.2f}% | "
            f"{d['verdict']} | {d['p_value']:.3g} |"
        )
    lines.append("")

    lines.append("## Score distribution\n")
    lines.append("| Score | Base count | LoRA count |")
    lines.append("|:---:|---:|---:|")
    for s in score_bins:
        lines.append(f"| {s} | {base_hist[s]} | {lora_hist[s]} |")
    lines.append("")

    if regressions:
        lines.append(f"## Regressions (LoRA scored ≥2 points lower) — {len(regressions)} probes\n")
        for r in regressions[:20]:
            lines.append(f"- **{r['probe_id']}** ({r['category']}, "
                         f"base={r['base_score']} → lora={r['lora_score']}): "
                         f"_{r['prompt']}_")
        if len(regressions) > 20:
            lines.append(f"- ... and {len(regressions) - 20} more (see h4_analysis_data.json)")
        lines.append("")
    else:
        lines.append("## Regressions\n\nNo probes regressed by ≥2 points.\n")

    lines.append("## Decision context\n")
    failed_cats = [c for c in CATEGORIES
                   if per_cat.get(c) and per_cat[c]["verdict"] == "FAIL"]
    if overall_verdict == "PASS" and not failed_cats:
        lines.append(
            "H4 **PASSED**. The LoRA-10K adapter does not cause catastrophic "
            "forgetting on out-of-domain capability — overall mean degradation "
            f"is below the {DEGRADATION_THRESHOLD_PCT:.0f}% threshold, and no "
            "individual category regressed beyond it either. The Phase 1 "
            "deployment recommendation (Qwen2.5-7B-Instruct + LoRA-10K) can "
            "remain a single-adapter deployment; no dual-mode loading "
            "(persona vs general-purpose) is required."
        )
    elif overall_verdict == "PASS" and failed_cats:
        worst = max(failed_cats, key=lambda c: per_cat[c]["degradation_pct"])
        cat_list = ", ".join(CATEGORY_LABELS[c] for c in failed_cats)
        lines.append(
            f"H4 **PASSED overall** ({overall_deg:+.2f}% mean degradation, "
            f"below the {DEGRADATION_THRESHOLD_PCT:.0f}% threshold), but with "
            f"a category-level caveat: **{cat_list}** regressed beyond "
            f"threshold (worst: {CATEGORY_LABELS[worst]} at "
            f"{per_cat[worst]['degradation_pct']:+.2f}%). The single-adapter "
            "deployment is viable for the general-purpose case, but consider "
            "(a) reviewing the regression list below for systematic patterns, "
            "and (b) loading the base model for queries in the affected "
            "category if precision matters there."
        )
    else:
        worst = max(
            (c for c in CATEGORIES if per_cat.get(c)),
            key=lambda c: per_cat[c]["degradation_pct"],
        )
        lines.append(
            "H4 **FAILED**. The LoRA-10K adapter degrades out-of-domain "
            f"capability by {overall_deg:.2f}%, exceeding the "
            f"{DEGRADATION_THRESHOLD_PCT:.0f}% threshold. Worst-affected "
            f"category: **{CATEGORY_LABELS[worst]}** "
            f"({per_cat[worst]['degradation_pct']:+.2f}%). Deployment "
            "implication: either (a) load the adapter only for in-persona "
            "contexts and serve a base model for general queries, or "
            "(b) retrain with regularization mixture (Plan §11 risk "
            "mitigation — add ~500 base-task examples to training data, "
            "lower learning rate, or reduce rank)."
        )
    lines.append("")

    (RESULTS_DIR / "h4_summary_report.md").write_text("\n".join(lines))

    # ── Console summary ──────────────────────────────────────────────────
    print("=" * 60)
    print("H4 Base-Capability Test")
    print("=" * 60)
    print(f"n probes:       {len(paired_ids)}")
    print(f"Base mean:      {base_mean:.3f} ± {base_std:.3f}")
    print(f"LoRA mean:      {lora_mean:.3f} ± {lora_std:.3f}")
    print(f"Δ (lora-base):  {diffs.mean():+.3f}")
    print(f"Degradation:    {overall_deg:+.2f}%  (threshold {DEGRADATION_THRESHOLD_PCT:.0f}%)")
    print(f"Paired t:       t={t_stat:+.3f}, p={p_value:.3g}")
    print(f"Cohen's d:      {cohens_d:+.3f}")
    print()
    print(f"Overall verdict: {overall_verdict}")
    print()
    print("Per-category:")
    for c in CATEGORIES:
        d = per_cat.get(c)
        if d is None:
            continue
        print(f"  {CATEGORY_LABELS[c]:25s}  "
              f"base={d['base_mean']:.2f}  lora={d['lora_mean']:.2f}  "
              f"Δ={d['delta_mean']:+.2f}  deg={d['degradation_pct']:+.2f}%  [{d['verdict']}]")
    print()
    print(f"Wrote {RESULTS_DIR / 'h4_summary_report.md'}")
    print(f"      {RESULTS_DIR / 'h4_analysis_data.json'}")
    print(f"      {RESULTS_DIR / 'h4_category_comparison.png'}")
    print(f"      {RESULTS_DIR / 'h4_score_distribution.png'}")


if __name__ == "__main__":
    main()
