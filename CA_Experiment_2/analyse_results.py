#!/usr/bin/env python3
"""
Analysis for CA Experiment 2: 4-condition evaluation.

Reads scores from logs/condition_<X>/scores_*.jsonl (X ∈ A/B/C/D), computes
per-condition statistics, produces plots, and writes a summary_report.md.

Per plan §6:
  6.1 Primary comparison: Condition C vs Condition D (paired t-test on 30
      scripts, target mean PersonaScore ≥ 3.5)
  6.2 Dimension-level breakdown (T/E/C/S means per condition); H2 is the
      ΔE = E_finetuned − E_base test (≥0.30 = LoRA addresses episodic;
      <0.15 = architectural ceiling at 7B)
  6.3 2×2 ANOVA on (fine-tuning × SCI) for H3 (interaction term)
  6.5 Failure-mode taxonomy across all 4 conditions

Optional H5 learning curve: pass --learning-curve-data with comma-separated
n:mean pairs to plot the fit (requires Condition C re-runs with smaller
adapters, which aren't part of the default Phase 3 evaluation).

Outputs to results/:
  persona_score_timeseries.{png,svg,pdf}    — mean PersonaScore by turn
  dimension_comparison.{png,svg,pdf}         — bar chart of T/E/C/S means
  condition_comparison.{png,pdf}             — overall mean + 95% CI per cond
  failure_modes.{png,pdf}                    — failure breakdown
  learning_curve.{png,pdf}                   — H5 log fit (if data passed)
  summary_report.md                          — text summary + decision rules
  analysis_data.json                         — raw stats (for paper figures)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# ── Paths ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
LOGS_BASE = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

CONDITIONS = ["A", "B", "C", "D"]
CONDITION_LABELS = {
    "A": "A: Fine-tuned, no SCI",
    "B": "B: Fine-tuned, baseline SCI",
    "C": "C: Fine-tuned, Combined SCI",
    "D": "D: Base model, Combined SCI (control)",
}
CONDITION_COLORS = {
    "A": "#9e9e9e",   # gray
    "B": "#2196f3",   # blue
    "C": "#4caf50",   # green (primary)
    "D": "#ff9800",   # orange (control)
}

PROBE_TURNS = (5, 10, 15, 20, 25, 30, 35, 40)
DIMENSIONS = ("T", "E", "C", "S")
PASS_THRESHOLD = 3.5  # plan §2 H1


# ── Data loading ─────────────────────────────────────────────────────────

def load_condition(cond: str, logs_root: Path = LOGS_BASE) -> list[dict]:
    """Load all score records for a condition (primary log dir only)."""
    return _load_logs_dir(logs_root / f"condition_{cond}")


def _load_logs_dir(cond_dir: Path) -> list[dict]:
    if not cond_dir.exists():
        return []
    records = []
    for path in sorted(cond_dir.glob("scores_*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


# Map adapter folder name → training-set size, for H5 auto-detection
ADAPTER_SIZE_MAP = {
    "lora_2k": 2000,
    "lora_5k": 5000,
    "lora_10k": 10000,  # primary Condition C
}


def auto_detect_h5_curve(logs_root: Path = LOGS_BASE) -> list[tuple[int, float]]:
    """Scan logs/ for Condition C sub-runs and compute (n, mean_score) points.

    Looks at:
      - condition_C/                 → LoRA-10K (primary, n=10000)
      - condition_C_lora_2k/         → n=2000
      - condition_C_lora_5k/         → n=5000
      - any condition_C_<other>/     → if the suffix matches ADAPTER_SIZE_MAP

    Returns sorted list of (n, mean) where logs were found and non-empty."""
    points: list[tuple[int, float]] = []

    primary = _load_logs_dir(logs_root / "condition_C")
    if primary:
        points.append((ADAPTER_SIZE_MAP["lora_10k"],
                       float(np.mean([r["score"] for r in primary]))))

    for sub in sorted(logs_root.glob("condition_C_*")):
        if not sub.is_dir():
            continue
        suffix = sub.name[len("condition_C_"):]
        n = ADAPTER_SIZE_MAP.get(suffix)
        if n is None:
            continue
        recs = _load_logs_dir(sub)
        if not recs:
            continue
        points.append((n, float(np.mean([r["score"] for r in recs]))))

    points.sort(key=lambda p: p[0])
    return points


# ── Stats ────────────────────────────────────────────────────────────────

def per_turn_means(records: list[dict]) -> dict[int, dict]:
    """For each probe turn, compute mean score across all probes (all dims, all scripts)."""
    by_turn: dict[int, list[float]] = defaultdict(list)
    for r in records:
        by_turn[r["turn"]].append(r["score"])
    out = {}
    for t in PROBE_TURNS:
        scores = by_turn.get(t, [])
        if not scores:
            out[t] = {"n": 0, "mean": float("nan"), "std": float("nan"),
                      "ci_low": float("nan"), "ci_high": float("nan")}
            continue
        arr = np.array(scores, dtype=float)
        n = len(arr)
        m = arr.mean()
        s = arr.std(ddof=1) if n > 1 else 0.0
        sem = s / np.sqrt(n) if n > 1 else 0.0
        ci_h = 1.96 * sem
        out[t] = {"n": n, "mean": float(m), "std": float(s),
                  "ci_low": float(m - ci_h), "ci_high": float(m + ci_h)}
    return out


def per_dimension_means(records: list[dict]) -> dict[str, dict]:
    """Per-dimension mean across all turns and scripts."""
    by_dim: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_dim[r["dimension"]].append(r["score"])
    out = {}
    for d in DIMENSIONS:
        scores = by_dim.get(d, [])
        if not scores:
            out[d] = {"n": 0, "mean": float("nan"), "std": float("nan")}
            continue
        arr = np.array(scores, dtype=float)
        out[d] = {"n": len(arr), "mean": float(arr.mean()),
                  "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}
    return out


def overall_mean(records: list[dict]) -> dict:
    if not records:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    arr = np.array([r["score"] for r in records], dtype=float)
    n = len(arr)
    m = arr.mean()
    s = arr.std(ddof=1) if n > 1 else 0.0
    sem = s / np.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": float(m), "std": float(s),
            "ci_low": float(m - 1.96 * sem), "ci_high": float(m + 1.96 * sem)}


def per_script_means(records: list[dict]) -> dict[int, float]:
    """Mean score per script (all probes within that script averaged)."""
    by_script: dict[int, list[float]] = defaultdict(list)
    for r in records:
        by_script[r["script_id"]].append(r["score"])
    return {sid: float(np.mean(s)) for sid, s in by_script.items()}


def paired_test_C_vs_D(records_C: list[dict], records_D: list[dict]):
    """Paired t-test on per-script means between Conditions C and D (§6.1)."""
    c_means = per_script_means(records_C)
    d_means = per_script_means(records_D)
    common = sorted(set(c_means) & set(d_means))
    if len(common) < 2:
        return None
    c_arr = np.array([c_means[s] for s in common])
    d_arr = np.array([d_means[s] for s in common])
    diff = c_arr - d_arr
    t_stat, p_val = stats.ttest_rel(c_arr, d_arr)
    pooled = np.sqrt((c_arr.var(ddof=1) + d_arr.var(ddof=1)) / 2)
    cohen_d = float(diff.mean() / pooled) if pooled > 0 else float("nan")
    return {
        "n_scripts": len(common),
        "C_mean": float(c_arr.mean()),
        "D_mean": float(d_arr.mean()),
        "delta": float(diff.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohen_d": cohen_d,
        "C_ge_threshold": float(c_arr.mean()) >= PASS_THRESHOLD,
    }


def two_by_two_anova(data: dict[str, list[dict]]):
    """2×2 marginal-means analysis: factors (fine_tuning, sci) (§6.3 H3).

    Cells:
      (FT=Y, SCI=none)    → A
      (FT=Y, SCI=Combined) → C
      (FT=N, SCI=Combined) → D
      (FT=N, SCI=none)    → not collected
    With one cell missing the formal interaction term is degenerate; we
    report main effects and the C vs D contrast instead. B is excluded
    because its SCI=baseline doesn't fit the binary SCI factor.
    """
    rows = []
    for c, recs in data.items():
        if c == "B":
            continue
        ft = 1 if c in ("A", "C") else 0
        sci = 1 if c in ("C", "D") else 0
        for r in recs:
            rows.append((ft, sci, r["score"]))
    if not rows:
        return None

    arr = np.array(rows, dtype=float)
    ft_col, sci_col, y = arr[:, 0], arr[:, 1], arr[:, 2]
    grand = y.mean()

    means = {}
    for ft_v in (0, 1):
        for sci_v in (0, 1):
            mask = (ft_col == ft_v) & (sci_col == sci_v)
            if mask.sum() == 0:
                continue
            means[(ft_v, sci_v)] = {"n": int(mask.sum()), "mean": float(y[mask].mean())}

    ft_main = {ft_v: float(y[ft_col == ft_v].mean()) for ft_v in (0, 1) if (ft_col == ft_v).any()}
    sci_main = {sci_v: float(y[sci_col == sci_v].mean()) for sci_v in (0, 1) if (sci_col == sci_v).any()}

    return {
        "grand_mean": float(grand),
        "cell_means": {f"FT={k[0]},SCI={k[1]}": v for k, v in means.items()},
        "main_effect_FT":   {str(k): v for k, v in ft_main.items()},
        "main_effect_SCI":  {str(k): v for k, v in sci_main.items()},
        "delta_FT":  ft_main.get(1, 0.0) - ft_main.get(0, 0.0) if 0 in ft_main and 1 in ft_main else None,
        "delta_SCI": sci_main.get(1, 0.0) - sci_main.get(0, 0.0) if 0 in sci_main and 1 in sci_main else None,
    }


def failure_mode_counts(records: list[dict]) -> dict[str, int]:
    """Count score ≤ 2 per dimension — proxy for failure-mode taxonomy (§6.5)."""
    out: dict[str, int] = {d: 0 for d in DIMENSIONS}
    for r in records:
        if r["score"] <= 2:
            out[r["dimension"]] = out.get(r["dimension"], 0) + 1
    return out


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_persona_score_timeseries(per_cond_per_turn: dict[str, dict[int, dict]], out_dir: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for cond in CONDITIONS:
        per_turn = per_cond_per_turn.get(cond)
        if not per_turn or all(np.isnan(per_turn[t]["mean"]) for t in PROBE_TURNS):
            continue
        means = [per_turn[t]["mean"] for t in PROBE_TURNS]
        ci_lo = [per_turn[t]["ci_low"] for t in PROBE_TURNS]
        ci_hi = [per_turn[t]["ci_high"] for t in PROBE_TURNS]
        ax.plot(PROBE_TURNS, means, marker="o", linewidth=2,
                color=CONDITION_COLORS[cond], label=CONDITION_LABELS[cond])
        ax.fill_between(PROBE_TURNS, ci_lo, ci_hi,
                        color=CONDITION_COLORS[cond], alpha=0.15)
    ax.axhline(PASS_THRESHOLD, linestyle="--", color="black", alpha=0.4,
               label=f"Pass threshold ({PASS_THRESHOLD})")
    ax.set_xlabel("Probe turn")
    ax.set_ylabel("Mean PersonaScore")
    ax.set_title("PersonaScore over conversation depth — 4-condition comparison")
    ax.set_ylim(1.0, 5.0)
    ax.set_xticks(PROBE_TURNS)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"persona_score_timeseries.{ext}", dpi=150)
    plt.close(fig)


def plot_dimension_comparison(per_cond_per_dim: dict[str, dict[str, dict]], out_dir: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    n_groups = len(DIMENSIONS)
    width = 0.18
    x = np.arange(n_groups)
    for i, cond in enumerate(CONDITIONS):
        per_dim = per_cond_per_dim.get(cond, {})
        means = [per_dim.get(d, {}).get("mean", float("nan")) for d in DIMENSIONS]
        ax.bar(x + (i - 1.5) * width, means, width,
               label=CONDITION_LABELS[cond], color=CONDITION_COLORS[cond])
    ax.axhline(PASS_THRESHOLD, linestyle="--", color="black", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["Trait", "Episodic", "Capability", "Style"])
    ax.set_ylabel("Mean PersonaScore")
    ax.set_title("Dimension-level PersonaScore by condition")
    ax.set_ylim(1.0, 5.0)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"dimension_comparison.{ext}", dpi=150)
    plt.close(fig)


def plot_condition_comparison(per_cond_overall: dict[str, dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    conds = [c for c in CONDITIONS if per_cond_overall.get(c, {}).get("n", 0) > 0]
    means = [per_cond_overall[c]["mean"] for c in conds]
    err_lo = [per_cond_overall[c]["mean"] - per_cond_overall[c]["ci_low"] for c in conds]
    err_hi = [per_cond_overall[c]["ci_high"] - per_cond_overall[c]["mean"] for c in conds]
    colors = [CONDITION_COLORS[c] for c in conds]
    ax.bar(conds, means, yerr=[err_lo, err_hi], color=colors, capsize=5)
    ax.axhline(PASS_THRESHOLD, linestyle="--", color="black", alpha=0.4,
               label=f"Pass threshold ({PASS_THRESHOLD})")
    for i, c in enumerate(conds):
        ax.text(i, means[i] + 0.05, f"{means[i]:.2f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean PersonaScore (95% CI)")
    ax.set_title("Overall PersonaScore by condition")
    ax.set_ylim(1.0, 5.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"condition_comparison.{ext}", dpi=150)
    plt.close(fig)


def plot_failure_modes(per_cond_failures: dict[str, dict[str, int]], out_dir: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    n_groups = len(DIMENSIONS)
    width = 0.18
    x = np.arange(n_groups)
    for i, cond in enumerate(CONDITIONS):
        counts = [per_cond_failures.get(cond, {}).get(d, 0) for d in DIMENSIONS]
        ax.bar(x + (i - 1.5) * width, counts, width,
               label=CONDITION_LABELS[cond], color=CONDITION_COLORS[cond])
    ax.set_xticks(x)
    ax.set_xticklabels(["Trait", "Episodic", "Capability", "Style"])
    ax.set_ylabel("Failures (score ≤ 2) — count")
    ax.set_title("Failure-mode taxonomy by dimension and condition")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"failure_modes.{ext}", dpi=150)
    plt.close(fig)


def plot_learning_curve(curve_points: list[tuple[int, float]], out_dir: Path):
    """Fit y = a*log(n) + b to (n, mean_PersonaScore) points (§6.4 H5)."""
    if len(curve_points) < 2:
        return None
    ns = np.array([p[0] for p in curve_points], dtype=float)
    ys = np.array([p[1] for p in curve_points], dtype=float)
    log_ns = np.log(ns)
    a, b = np.polyfit(log_ns, ys, 1)
    fit_x = np.linspace(ns.min() * 0.9, ns.max() * 2.1, 100)
    fit_y = a * np.log(fit_x) + b

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(fit_x, fit_y, color="#666", linestyle="--",
            label=f"Fit: {a:.3f}·log(n) + {b:.3f}")
    ax.scatter(ns, ys, color="#4caf50", s=80, zorder=3, label="Observed")
    for n, y in curve_points:
        ax.annotate(f"n={n}\n{y:.2f}", (n, y),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.axhline(PASS_THRESHOLD, linestyle=":", color="black", alpha=0.4,
               label=f"Pass threshold ({PASS_THRESHOLD})")
    ax.set_xscale("log")
    ax.set_xlabel("LoRA training examples (n)")
    ax.set_ylabel("Mean PersonaScore")
    ax.set_title("H5: data-scaling learning curve")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"learning_curve.{ext}", dpi=150)
    plt.close(fig)
    return {"a": float(a), "b": float(b),
            "predicted_at_20k": float(a * np.log(20000) + b)}


# ── Report ───────────────────────────────────────────────────────────────

def write_summary_report(out_dir: Path, results: dict):
    lines: list[str] = []
    lines.append("# CA Experiment 2 — Analysis Summary\n")
    lines.append(f"_Generated: {results.get('timestamp', '')}_\n\n")

    # Per-condition overall
    lines.append("## 1. Overall PersonaScore by Condition\n\n")
    lines.append("| Condition | Description | n_probes | Mean | 95% CI | Std |\n")
    lines.append("|---|---|---:|---:|---|---:|\n")
    for c in CONDITIONS:
        ov = results["overall"].get(c, {})
        if ov.get("n", 0) == 0:
            lines.append(f"| {c} | {CONDITION_LABELS[c]} | 0 | — | — | — |\n")
            continue
        lines.append(
            f"| {c} | {CONDITION_LABELS[c]} | {ov['n']} | "
            f"**{ov['mean']:.3f}** | [{ov['ci_low']:.3f}, {ov['ci_high']:.3f}] | "
            f"{ov['std']:.3f} |\n"
        )
    lines.append("\n")

    # Per-turn
    lines.append("## 2. PersonaScore by Probe Turn\n\n")
    lines.append("| Turn | " + " | ".join(CONDITIONS) + " |\n")
    lines.append("|---:|" + "|".join([":---:" for _ in CONDITIONS]) + "|\n")
    for t in PROBE_TURNS:
        cells = [f"{t}"]
        for c in CONDITIONS:
            stats_t = results["per_turn"].get(c, {}).get(t, {})
            if stats_t.get("n", 0) == 0:
                cells.append("—")
            else:
                cells.append(f"{stats_t['mean']:.2f}")
        lines.append("| " + " | ".join(cells) + " |\n")
    lines.append("\n")

    # Dimension breakdown
    lines.append("## 3. Per-Dimension Means (H2 diagnostic)\n\n")
    lines.append("| Dimension | A | B | C | D | ΔE = C − D |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for d, name in zip(DIMENSIONS, ["Trait", "Episodic", "Capability", "Style"]):
        cells = [f"{name} ({d})"]
        for c in CONDITIONS:
            stats_d = results["per_dim"].get(c, {}).get(d, {})
            cells.append(f"{stats_d['mean']:.3f}" if stats_d.get("n", 0) else "—")
        if d == "E":
            c_e = results["per_dim"].get("C", {}).get("E", {}).get("mean")
            d_e = results["per_dim"].get("D", {}).get("E", {}).get("mean")
            delta_str = f"**{c_e - d_e:+.3f}**" if (c_e is not None and d_e is not None) else "—"
            cells.append(delta_str)
        else:
            cells.append("—")
        lines.append("| " + " | ".join(cells) + " |\n")
    lines.append("\n")
    lines.append("**H2 interpretation (§6.2):**  \n")
    lines.append("- ΔE ≥ +0.30 → fine-tuning meaningfully addresses episodic fabrication.  \n")
    lines.append("- ΔE < +0.15 → episodic ceiling is architectural at 7B (LoRA can't fix).  \n\n")

    # Primary comparison: C vs D
    pc = results.get("paired_C_vs_D")
    lines.append("## 4. Primary Comparison: Condition C vs Condition D (§6.1)\n\n")
    if pc:
        sig = pc["p_value"] < 0.05
        lines.append(f"- **n_scripts**: {pc['n_scripts']}\n")
        lines.append(f"- **Mean (Condition C)**: {pc['C_mean']:.3f}\n")
        lines.append(f"- **Mean (Condition D)**: {pc['D_mean']:.3f}\n")
        lines.append(f"- **Δ (C − D)**: {pc['delta']:+.3f}\n")
        lines.append(f"- **Paired t-test**: t = {pc['t_stat']:.3f}, p = {pc['p_value']:.4f} "
                     f"({'significant' if sig else 'not significant'} at α=0.05)\n")
        lines.append(f"- **Cohen's d**: {pc['cohen_d']:.3f}\n")
        lines.append(f"- **H1 (C ≥ {PASS_THRESHOLD})**: "
                     f"{'PASSED ✓' if pc['C_ge_threshold'] else 'FAILED ✗'}\n\n")
    else:
        lines.append("_Not enough data — need both Condition C and Condition D logs._\n\n")

    # 2×2 ANOVA
    av = results.get("anova")
    if av:
        lines.append("## 5. 2-Way Marginal Means: Fine-tuning × SCI (§6.3, H3)\n\n")
        lines.append("| Cell (FT, SCI) | Mean |\n|---|---:|\n")
        for k, v in av["cell_means"].items():
            lines.append(f"| {k} | {v['mean']:.3f} (n={v['n']}) |\n")
        lines.append("\n")
        if av.get("delta_FT") is not None:
            lines.append(f"- **Main effect of fine-tuning**: Δ = {av['delta_FT']:+.3f}\n")
        if av.get("delta_SCI") is not None:
            lines.append(f"- **Main effect of SCI strategy**: Δ = {av['delta_SCI']:+.3f}\n")
        lines.append("\n_Note: Condition E (FT=N, SCI=none) was not collected, so a full 2×2 "
                     "interaction term is not estimable. The C vs D comparison above is the "
                     "cleanest test of fine-tuning's contribution given Combined SCI._\n\n")

    # Failure modes
    fm = results.get("failures")
    if fm:
        lines.append("## 6. Failure Modes (score ≤ 2) by Dimension (§6.5)\n\n")
        lines.append("| Dimension | A | B | C | D |\n|---|---:|---:|---:|---:|\n")
        for d in DIMENSIONS:
            cells = [d]
            for c in CONDITIONS:
                cells.append(str(fm.get(c, {}).get(d, 0)))
            lines.append("| " + " | ".join(cells) + " |\n")
        lines.append("\n")

    # Learning curve (if present)
    lc = results.get("learning_curve")
    if lc:
        lines.append("## 7. H5 Data-Scaling Learning Curve\n\n")
        lines.append(f"- Fit: PersonaScore(n) = {lc['a']:.4f} · log(n) + {lc['b']:.4f}\n")
        lines.append(f"- **Predicted at n = 20,000**: {lc['predicted_at_20k']:.3f}\n\n")

    # Decision rules
    lines.append("## 8. Decision Rules Triggered (§7)\n\n")
    if pc:
        c_mean = pc["C_mean"]
        d_e = results["per_dim"].get("D", {}).get("E", {}).get("mean", 0.0)
        c_e = results["per_dim"].get("C", {}).get("E", {}).get("mean", 0.0)
        delta_e = c_e - d_e
        if c_mean >= PASS_THRESHOLD and delta_e >= 0.30:
            lines.append("→ **Outcome A**: Fine-tuning solves both overall and episodic gaps. "
                         "SMC architecture complete at 7B; 14B experiment unnecessary.\n\n")
        elif c_mean >= PASS_THRESHOLD and delta_e < 0.15:
            lines.append("→ **Outcome B**: Fine-tuning closes overall gap but not episodic. "
                         "Deploy with Combined SCI; document episodic ceiling as 7B-architectural.\n\n")
        elif 3.35 <= c_mean < PASS_THRESHOLD:
            lines.append("→ **Outcome C**: Fine-tuning narrows but doesn't close gap. "
                         "Consider LoRA-20K, larger rank, or accept 3.4 as practical threshold.\n\n")
        elif c_mean < 3.35:
            lines.append("→ **Outcome D**: Fine-tuning has minimal effect. "
                         "7B insufficient; 14B experiment becomes mandatory.\n\n")
        else:
            lines.append("→ Outcome between thresholds — see numbers above.\n\n")

    # Replication check (§6.5)
    if results["overall"].get("D", {}).get("n", 0) > 0:
        lines.append("## 9. Condition D Replication Check (§6.5)\n\n")
        lines.append("Per Experiment 1, the Combined-SCI mean was **3.20** on identical scripts.  \n")
        d_mean = results["overall"]["D"]["mean"]
        delta = d_mean - 3.20
        lines.append(f"Condition D mean: **{d_mean:.3f}** (Δ = {delta:+.3f})\n")
        if abs(delta) > 0.10:
            lines.append(f"\n⚠ **Replication concern**: |Δ| > 0.10 may indicate judge drift between "
                         f"Sonnet 4.5 (Exp 1) and Sonnet 4.6 (Exp 2). Run inter-rater reliability "
                         f"check before interpreting fine-tuning effects.\n\n")
        else:
            lines.append(f"\n✓ Replication within tolerance (|Δ| ≤ 0.10) — judge stability OK.\n\n")

    out_path = out_dir / "summary_report.md"
    out_path.write_text("".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CA Experiment 2 analysis")
    p.add_argument("--logs-root", type=Path, default=LOGS_BASE,
                   help="Root containing condition_<X>/ folders")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                   help="Output directory for plots and report")
    p.add_argument("--learning-curve-data", type=str, default=None,
                   help="Optional comma-separated 'n:mean' pairs for H5 plot, "
                        "e.g. '2000:2.85,5000:3.10,10000:3.32'. Skipped if omitted.")
    args = p.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading logs from {args.logs_root}...")
    data = {c: load_condition(c, args.logs_root) for c in CONDITIONS}
    for c in CONDITIONS:
        print(f"  Condition {c}: {len(data[c])} probe records")

    if not any(data.values()):
        print("No condition data found. Run experiment_runner.py first.", file=sys.stderr)
        sys.exit(1)

    results = {
        "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "overall": {c: overall_mean(data[c]) for c in CONDITIONS},
        "per_turn": {c: per_turn_means(data[c]) for c in CONDITIONS},
        "per_dim": {c: per_dimension_means(data[c]) for c in CONDITIONS},
        "failures": {c: failure_mode_counts(data[c]) for c in CONDITIONS},
        "paired_C_vs_D": paired_test_C_vs_D(data["C"], data["D"]) if data["C"] and data["D"] else None,
        "anova": two_by_two_anova(data),
    }

    # H5 learning curve: prefer manual data, fall back to auto-detection from logs
    points: list[tuple[int, float]] = []
    if args.learning_curve_data:
        try:
            for chunk in args.learning_curve_data.split(","):
                n_str, m_str = chunk.split(":")
                points.append((int(n_str.strip()), float(m_str.strip())))
            print(f"H5: using {len(points)} manual data points from --learning-curve-data")
        except Exception as e:
            print(f"Failed to parse --learning-curve-data: {e}", file=sys.stderr)
    else:
        points = auto_detect_h5_curve(args.logs_root)
        if points:
            print(f"H5: auto-detected {len(points)} Condition-C sub-runs:")
            for n, m in points:
                print(f"  n={n:>5}: mean PersonaScore = {m:.3f}")

    if len(points) >= 2:
        lc = plot_learning_curve(points, args.results_dir)
        if lc:
            lc["points"] = points
        results["learning_curve"] = lc
    elif len(points) == 1:
        print("H5: only 1 data point — need ≥2 for the log fit. Run Condition C with "
              "another adapter (e.g., --adapter lora_2k --logs-suffix _lora_2k).")

    print("\nGenerating plots...")
    plot_persona_score_timeseries(results["per_turn"], args.results_dir)
    plot_dimension_comparison(results["per_dim"], args.results_dir)
    plot_condition_comparison(results["overall"], args.results_dir)
    plot_failure_modes(results["failures"], args.results_dir)

    print("Generating summary report...")
    write_summary_report(args.results_dir, results)

    (args.results_dir / "analysis_data.json").write_text(json.dumps(results, indent=2, default=str))

    print("\n" + "=" * 60)
    print(f"Deliverables saved to {args.results_dir}/")
    for p_path in sorted(args.results_dir.iterdir()):
        print(f"  - {p_path.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
