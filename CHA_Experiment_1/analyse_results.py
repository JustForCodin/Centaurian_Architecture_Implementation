#!/usr/bin/env python3
"""
Phase 5: Analysis for CHA Experiment 1.

Produces all deliverables from Section 9:
- PersonaScore time series with 95% CIs
- Degradation model fits (linear/step/exponential/piecewise) with AIC
- Dimension-level T* values
- H4 correlation analysis
- Adversarial vs. naturalistic comparison
- Failure mode taxonomy
- Decision rules table populated with actual T*
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"  # may be overridden by --model arg
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULTS_DIR = BASE_DIR / "results"  # may be overridden by --model arg

PROBE_TURNS = [5, 10, 15, 20, 25, 30, 35, 40]
THRESHOLD = 3.5

# Script IDs used in the experiment
NATURALISTIC_IDS = list(range(1, 23))
ADVERSARIAL_IDS = list(range(81, 89))
ALL_IDS = NATURALISTIC_IDS + ADVERSARIAL_IDS

# ── Data Loading ───────────────────────────────────────────────────────────

def load_scores() -> list[dict]:
    """Load all score records from logs."""
    records = []
    for sid in ALL_IDS:
        path = LOGS_DIR / f"scores_{sid:03d}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_context() -> list[dict]:
    """Load all context fill records from logs."""
    records = []
    for sid in ALL_IDS:
        path = LOGS_DIR / f"context_{sid:03d}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_script_metadata() -> dict:
    """Load script metadata (adversarial flag)."""
    meta = {}
    for sid in ALL_IDS:
        path = SCRIPTS_DIR / f"script_{sid:03d}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            meta[sid] = {
                "is_adversarial": data.get("is_adversarial", False),
                "scenario": data.get("scenario", ""),
            }
    return meta


# ── 7.1 Primary Analysis: Finding T* ──────────────────────────────────────

def compute_persona_scores(scores: list[dict]) -> dict:
    """Compute per-script, per-turn composite PersonaScores.
    Returns {script_id: {turn: {"composite": float, "T": float, ...}}}"""
    by_script_turn = defaultdict(lambda: defaultdict(dict))

    for r in scores:
        sid = r["script_id"]
        turn = r["turn"]
        dim = r["dimension"]
        by_script_turn[sid][turn][dim] = r["score"]

    result = {}
    for sid, turns in by_script_turn.items():
        result[sid] = {}
        for turn, dim_scores in turns.items():
            dims_present = [d for d in ["T", "E", "C", "S"] if d in dim_scores]
            if dims_present:
                composite = np.mean([dim_scores[d] for d in dims_present])
                result[sid][turn] = {
                    "composite": composite,
                    **{d: dim_scores.get(d) for d in ["T", "E", "C", "S"]},
                }
    return result


def compute_mean_timeseries(persona_scores: dict, script_ids: list = None):
    """Compute mean PersonaScore and 95% CI at each probe turn.
    Returns (turns, means, ci_lower, ci_upper, stds, ns)."""
    if script_ids is None:
        script_ids = list(persona_scores.keys())

    turns = []
    means = []
    ci_lowers = []
    ci_uppers = []
    stds_out = []
    ns_out = []

    for t in PROBE_TURNS:
        values = []
        for sid in script_ids:
            if sid in persona_scores and t in persona_scores[sid]:
                values.append(persona_scores[sid][t]["composite"])
        if values:
            turns.append(t)
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0
            n = len(values)
            se = std / np.sqrt(n) if n > 1 else 0
            ci = 1.96 * se
            means.append(mean)
            ci_lowers.append(mean - ci)
            ci_uppers.append(mean + ci)
            stds_out.append(std)
            ns_out.append(n)

    return np.array(turns), np.array(means), np.array(ci_lowers), np.array(ci_uppers), stds_out, ns_out


def find_tstar(turns, means) -> int | None:
    """Find T* — first turn where mean PersonaScore < threshold."""
    for t, m in zip(turns, means):
        if m < THRESHOLD:
            return int(t)
    return None


# ── 7.2 Degradation Profile Fits ──────────────────────────────────────────

def fit_linear(turns, means):
    """PS(t) = alpha - beta*t"""
    try:
        slope, intercept, r, p, se = stats.linregress(turns, means)
        predicted = intercept + slope * turns
        residuals = means - predicted
        n = len(means)
        k = 2
        sse = np.sum(residuals**2)
        aic = n * np.log(sse / n) + 2 * k if sse > 0 else float("inf")
        return {"name": "linear", "params": {"alpha": intercept, "beta": -slope},
                "aic": aic, "r_squared": r**2, "predicted": predicted}
    except Exception:
        return {"name": "linear", "aic": float("inf"), "r_squared": 0, "predicted": means}


def fit_step(turns, means):
    """PS(t) = alpha if t < T*, else alpha - delta. Try all possible breakpoints."""
    best_aic = float("inf")
    best_result = None
    n = len(means)

    for bp_idx in range(1, n - 1):
        bp = turns[bp_idx]
        alpha = np.mean(means[:bp_idx])
        alpha_minus_delta = np.mean(means[bp_idx:])
        delta = alpha - alpha_minus_delta
        predicted = np.where(turns < bp, alpha, alpha - delta)
        residuals = means - predicted
        sse = np.sum(residuals**2)
        k = 3  # alpha, delta, breakpoint
        aic = n * np.log(sse / n) + 2 * k if sse > 0 else float("inf")
        if aic < best_aic:
            best_aic = aic
            best_result = {
                "name": "step", "params": {"alpha": alpha, "delta": delta, "breakpoint": int(bp)},
                "aic": aic, "predicted": predicted,
            }

    if best_result is None:
        best_result = {"name": "step", "aic": float("inf"), "predicted": means}
    return best_result


def fit_exponential(turns, means):
    """PS(t) = alpha * exp(-lambda*t)"""
    try:
        def exp_func(t, alpha, lam):
            return alpha * np.exp(-lam * t)
        popt, _ = optimize.curve_fit(exp_func, turns, means, p0=[5.0, 0.01],
                                      maxfev=5000, bounds=([0, 0], [6, 1]))
        predicted = exp_func(turns, *popt)
        residuals = means - predicted
        n = len(means)
        k = 2
        sse = np.sum(residuals**2)
        aic = n * np.log(sse / n) + 2 * k if sse > 0 else float("inf")
        return {"name": "exponential", "params": {"alpha": popt[0], "lambda": popt[1]},
                "aic": aic, "predicted": predicted}
    except Exception:
        return {"name": "exponential", "aic": float("inf"), "predicted": means}


def fit_piecewise(turns, means):
    """PS(t) = alpha if t < T0, else alpha - beta*(t - T0)"""
    best_aic = float("inf")
    best_result = None
    n = len(means)

    for bp_idx in range(1, n - 1):
        t0 = turns[bp_idx]
        alpha = np.mean(means[:bp_idx + 1])
        # Fit slope on declining portion
        declining_turns = turns[bp_idx:] - t0
        declining_means = means[bp_idx:]
        if len(declining_turns) >= 2:
            slope, _, _, _, _ = stats.linregress(declining_turns, declining_means)
            beta = -slope
            predicted = np.where(turns <= t0, alpha, alpha - beta * (turns - t0))
            residuals = means - predicted
            sse = np.sum(residuals**2)
            k = 3
            aic = n * np.log(sse / n) + 2 * k if sse > 0 else float("inf")
            if aic < best_aic:
                best_aic = aic
                best_result = {
                    "name": "piecewise", "params": {"alpha": alpha, "beta": beta, "T0": int(t0)},
                    "aic": aic, "predicted": predicted,
                }

    if best_result is None:
        best_result = {"name": "piecewise", "aic": float("inf"), "predicted": means}
    return best_result


# ── 7.3 Dimension Ordering ────────────────────────────────────────────────

def compute_dimension_tstar(persona_scores: dict):
    """Compute T* for each dimension separately."""
    dim_tstar = {}
    for dim in ["T", "E", "C", "S"]:
        dim_means_by_turn = {}
        for t in PROBE_TURNS:
            values = []
            for sid, turns in persona_scores.items():
                if t in turns and turns[t].get(dim) is not None:
                    values.append(turns[t][dim])
            if values:
                dim_means_by_turn[t] = np.mean(values)

        tstar = None
        for t in PROBE_TURNS:
            if t in dim_means_by_turn and dim_means_by_turn[t] < THRESHOLD:
                tstar = t
                break
        dim_tstar[dim] = {
            "tstar": tstar,
            "means": dim_means_by_turn,
        }
    return dim_tstar


# ── 7.4 H4 Correlation Analysis ───────────────────────────────────────────

def h4_correlation(scores: list[dict], context: list[dict]):
    """Compute correlation between context fill % and PersonaScore."""
    # Build context fill lookup: {(script_id, turn): fill_pct}
    ctx_lookup = {}
    for r in context:
        key = (r["script_id"], r["turn"])
        ctx_lookup[key] = r.get("context_fill_pct", 0)

    # Build per-conversation correlations
    persona_scores = compute_persona_scores(scores)
    per_conv_r_ctx = []
    per_conv_r_turn = []

    for sid, turns in persona_scores.items():
        ctx_vals = []
        score_vals = []
        turn_vals = []
        for t in sorted(turns.keys()):
            fill = ctx_lookup.get((sid, t))
            if fill is not None:
                ctx_vals.append(fill)
                score_vals.append(turns[t]["composite"])
                turn_vals.append(t)

        if len(ctx_vals) >= 4:
            r_ctx, p_ctx = stats.pearsonr(ctx_vals, score_vals)
            r_turn, p_turn = stats.pearsonr(turn_vals, score_vals)
            per_conv_r_ctx.append(r_ctx)
            per_conv_r_turn.append(r_turn)

    result = {
        "context_fill_correlation": {
            "mean_r": float(np.mean(per_conv_r_ctx)) if per_conv_r_ctx else 0,
            "std_r": float(np.std(per_conv_r_ctx)) if per_conv_r_ctx else 0,
            "n_conversations": len(per_conv_r_ctx),
        },
        "turn_count_correlation": {
            "mean_r": float(np.mean(per_conv_r_turn)) if per_conv_r_turn else 0,
            "std_r": float(np.std(per_conv_r_turn)) if per_conv_r_turn else 0,
            "n_conversations": len(per_conv_r_turn),
        },
    }

    # Which is more predictive?
    if per_conv_r_ctx and per_conv_r_turn:
        ctx_strength = abs(result["context_fill_correlation"]["mean_r"])
        turn_strength = abs(result["turn_count_correlation"]["mean_r"])
        if ctx_strength > turn_strength:
            result["h4_supported"] = True
            result["interpretation"] = (
                "Context fill % is more predictive than turn count → "
                "SCI token budget is the primary design variable"
            )
        else:
            result["h4_supported"] = False
            result["interpretation"] = (
                "Turn count is more predictive than context fill % → "
                "Cognitive drift is the primary driver, not displacement"
            )

    return result


# ── 7.5 Adversarial vs. Naturalistic ──────────────────────────────────────

def adversarial_comparison(persona_scores: dict, script_meta: dict):
    """Compare T* and degradation between adversarial and naturalistic scripts."""
    nat_ids = [sid for sid in persona_scores if not script_meta.get(sid, {}).get("is_adversarial", False)]
    adv_ids = [sid for sid in persona_scores if script_meta.get(sid, {}).get("is_adversarial", False)]

    nat_turns, nat_means, nat_ci_lo, nat_ci_hi, _, _ = compute_mean_timeseries(persona_scores, nat_ids)
    adv_turns, adv_means, adv_ci_lo, adv_ci_hi, _, _ = compute_mean_timeseries(persona_scores, adv_ids)

    nat_tstar = find_tstar(nat_turns, nat_means)
    adv_tstar = find_tstar(adv_turns, adv_means)

    return {
        "naturalistic": {"n": len(nat_ids), "tstar": nat_tstar,
                         "turns": nat_turns.tolist(), "means": nat_means.tolist()},
        "adversarial": {"n": len(adv_ids), "tstar": adv_tstar,
                        "turns": adv_turns.tolist(), "means": adv_means.tolist()},
        "tstar_diff": (nat_tstar or 40) - (adv_tstar or 40),
    }


# ── 7.6 Failure Mode Taxonomy ─────────────────────────────────────────────

def failure_taxonomy(scores: list[dict]):
    """Categorize failures (score <= 2) by failure mode."""
    taxonomy = {
        "trait_drift": {"count": 0, "examples": []},
        "episodic_fabrication": {"count": 0, "examples": []},
        "capability_overstatement": {"count": 0, "examples": []},
        "register_shift": {"count": 0, "examples": []},
    }

    dim_to_mode = {
        "T": "trait_drift",
        "E": "episodic_fabrication",
        "C": "capability_overstatement",
        "S": "register_shift",
    }

    for r in scores:
        if r["score"] <= 2:
            mode = dim_to_mode.get(r["dimension"])
            if mode:
                taxonomy[mode]["count"] += 1
                if len(taxonomy[mode]["examples"]) < 5:
                    taxonomy[mode]["examples"].append({
                        "script_id": r["script_id"],
                        "turn": r["turn"],
                        "score": r["score"],
                        "probe": r["probe"][:80],
                        "response": r["response"][:120],
                        "reason": r.get("reason", "")[:100],
                    })

    return taxonomy


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_timeseries(turns, means, ci_lo, ci_hi, tstar, title, filename,
                    adv_data=None):
    """Plot PersonaScore time series with CIs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(turns, ci_lo, ci_hi, alpha=0.2, color="steelblue", label="95% CI")
    ax.plot(turns, means, "o-", color="steelblue", linewidth=2, markersize=8, label="Mean PersonaScore")

    if adv_data:
        adv_turns = np.array(adv_data["turns"])
        adv_means = np.array(adv_data["means"])
        ax.plot(adv_turns, adv_means, "s--", color="crimson", linewidth=2, markersize=7,
                label="Adversarial mean")

    ax.axhline(y=THRESHOLD, color="red", linestyle=":", linewidth=1.5, alpha=0.7,
               label=f"T* threshold ({THRESHOLD})")

    if tstar:
        ax.axvline(x=tstar, color="orange", linestyle="--", linewidth=1.5, alpha=0.7,
                   label=f"T* = turn {tstar}")

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("PersonaScore (1-5)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(1, 5.2)
    ax.set_xticks(PROBE_TURNS)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{filename}.png", dpi=150)
    plt.savefig(RESULTS_DIR / f"{filename}.svg")
    plt.savefig(RESULTS_DIR / f"{filename}.pdf")
    plt.close()


def plot_dimensions(dim_tstar, filename="dimension_timeseries"):
    """Plot per-dimension time series."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    dim_names = {"T": "Trait Self-Description", "E": "Episodic Recall",
                 "C": "Capability/Limitation", "S": "Style/Register"}
    colors = {"T": "steelblue", "E": "forestgreen", "C": "darkorange", "S": "purple"}

    for ax, dim in zip(axes.flat, ["T", "E", "C", "S"]):
        data = dim_tstar[dim]
        turns = sorted(data["means"].keys())
        means = [data["means"][t] for t in turns]

        ax.plot(turns, means, "o-", color=colors[dim], linewidth=2, markersize=7)
        ax.axhline(y=THRESHOLD, color="red", linestyle=":", linewidth=1, alpha=0.7)
        if data["tstar"]:
            ax.axvline(x=data["tstar"], color="orange", linestyle="--", alpha=0.7)
            ax.set_title(f"{dim_names[dim]} (T*={data['tstar']})", fontsize=11)
        else:
            ax.set_title(f"{dim_names[dim]} (T*=none)", fontsize=11)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Score (1-5)")
        ax.set_ylim(1, 5.2)
        ax.set_xticks(PROBE_TURNS)
        ax.grid(True, alpha=0.3)

    plt.suptitle("PersonaScore by Dimension", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{filename}.png", dpi=150)
    plt.savefig(RESULTS_DIR / f"{filename}.svg")
    plt.savefig(RESULTS_DIR / f"{filename}.pdf")
    plt.close()


def plot_model_fits(turns, means, fits, filename="degradation_fits"):
    """Plot degradation model fits."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(turns, means, "ko", markersize=10, label="Observed", zorder=5)

    colors = {"linear": "steelblue", "step": "crimson", "exponential": "forestgreen",
              "piecewise": "darkorange"}

    for fit in fits:
        name = fit["name"]
        aic = fit["aic"]
        pred = fit.get("predicted", [])
        if len(pred) == len(turns):
            ax.plot(turns, pred, "--", color=colors.get(name, "gray"), linewidth=2,
                    label=f"{name} (AIC={aic:.1f})")

    ax.axhline(y=THRESHOLD, color="red", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("PersonaScore", fontsize=12)
    ax.set_title("Degradation Model Fits", fontsize=14)
    ax.set_ylim(1, 5.2)
    ax.set_xticks(PROBE_TURNS)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{filename}.png", dpi=150)
    plt.savefig(RESULTS_DIR / f"{filename}.pdf")
    plt.close()


# ── Report Generation ─────────────────────────────────────────────────────

def generate_report(tstar, dim_tstar, fits, h4, adv_comp, taxonomy, turns, means,
                    stds, ns):
    """Generate the summary markdown report."""
    best_fit = min(fits, key=lambda f: f["aic"])

    lines = [
        "# CHA Experiment 1 — Results Summary",
        "",
        "## 1. Primary Result: T*",
        "",
        f"**T* = {tstar if tstar else 'Not reached (>40)'}**",
        "",
    ]

    if tstar and tstar >= 30:
        lines.append("Interpretation: Phi-4-mini is more robust than expected. "
                      "SCI can use conservative refresh strategy (every 25 turns).")
    elif tstar and 15 <= tstar < 30:
        lines.append(f"Interpretation: Expected range. SCI must refresh at T*-5 = turn {tstar-5}. "
                      "Token budget needs careful design.")
    elif tstar and tstar < 15:
        lines.append("Interpretation: Model is less robust than expected. "
                      "SCI needs aggressive refresh every 10 turns.")
    else:
        lines.append("Interpretation: PersonaScore never dropped below 3.5 within 40 turns. "
                      "Phi-4-mini maintains persona well at this conversation length.")

    lines.extend([
        "",
        "## 2. PersonaScore Time Series",
        "",
        "| Turn | Mean | Std | 95% CI | n |",
        "|------|------|-----|--------|---|",
    ])
    for t, m, s, n in zip(turns, means, stds, ns):
        ci = 1.96 * s / np.sqrt(n) if n > 1 else 0
        lines.append(f"| {int(t)} | {m:.2f} | {s:.2f} | [{m-ci:.2f}, {m+ci:.2f}] | {n} |")

    lines.extend([
        "",
        "## 3. Degradation Profile",
        "",
        "| Model | AIC | Parameters |",
        "|-------|-----|------------|",
    ])
    for fit in sorted(fits, key=lambda f: f["aic"]):
        params = fit.get("params", {})
        param_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in params.items())
        marker = " **← best**" if fit["name"] == best_fit["name"] else ""
        lines.append(f"| {fit['name']} | {fit['aic']:.1f} | {param_str} |{marker}")

    lines.extend([
        "",
        f"**Best fit: {best_fit['name']}**",
        "",
    ])

    if best_fit["name"] == "linear":
        lines.append("Implication: Continuous context crowding → SCI should use "
                      "sliding window compression.")
    elif best_fit["name"] == "step":
        bp = best_fit.get("params", {}).get("breakpoint", "?")
        lines.append(f"Implication: Sudden capacity failure at turn {bp} → "
                      "SCI needs hard trigger at 60% context fill.")
    elif best_fit["name"] == "exponential":
        lines.append("Implication: Rapid early decline → persona needs front-loading "
                      "and early reinforcement.")
    elif best_fit["name"] == "piecewise":
        t0 = best_fit.get("params", {}).get("T0", "?")
        lines.append(f"Implication: Stable until turn {t0}, then declining → "
                      f"SCI must intervene before turn {t0}.")

    lines.extend([
        "",
        "## 4. Dimension T* Ordering",
        "",
        "| Dimension | T* | Interpretation |",
        "|-----------|-----|----------------|",
    ])
    dim_names = {"T": "Trait", "E": "Episodic", "C": "Capability", "S": "Style"}
    dim_implications = {
        "T": "Increase trait section token budget",
        "E": "Compress/remove episodic section; move to dedicated retrieval",
        "C": "Move capabilities/limitations to separate persistent constraint section",
        "S": "Add style anchoring phrases to self_beliefs section",
    }
    for dim in sorted(dim_tstar.keys(), key=lambda d: dim_tstar[d]["tstar"] or 999):
        ts = dim_tstar[dim]["tstar"]
        ts_str = str(ts) if ts else ">40"
        lines.append(f"| {dim_names[dim]} ({dim}) | {ts_str} | {dim_implications[dim]} |")

    degradation_order = [d for d in sorted(dim_tstar.keys(),
                         key=lambda d: dim_tstar[d]["tstar"] or 999)
                         if dim_tstar[d]["tstar"] is not None]
    if degradation_order:
        first = dim_names[degradation_order[0]]
        lines.append(f"\n**First to degrade: {first}** — prioritize in SCI compression budget.")

    lines.extend([
        "",
        "## 5. H4 Correlation Analysis",
        "",
        f"- Context fill correlation: r = {h4['context_fill_correlation']['mean_r']:.3f} "
        f"(±{h4['context_fill_correlation']['std_r']:.3f})",
        f"- Turn count correlation: r = {h4['turn_count_correlation']['mean_r']:.3f} "
        f"(±{h4['turn_count_correlation']['std_r']:.3f})",
        f"- **H4 {'supported' if h4.get('h4_supported') else 'not supported'}**: "
        f"{h4.get('interpretation', 'N/A')}",
    ])

    lines.extend([
        "",
        "## 6. Adversarial vs. Naturalistic",
        "",
        f"- Naturalistic T*: {adv_comp['naturalistic']['tstar'] or '>40'} "
        f"(n={adv_comp['naturalistic']['n']})",
        f"- Adversarial T*: {adv_comp['adversarial']['tstar'] or '>40'} "
        f"(n={adv_comp['adversarial']['n']})",
        f"- Difference: {adv_comp['tstar_diff']} turns",
    ])
    if adv_comp["tstar_diff"] > 5:
        lines.append("\n**Adversarial fragility detected.** Naturalistic T* is optimistic. "
                      "Add adversarial robustness training to LoRA fine-tuning data.")
    elif adv_comp["tstar_diff"] <= 5:
        lines.append("\nAdversarial probing does not significantly accelerate degradation.")

    lines.extend([
        "",
        "## 7. Failure Mode Taxonomy",
        "",
        "| Failure Mode | Count | SCI Design Implication |",
        "|-------------|-------|------------------------|",
    ])
    mode_names = {
        "trait_drift": "Trait drift",
        "episodic_fabrication": "Episodic fabrication",
        "capability_overstatement": "Capability overstatement",
        "register_shift": "Register shift",
    }
    mode_implications = {
        "trait_drift": "Increase trait section token budget",
        "episodic_fabrication": "Compress/remove episodic section",
        "capability_overstatement": "Add explicit constraint reinforcement",
        "register_shift": "Add style anchoring phrases",
    }
    for mode, data in taxonomy.items():
        lines.append(f"| {mode_names[mode]} | {data['count']} | {mode_implications[mode]} |")

    lines.extend([
        "",
        "## 8. Decision Rules (Populated)",
        "",
        "| Result | Observed | SMC Design Change |",
        "|--------|----------|-------------------|",
    ])

    if tstar and tstar >= 30:
        lines.append(f"| T* >= 30 | T*={tstar} ✓ | SCI: conservative refresh every 25 turns |")
    elif tstar and 15 <= tstar < 30:
        lines.append(f"| T* 15-29 | T*={tstar} ✓ | SCI: refresh at turn {tstar-5}; careful token budget |")
    elif tstar and tstar < 15:
        lines.append(f"| T* < 15 | T*={tstar} ✓ | SCI: aggressive refresh every 10 turns |")
    else:
        lines.append("| T* > 40 | Not reached | SCI: minimal intervention needed at 40 turns |")

    lines.append(f"| Degradation type | {best_fit['name']} | See Section 3 implications |")

    if degradation_order:
        first_dim = degradation_order[0]
        lines.append(f"| First dimension to degrade | {dim_names[first_dim]} | "
                     f"{dim_implications[first_dim]} |")

    lines.append(f"| H4 (context fill) | {'Supported' if h4.get('h4_supported') else 'Not supported'} | "
                 f"{h4.get('interpretation', 'N/A')[:60]} |")

    if adv_comp["tstar_diff"] > 5:
        lines.append("| Adversarial fragility | Yes | Add adversarial robustness to LoRA data |")
    else:
        lines.append("| Adversarial fragility | No | No special adversarial training needed |")

    lines.extend(["", "---",
                  f"*Generated {len(ALL_IDS)} conversations, {len(PROBE_TURNS)} probe turns each.*"])

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CHA Experiment 1 Analysis")
    parser.add_argument("--model", type=str, default=None,
                        help="Subject model name (e.g. qwen2.5:7b) to find model-specific logs dir")
    args = parser.parse_args()

    # Use model-specific dirs if --model is provided
    global LOGS_DIR, RESULTS_DIR
    if args.model and args.model != "phi4-mini":
        safe_name = args.model.replace(":", "_").replace("/", "_")
        LOGS_DIR = BASE_DIR / f"logs_{safe_name}"
        RESULTS_DIR = BASE_DIR / f"results_{safe_name}"
    RESULTS_DIR.mkdir(exist_ok=True)

    print("CHA Experiment 1 — Analysis")
    print(f"Logs dir: {LOGS_DIR}")
    print(f"Results dir: {RESULTS_DIR}")
    print("=" * 60)

    # Load data
    scores = load_scores()
    context = load_context()
    script_meta = load_script_metadata()

    if not scores:
        print("ERROR: No score data found in logs/")
        sys.exit(1)

    print(f"Loaded {len(scores)} score records, {len(context)} context records")
    print(f"Scripts with metadata: {len(script_meta)}")

    # 7.1 Compute PersonaScores and find T*
    persona_scores = compute_persona_scores(scores)
    turns, means, ci_lo, ci_hi, stds, ns = compute_mean_timeseries(persona_scores)
    tstar = find_tstar(turns, means)
    print(f"\nT* = {tstar if tstar else 'Not reached (>40)'}")

    # 7.2 Fit degradation models
    fits = []
    if len(turns) >= 3:
        fits.append(fit_linear(turns, means))
        fits.append(fit_step(turns, means))
        fits.append(fit_exponential(turns, means))
        fits.append(fit_piecewise(turns, means))
        best = min(fits, key=lambda f: f["aic"])
        print(f"Best degradation model: {best['name']} (AIC={best['aic']:.1f})")

    # 7.3 Dimension T*
    dim_tstar = compute_dimension_tstar(persona_scores)
    for dim in ["T", "E", "C", "S"]:
        ts = dim_tstar[dim]["tstar"]
        print(f"  T*_{dim} = {ts if ts else '>40'}")

    # 7.4 H4 correlation
    h4 = h4_correlation(scores, context)
    print(f"\nH4: ctx_fill r={h4['context_fill_correlation']['mean_r']:.3f}, "
          f"turn r={h4['turn_count_correlation']['mean_r']:.3f}")

    # 7.5 Adversarial comparison
    adv_comp = adversarial_comparison(persona_scores, script_meta)
    print(f"Adversarial T*: {adv_comp['adversarial']['tstar'] or '>40'}, "
          f"Naturalistic T*: {adv_comp['naturalistic']['tstar'] or '>40'}")

    # 7.6 Failure taxonomy
    taxonomy = failure_taxonomy(scores)
    total_failures = sum(t["count"] for t in taxonomy.values())
    print(f"Total failures (score<=2): {total_failures}")

    # Generate plots
    print("\nGenerating plots...")
    plot_timeseries(turns, means, ci_lo, ci_hi, tstar,
                    "PersonaScore Time Series (All Conversations)",
                    "persona_score_timeseries",
                    adv_data=adv_comp.get("adversarial"))
    plot_dimensions(dim_tstar)
    if fits:
        plot_model_fits(turns, means, fits)

    # Generate report
    print("Generating report...")
    report = generate_report(tstar, dim_tstar, fits, h4, adv_comp, taxonomy,
                             turns, means, stds, ns)
    with open(RESULTS_DIR / "summary_report.md", "w") as f:
        f.write(report)

    # Save raw analysis data
    analysis_data = {
        "tstar": tstar,
        "timeseries": {"turns": turns.tolist(), "means": means.tolist(),
                       "ci_lower": ci_lo.tolist(), "ci_upper": ci_hi.tolist()},
        "degradation_fits": [{k: v for k, v in f.items() if k != "predicted"}
                             for f in fits],
        "dimension_tstar": {d: {"tstar": v["tstar"],
                                "means": {str(k): float(val) for k, val in v["means"].items()}}
                            for d, v in dim_tstar.items()},
        "h4_analysis": h4,
        "adversarial_comparison": adv_comp,
        "failure_taxonomy": {k: {"count": v["count"], "examples": v["examples"]}
                             for k, v in taxonomy.items()},
    }
    with open(RESULTS_DIR / "analysis_data.json", "w") as f:
        json.dump(analysis_data, f, indent=2)

    print("\n" + "=" * 60)
    print("DELIVERABLES saved to results/:")
    print(f"  - persona_score_timeseries.png/.svg/.pdf")
    print(f"  - dimension_timeseries.png/.svg/.pdf")
    print(f"  - degradation_fits.png/.pdf")
    print(f"  - summary_report.md")
    print(f"  - analysis_data.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
