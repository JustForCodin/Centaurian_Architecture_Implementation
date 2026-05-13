#!/usr/bin/env python3
"""
Phase 3: Inter-rater reliability check.

Re-scores a random 20% of probe responses using a secondary judge model
and computes Cohen's kappa per dimension and overall.
"""

import json
import os
import random
import sys
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
os.environ["ANTHROPIC_API_KEY"] = os.environ.get("CHA_EXPERIMENT_SONNET_KEY", "")

from experiment_runner import llm_judge, JUDGE_SECONDARY_MODEL

BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"  # may be overridden by --model arg


def cohens_kappa(ratings1: list[int], ratings2: list[int], weighted: bool = True,
                 categories: list[int] | None = None) -> float:
    """Compute Cohen's kappa for two lists of ordinal ratings.

    Args:
        weighted: If True, use quadratic weights (standard for ordinal scales).
                  If False, use unweighted (nominal) kappa.
        categories: Possible rating values. Defaults to [1,2,3,4,5].
    """
    assert len(ratings1) == len(ratings2)
    n = len(ratings1)
    if n == 0:
        return 0.0

    if categories is None:
        categories = list(range(1, 6))
    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    # Build confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for r1, r2 in zip(ratings1, ratings2):
        i = cat_idx.get(r1, 2)  # default to middle
        j = cat_idx.get(r2, 2)
        matrix[i][j] += 1

    # Build weight matrix: quadratic weights w_ij = 1 - (i-j)^2 / (k-1)^2
    # Perfect agreement = 1, maximum disagreement = 0
    if weighted:
        max_dist_sq = (k - 1) ** 2
        weights = [[1.0 - (i - j) ** 2 / max_dist_sq for j in range(k)] for i in range(k)]
    else:
        weights = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]

    row_sums = [sum(matrix[i]) for i in range(k)]
    col_sums = [sum(matrix[i][j] for i in range(k)) for j in range(k)]

    # Observed weighted agreement
    p_o = sum(weights[i][j] * matrix[i][j] for i in range(k) for j in range(k)) / n

    # Expected weighted agreement by chance
    p_e = sum(weights[i][j] * row_sums[i] * col_sums[j] for i in range(k) for j in range(k)) / (n * n)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--sample-pct", type=float, default=0.40)
    parser.add_argument("--model", type=str, default=None,
                        help="Subject model name (e.g. qwen2.5:7b) to find model-specific logs dir")
    args = parser.parse_args()

    # Use model-specific logs dir if --model is provided
    global LOGS_DIR
    if args.model and args.model != "phi4-mini":
        safe_name = args.model.replace(":", "_").replace("/", "_")
        LOGS_DIR = BASE_DIR / f"logs_{safe_name}"

    # Collect all primary judge scores
    all_records = []
    for sid in range(args.start, args.end + 1):
        scores_path = LOGS_DIR / f"scores_{sid:03d}.jsonl"
        if not scores_path.exists():
            continue
        with open(scores_path) as f:
            for line in f:
                record = json.loads(line)
                all_records.append(record)

    if not all_records:
        print("No score records found.")
        sys.exit(1)

    print(f"Total probe scores: {len(all_records)}")

    # Random sample
    rng = random.Random(42)
    sample_size = max(1, int(len(all_records) * args.sample_pct))
    sample = rng.sample(all_records, sample_size)
    print(f"Sample size ({args.sample_pct:.0%}): {sample_size}")
    print(f"Secondary judge: {JUDGE_SECONDARY_MODEL}")
    print("=" * 60)

    # Re-score with secondary judge
    primary_scores = defaultdict(list)
    secondary_scores = defaultdict(list)
    disagreements = []

    for i, record in enumerate(sample):
        dim = record["dimension"]
        probe = record["probe"]
        response = record["response"]
        primary_score = record["score"]

        print(f"  Re-scoring {i+1}/{sample_size} (script {record['script_id']:03d}, "
              f"turn {record['turn']}, dim {dim})...")

        secondary_score, secondary_reason = llm_judge(
            probe, response, dim, model=JUDGE_SECONDARY_MODEL
        )

        primary_scores[dim].append(primary_score)
        secondary_scores[dim].append(secondary_score)
        primary_scores["overall"].append(primary_score)
        secondary_scores["overall"].append(secondary_score)

        diff = abs(primary_score - secondary_score)
        if diff >= 1:
            disagreements.append({
                "script_id": record["script_id"],
                "turn": record["turn"],
                "dimension": dim,
                "probe": probe[:60],
                "response": response[:80],
                "primary_score": primary_score,
                "primary_reason": record.get("reason", ""),
                "secondary_score": secondary_score,
                "secondary_reason": secondary_reason,
                "diff": diff,
            })

    # Compute kappa per dimension and overall
    # Binary kappa: score >= 4 = "consistent" (1), score <= 3 = "inconsistent" (0)
    # This aligns with the PersonaScore threshold of 3.5
    def binarize(scores):
        return [1 if s >= 4 else 0 for s in scores]

    print("\n" + "=" * 60)
    print("INTER-RATER RELIABILITY")
    print("=" * 60)

    results = {}
    all_pass = True
    for dim in ["T", "E", "C", "S", "overall"]:
        if dim in primary_scores and primary_scores[dim]:
            kappa_w = cohens_kappa(primary_scores[dim], secondary_scores[dim], weighted=True)
            kappa_u = cohens_kappa(primary_scores[dim], secondary_scores[dim], weighted=False)
            # Binary kappa on the decision-relevant split (≥4 vs ≤3)
            bin_p = binarize(primary_scores[dim])
            bin_s = binarize(secondary_scores[dim])
            kappa_bin = cohens_kappa(bin_p, bin_s, weighted=False, categories=[0, 1])
            n = len(primary_scores[dim])
            results[dim] = {
                "kappa_weighted": kappa_w,
                "kappa_unweighted": kappa_u,
                "kappa_binary": kappa_bin,
                "n": n,
            }
            status = "✓ PASS" if kappa_bin >= 0.70 else "✗ FAIL"
            if kappa_bin < 0.70:
                all_pass = False
            print(f"  {dim:8s}: κ_bin = {kappa_bin:.3f}  κ_w = {kappa_w:.3f}  κ_u = {kappa_u:.3f}  (n={n})  {status}")

    # Save results
    results_path = LOGS_DIR / "interrater_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "results": results,
            "disagreements": sorted(disagreements, key=lambda x: -x["diff"]),
            "all_pass": all_pass,
        }, f, indent=2)

    if not all_pass:
        print(f"\n{'=' * 60}")
        print("LARGEST DISAGREEMENTS (for failing dimensions)")
        print("=" * 60)
        for dim in ["T", "E", "C", "S"]:
            if dim in results and results[dim]["kappa_binary"] < 0.70:
                dim_disagree = [d for d in disagreements if d["dimension"] == dim]
                dim_disagree.sort(key=lambda x: -x["diff"])
                print(f"\n  Dimension {dim} (κ_bin={results[dim]['kappa_binary']:.3f}):")
                for d in dim_disagree[:5]:
                    print(f"    Script {d['script_id']:03d} T{d['turn']:02d}: "
                          f"primary={d['primary_score']} secondary={d['secondary_score']} "
                          f"diff={d['diff']}")
                    print(f"      Probe: {d['probe']}")
                    print(f"      Primary reason: {d['primary_reason'][:80]}")
                    print(f"      Secondary reason: {d['secondary_reason'][:80]}")

    print(f"\nResults saved to {results_path}")
    print(f"\nOverall: {'ALL DIMENSIONS PASS (κ ≥ 0.70)' if all_pass else 'SOME DIMENSIONS BELOW THRESHOLD'}")


if __name__ == "__main__":
    main()
