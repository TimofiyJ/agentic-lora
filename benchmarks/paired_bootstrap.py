"""Paired bootstrap significance test between two quality benchmark results
(e.g. B0 vs OURS), matched by task id.

Run: python -m benchmarks.paired_bootstrap <b0_results.json> <ours_results.json>
"""

import argparse
import json
import random
from pathlib import Path


def paired_bootstrap(
    scores_a: list[float], scores_b: list[float], n_resamples: int = 10000, seed: int = 0
) -> dict:
    """Bootstrap the mean paired difference (b - a) and a one-sided p-value
    for "b is not better than a".

    Args:
        scores_a: Baseline per-task scores (e.g. B0), same length/order as scores_b.
        scores_b: Treatment per-task scores (e.g. OURS).
        n_resamples: Number of bootstrap resamples.
        seed: RNG seed, for reproducibility.

    Returns:
        {"mean_diff": float, "ci_low": float, "ci_high": float, "p_value": float}
        ci_low/ci_high are the 2.5th/97.5th percentile of the resampled means
        (95% CI). p_value is the fraction of resamples where the mean
        difference is <= 0 (evidence against "b improves on a").
    """
    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    n = len(diffs)
    mean_diff = sum(diffs) / n

    rng = random.Random(seed)
    resampled_means = []
    for _ in range(n_resamples):
        resample = [diffs[rng.randrange(n)] for _ in range(n)]
        resampled_means.append(sum(resample) / n)

    resampled_means.sort()
    ci_low = resampled_means[int(0.025 * n_resamples)]
    ci_high = resampled_means[int(0.975 * n_resamples) - 1]
    p_value = sum(1 for m in resampled_means if m <= 0) / n_resamples

    return {
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }


def load_scores_by_task_id(path: str | Path, domain: str) -> dict[str, float]:
    """Load {task_id: score} for every task in domain from a quality results JSON."""
    data = json.loads(Path(path).read_text())
    return {
        task["id"]: task["score"]
        for task in data["tasks"]
        if task["domain"] == domain
    }


def compare_domain(
    b0_path: str | Path, ours_path: str | Path, domain: str,
    n_resamples: int = 10000, seed: int = 0,
) -> dict:
    """Load both result files, pair scores by task id for domain, bootstrap the diff.

    Returns paired_bootstrap's dict plus "n_paired_tasks" (how many task ids
    were present in both files -- tasks missing from either side are dropped).
    """
    b0_scores = load_scores_by_task_id(b0_path, domain)
    ours_scores = load_scores_by_task_id(ours_path, domain)
    shared_ids = sorted(set(b0_scores) & set(ours_scores))

    scores_a = [b0_scores[task_id] for task_id in shared_ids]
    scores_b = [ours_scores[task_id] for task_id in shared_ids]

    result = paired_bootstrap(scores_a, scores_b, n_resamples=n_resamples, seed=seed)
    result["n_paired_tasks"] = len(shared_ids)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap: B0 vs OURS quality benchmark.")
    parser.add_argument("b0_results")
    parser.add_argument("ours_results")
    parser.add_argument("--domains", nargs="+", default=["sql", "legal", "medical"])
    parser.add_argument("--n-resamples", type=int, default=10000)
    args = parser.parse_args()

    for domain in args.domains:
        result = compare_domain(args.b0_results, args.ours_results, domain, n_resamples=args.n_resamples)
        significant = "significant (p<0.05)" if result["p_value"] < 0.05 else "not significant"
        print(
            f"{domain}: n={result['n_paired_tasks']} mean_diff={result['mean_diff']:.4f} "
            f"95% CI=[{result['ci_low']:.4f}, {result['ci_high']:.4f}] "
            f"p={result['p_value']:.4f} ({significant})"
        )


if __name__ == "__main__":
    main()
