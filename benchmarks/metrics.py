"""Pure metric-computation helpers for benchmark results."""


def count_adapter_switches(steps: list[dict]) -> int:
    """Count adapter changes between consecutive steps in execution order.

    Args:
        steps: Step dicts in execution order, each with an "adapter" key
            (as returned by the orchestrator's TaskResult.steps).

    Returns:
        Number of times the adapter differs from the previous step.
    """
    switches = 0
    for prev, curr in zip(steps, steps[1:]):
        if prev["adapter"] != curr["adapter"]:
            switches += 1
    return switches


def is_subsequence(expected: list[str], actual: list[str]) -> bool:
    """Check whether expected tags appear in actual, in the same relative order.

    Consecutive duplicate adapters in actual are collapsed before matching,
    so repeated calls to the same adapter don't affect order correctness.

    Args:
        expected: Expected adapter tags, in required order.
        actual: Actual adapters used, in execution order.

    Returns:
        True if expected is a subsequence of the deduplicated actual sequence.
    """
    collapsed = [a for i, a in enumerate(actual) if i == 0 or a != actual[i - 1]]
    it = iter(collapsed)
    return all(tag in it for tag in expected)


def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence between two tag sequences.

    Standard O(len(a) * len(b)) DP. Order-sensitive but gap-tolerant: unlike
    is_subsequence (all-or-nothing), this gives partial credit when only part
    of the expected order is respected.

    Args:
        a: First sequence of tags.
        b: Second sequence of tags.

    Returns:
        Length of the longest common subsequence.
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def score_plan(actual_adapters: list[str], expected_adapters: list[str]) -> dict:
    """Score how well actual_adapters matches expected_adapters.

    Args:
        actual_adapters: Adapters used by the orchestrator, in execution order.
        expected_adapters: Adapters the task expects, in required order.

    Returns:
        Dict with precision, recall, f1 (set-based over adapter tags),
        order_correct (whether expected_adapters is a subsequence of
        actual_adapters — all-or-nothing), and order_score (LCS-based, graded
        in [0.0, 1.0]: len(LCS(expected, actual)) / len(expected). Unlike
        order_correct, this distinguishes "right adapters, reversed order"
        (low but nonzero score) from "right adapters, one swap in a 4-step
        plan" (high score) — both would tie at order_correct=False.
        1.0 when expected_adapters is empty (nothing to get wrong).
    """
    actual_set = set(actual_adapters)
    expected_set = set(expected_adapters)
    inter = actual_set & expected_set

    precision = len(inter) / len(actual_set) if actual_set else 0.0
    recall = len(inter) / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    order_score = (
        lcs_length(expected_adapters, actual_adapters) / len(expected_adapters)
        if expected_adapters else 1.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "order_correct": is_subsequence(expected_adapters, actual_adapters),
        "order_score": order_score,
    }


def summarize(records: list[dict]) -> dict:
    """Aggregate per-task benchmark records into summary statistics.

    Only records with success=True contribute to the averages.

    Args:
        records: Per-task result records. Successful records must include
            "latency_s", "adapter_switches", and "replanned".

    Returns:
        Dict with n_tasks, n_success, avg_latency_s, avg_switches, and
        replan_rate. Averages are 0.0 when there are no successful records.
    """
    n_tasks = len(records)
    successful = [r for r in records if r["success"]]
    n_success = len(successful)

    if n_success == 0:
        return {
            "n_tasks": n_tasks,
            "n_success": 0,
            "avg_latency_s": 0.0,
            "avg_switches": 0.0,
            "replan_rate": 0.0,
            "avg_plan_precision": 0.0,
            "avg_plan_recall": 0.0,
            "avg_plan_f1": 0.0,
            "plan_order_correct_rate": 0.0,
            "avg_plan_order_score": 0.0,
        }

    avg_latency_s = sum(r["latency_s"] for r in successful) / n_success
    avg_switches = sum(r["adapter_switches"] for r in successful) / n_success
    replan_rate = sum(1 for r in successful if r["replanned"]) / n_success
    avg_plan_precision = sum(r["plan_precision"] for r in successful) / n_success
    avg_plan_recall = sum(r["plan_recall"] for r in successful) / n_success
    avg_plan_f1 = sum(r["plan_f1"] for r in successful) / n_success
    plan_order_correct_rate = sum(1 for r in successful if r["plan_order_correct"]) / n_success
    avg_plan_order_score = sum(r["plan_order_score"] for r in successful) / n_success

    return {
        "n_tasks": n_tasks,
        "n_success": n_success,
        "avg_latency_s": avg_latency_s,
        "avg_switches": avg_switches,
        "replan_rate": replan_rate,
        "avg_plan_precision": avg_plan_precision,
        "avg_plan_recall": avg_plan_recall,
        "avg_plan_f1": avg_plan_f1,
        "plan_order_correct_rate": plan_order_correct_rate,
        "avg_plan_order_score": avg_plan_order_score,
    }
