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
        }

    avg_latency_s = sum(r["latency_s"] for r in successful) / n_success
    avg_switches = sum(r["adapter_switches"] for r in successful) / n_success
    replan_rate = sum(1 for r in successful if r["replanned"]) / n_success

    return {
        "n_tasks": n_tasks,
        "n_success": n_success,
        "avg_latency_s": avg_latency_s,
        "avg_switches": avg_switches,
        "replan_rate": replan_rate,
    }
