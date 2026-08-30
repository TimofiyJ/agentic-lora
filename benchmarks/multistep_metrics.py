"""Alignment and aggregate metrics for the multi-step cross-domain benchmark."""


def align_steps(oracle_steps: list[dict], actual_steps: list[dict]) -> list[dict]:
    """Align oracle steps to actual orchestrator steps by adapter match.

    For each oracle step, in order, finds the first not-yet-consumed actual
    step whose adapter matches the oracle step's oracle_adapter. This keeps
    step-output scoring separate from adapter-routing scoring: an oracle
    step is only ever compared against an actual step that used the correct
    adapter.

    Args:
        oracle_steps: Oracle step dicts, each with "oracle_adapter".
        actual_steps: Actual steps from the orchestrator's TaskResult.steps,
            each with an "adapter" key, in execution order.

    Returns:
        One entry per oracle step, in oracle order:
        {"oracle_step": dict, "actual_step": dict | None}.
        actual_step is None if no matching adapter was found.
    """
    remaining = list(actual_steps)
    aligned = []
    for oracle_step in oracle_steps:
        match = None
        for i, actual_step in enumerate(remaining):
            if actual_step["adapter"] == oracle_step["oracle_adapter"]:
                match = remaining.pop(i)
                break
        aligned.append({"oracle_step": oracle_step, "actual_step": match})
    return aligned


def task_completion(aligned: list[dict]) -> bool:
    """Whether every oracle step matched an actual step AND passed.

    Args:
        aligned: Output of align_steps, with "score"/"passed" added per entry.

    Returns:
        True iff every entry has a non-None actual_step and passed=True.
    """
    return all(entry["actual_step"] is not None and entry["passed"] for entry in aligned)


def step_accuracy(aligned: list[dict]) -> float:
    """Mean per-step score across all oracle steps; unmatched steps score 0.0.

    Args:
        aligned: Output of align_steps, with "score" added per entry.

    Returns:
        Mean score in [0.0, 1.0], or 0.0 if aligned is empty.
    """
    if not aligned:
        return 0.0
    return sum(entry["score"] for entry in aligned) / len(aligned)


def error_propagation_rate(aligned: list[dict]) -> float | None:
    """Fraction of steps after the first failure that also failed.

    A step "fails" if unmatched (actual_step is None) or passed=False.

    Args:
        aligned: Output of align_steps, with "passed" added per entry.

    Returns:
        Fraction in [0.0, 1.0] of downstream steps (after the first failure)
        that also failed. None if there were zero failures (nothing to
        propagate). 0.0 if the only failure is the last step (no downstream
        steps to check).
    """
    failed = [entry["actual_step"] is None or not entry["passed"] for entry in aligned]
    if not any(failed):
        return None
    first_failure = failed.index(True)
    downstream = failed[first_failure + 1:]
    if not downstream:
        return 0.0
    return sum(downstream) / len(downstream)


def summarize_multistep(records: list[dict]) -> dict:
    """Aggregate per-task multistep records into summary statistics.

    Only records with success=True contribute to averages.
    avg_error_propagation_rate averages only over successful records whose
    error_propagation_rate is not None (i.e. records that had a failure).

    Args:
        records: Per-task result records. Successful records must include
            "latency_s", "adapter_switches", "replanned", "plan_precision",
            "plan_recall", "plan_f1", "plan_order_correct", "plan_order_score",
            "task_completion", "step_accuracy", "coverage",
            "error_propagation_rate".

    Returns:
        Dict with n_tasks, n_success, task_completion_rate,
        avg_step_accuracy, avg_plan_precision, avg_plan_recall,
        avg_plan_f1, plan_order_correct_rate, avg_coverage,
        avg_error_propagation_rate, avg_latency_s, avg_switches,
        replan_rate. All averages are 0.0 when there are no successful
        (or, for propagation, no failing) records.
    """
    n_tasks = len(records)
    successful = [r for r in records if r["success"]]
    n_success = len(successful)

    if n_success == 0:
        return {
            "n_tasks": n_tasks,
            "n_success": 0,
            "task_completion_rate": 0.0,
            "avg_step_accuracy": 0.0,
            "avg_plan_precision": 0.0,
            "avg_plan_recall": 0.0,
            "avg_plan_f1": 0.0,
            "plan_order_correct_rate": 0.0,
            "avg_plan_order_score": 0.0,
            "avg_coverage": 0.0,
            "avg_error_propagation_rate": 0.0,
            "avg_latency_s": 0.0,
            "avg_switches": 0.0,
            "replan_rate": 0.0,
        }

    propagation_values = [
        r["error_propagation_rate"] for r in successful if r["error_propagation_rate"] is not None
    ]

    return {
        "n_tasks": n_tasks,
        "n_success": n_success,
        "task_completion_rate": sum(1 for r in successful if r["task_completion"]) / n_success,
        "avg_step_accuracy": sum(r["step_accuracy"] for r in successful) / n_success,
        "avg_plan_precision": sum(r["plan_precision"] for r in successful) / n_success,
        "avg_plan_recall": sum(r["plan_recall"] for r in successful) / n_success,
        "avg_plan_f1": sum(r["plan_f1"] for r in successful) / n_success,
        "plan_order_correct_rate": sum(1 for r in successful if r["plan_order_correct"]) / n_success,
        "avg_plan_order_score": sum(r["plan_order_score"] for r in successful) / n_success,
        "avg_coverage": sum(r["coverage"] for r in successful) / n_success,
        "avg_error_propagation_rate": (
            sum(propagation_values) / len(propagation_values) if propagation_values else 0.0
        ),
        "avg_latency_s": sum(r["latency_s"] for r in successful) / n_success,
        "avg_switches": sum(r["adapter_switches"] for r in successful) / n_success,
        "replan_rate": sum(1 for r in successful if r["replanned"]) / n_success,
    }
