import pytest

from benchmarks.multistep_metrics import (
    align_steps,
    error_propagation_rate,
    step_accuracy,
    summarize_multistep,
    task_completion,
)


def _oracle(id_, domain):
    return {"id": id_, "domain": domain, "oracle_adapter": domain, "reference": f"ref-{domain}", "meta": {}}


def test_align_steps_exact_match():
    oracle = [_oracle(1, "medical"), _oracle(2, "summarizer")]
    actual = [{"adapter": "medical", "output": "a"}, {"adapter": "summarizer", "output": "b"}]
    aligned = align_steps(oracle, actual)
    assert len(aligned) == 2
    assert aligned[0]["oracle_step"] == oracle[0]
    assert aligned[0]["actual_step"] == actual[0]
    assert aligned[1]["actual_step"] == actual[1]


def test_align_steps_missing_step():
    oracle = [_oracle(1, "medical"), _oracle(2, "summarizer")]
    actual = [{"adapter": "medical", "output": "a"}]
    aligned = align_steps(oracle, actual)
    assert aligned[0]["actual_step"] == actual[0]
    assert aligned[1]["actual_step"] is None


def test_align_steps_extra_actual_steps_ignored():
    oracle = [_oracle(1, "medical")]
    actual = [{"adapter": "grammar", "output": "x"}, {"adapter": "medical", "output": "a"}]
    aligned = align_steps(oracle, actual)
    assert aligned[0]["actual_step"] == actual[1]


def test_align_steps_reordered():
    oracle = [_oracle(1, "medical"), _oracle(2, "summarizer")]
    actual = [{"adapter": "summarizer", "output": "b"}, {"adapter": "medical", "output": "a"}]
    aligned = align_steps(oracle, actual)
    assert aligned[0]["actual_step"] == actual[1]
    assert aligned[1]["actual_step"] == actual[0]


def test_align_steps_duplicate_adapter_consumed_once():
    oracle = [_oracle(1, "medical"), _oracle(2, "medical")]
    actual = [{"adapter": "medical", "output": "a"}, {"adapter": "medical", "output": "b"}]
    aligned = align_steps(oracle, actual)
    assert aligned[0]["actual_step"] == actual[0]
    assert aligned[1]["actual_step"] == actual[1]


def test_align_steps_empty_actual():
    oracle = [_oracle(1, "medical")]
    aligned = align_steps(oracle, [])
    assert aligned[0]["actual_step"] is None


def _aligned(*scored):
    """scored: list of (score, passed) tuples, actual_step present iff passed is not None."""
    out = []
    for score, passed in scored:
        actual = None if passed is None else {"adapter": "x", "output": "y"}
        out.append({
            "oracle_step": {"id": 1, "domain": "medical"},
            "actual_step": actual,
            "score": score,
            "passed": bool(passed),
        })
    return out


def test_task_completion_all_pass():
    aligned = _aligned((1.0, True), (1.0, True))
    assert task_completion(aligned) is True


def test_task_completion_one_failed():
    aligned = _aligned((1.0, True), (0.2, False))
    assert task_completion(aligned) is False


def test_task_completion_one_unmatched():
    aligned = _aligned((1.0, True), (0.0, None))
    assert task_completion(aligned) is False


def test_step_accuracy_mixed():
    aligned = _aligned((1.0, True), (0.5, False), (0.0, None))
    assert step_accuracy(aligned) == pytest.approx(0.5)


def test_step_accuracy_empty():
    assert step_accuracy([]) == 0.0


def test_error_propagation_no_failures_is_none():
    aligned = _aligned((1.0, True), (1.0, True))
    assert error_propagation_rate(aligned) is None


def test_error_propagation_first_step_fails_all_downstream_fail():
    aligned = _aligned((0.0, False), (0.0, False), (0.0, False))
    assert error_propagation_rate(aligned) == 1.0


def test_error_propagation_recovers_after_failure():
    aligned = _aligned((1.0, True), (0.0, False), (1.0, True))
    assert error_propagation_rate(aligned) == 0.0


def test_error_propagation_last_step_only_failure():
    aligned = _aligned((1.0, True), (1.0, True), (0.0, False))
    # no downstream steps after the last one, so nothing to propagate to
    assert error_propagation_rate(aligned) == 0.0


def test_summarize_multistep_all_success():
    records = [
        {
            "success": True, "latency_s": 2.0, "adapter_switches": 1, "replanned": False,
            "plan_precision": 1.0, "plan_recall": 1.0, "plan_f1": 1.0, "plan_order_correct": True,
            "task_completion": True, "step_accuracy": 1.0, "coverage": 1.0,
            "error_propagation_rate": None,
        },
        {
            "success": True, "latency_s": 4.0, "adapter_switches": 2, "replanned": True,
            "plan_precision": 0.5, "plan_recall": 0.5, "plan_f1": 0.5, "plan_order_correct": False,
            "task_completion": False, "step_accuracy": 0.5, "coverage": 0.5,
            "error_propagation_rate": 1.0,
        },
    ]
    result = summarize_multistep(records)
    assert result["n_tasks"] == 2
    assert result["n_success"] == 2
    assert result["task_completion_rate"] == 0.5
    assert result["avg_step_accuracy"] == pytest.approx(0.75)
    assert result["avg_coverage"] == pytest.approx(0.75)
    assert result["avg_error_propagation_rate"] == 1.0
    assert result["avg_plan_precision"] == pytest.approx(0.75)
    assert result["avg_latency_s"] == 3.0
    assert result["replan_rate"] == 0.5


def test_summarize_multistep_no_failures_excluded_from_propagation_avg():
    records = [
        {
            "success": True, "latency_s": 1.0, "adapter_switches": 0, "replanned": False,
            "plan_precision": 1.0, "plan_recall": 1.0, "plan_f1": 1.0, "plan_order_correct": True,
            "task_completion": True, "step_accuracy": 1.0, "coverage": 1.0,
            "error_propagation_rate": None,
        },
    ]
    result = summarize_multistep(records)
    assert result["avg_error_propagation_rate"] == 0.0


def test_summarize_multistep_no_success_returns_zeros():
    records = [{"success": False, "error": "timeout"}]
    result = summarize_multistep(records)
    assert result["n_tasks"] == 1
    assert result["n_success"] == 0
    assert result["task_completion_rate"] == 0.0
    assert result["avg_step_accuracy"] == 0.0
    assert result["avg_error_propagation_rate"] == 0.0
