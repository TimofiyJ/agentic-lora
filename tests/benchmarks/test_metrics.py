import pytest

from benchmarks.metrics import count_adapter_switches, is_subsequence, score_plan, summarize


def test_count_adapter_switches_empty():
    assert count_adapter_switches([]) == 0


def test_count_adapter_switches_single_step():
    assert count_adapter_switches([{"adapter": "sql"}]) == 0


def test_count_adapter_switches_no_switches():
    steps = [{"adapter": "sql"}, {"adapter": "sql"}]
    assert count_adapter_switches(steps) == 0


def test_count_adapter_switches_with_switches():
    steps = [
        {"adapter": "sql"},
        {"adapter": "summarizer"},
        {"adapter": "summarizer"},
        {"adapter": "grammar"},
    ]
    assert count_adapter_switches(steps) == 2


def test_is_subsequence_true():
    assert is_subsequence(["sql", "summarizer"], ["sql", "sql", "summarizer"])


def test_is_subsequence_false_wrong_order():
    assert not is_subsequence(["sql", "summarizer"], ["summarizer", "sql"])


def test_is_subsequence_false_missing_tag():
    assert not is_subsequence(["sql", "summarizer"], ["sql"])


def test_is_subsequence_empty_expected_always_true():
    assert is_subsequence([], ["sql", "code"])


def test_score_plan_exact_match():
    result = score_plan(["sql", "summarizer"], ["sql", "summarizer"])
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "order_correct": True}


def test_score_plan_extra_adapter_hurts_precision():
    result = score_plan(["sql", "summarizer", "grammar"], ["sql", "summarizer"])
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1.0
    assert result["order_correct"] is True


def test_score_plan_missing_adapter_hurts_recall():
    result = score_plan(["sql"], ["sql", "summarizer"])
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["order_correct"] is False


def test_score_plan_wrong_order():
    result = score_plan(["summarizer", "sql"], ["sql", "summarizer"])
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["order_correct"] is False


def test_score_plan_empty_actual():
    result = score_plan([], ["sql"])
    assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "order_correct": False}


def test_summarize_all_success():
    records = [
        {
            "success": True,
            "latency_s": 2.0,
            "adapter_switches": 1,
            "replanned": False,
            "plan_precision": 1.0,
            "plan_recall": 1.0,
            "plan_f1": 1.0,
            "plan_order_correct": True,
        },
        {
            "success": True,
            "latency_s": 4.0,
            "adapter_switches": 3,
            "replanned": True,
            "plan_precision": 0.5,
            "plan_recall": 0.5,
            "plan_f1": 0.5,
            "plan_order_correct": False,
        },
    ]
    result = summarize(records)
    assert result == {
        "n_tasks": 2,
        "n_success": 2,
        "avg_latency_s": 3.0,
        "avg_switches": 2.0,
        "replan_rate": 0.5,
        "avg_plan_precision": 0.75,
        "avg_plan_recall": 0.75,
        "avg_plan_f1": 0.75,
        "plan_order_correct_rate": 0.5,
    }


def test_summarize_excludes_failures():
    records = [
        {
            "success": True,
            "latency_s": 2.0,
            "adapter_switches": 0,
            "replanned": False,
            "plan_precision": 1.0,
            "plan_recall": 1.0,
            "plan_f1": 1.0,
            "plan_order_correct": True,
        },
        {"success": False, "error": "timeout"},
    ]
    result = summarize(records)
    assert result["n_tasks"] == 2
    assert result["n_success"] == 1
    assert result["avg_latency_s"] == 2.0
    assert result["avg_switches"] == 0.0
    assert result["replan_rate"] == 0.0
    assert result["avg_plan_precision"] == 1.0
    assert result["avg_plan_recall"] == 1.0
    assert result["avg_plan_f1"] == 1.0
    assert result["plan_order_correct_rate"] == 1.0


def test_summarize_no_success_returns_zeros():
    records = [{"success": False, "error": "connection refused"}]
    result = summarize(records)
    assert result == {
        "n_tasks": 1,
        "n_success": 0,
        "avg_latency_s": 0.0,
        "avg_switches": 0.0,
        "replan_rate": 0.0,
        "avg_plan_precision": 0.0,
        "avg_plan_recall": 0.0,
        "avg_plan_f1": 0.0,
        "plan_order_correct_rate": 0.0,
    }
