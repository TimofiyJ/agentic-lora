from benchmarks.metrics import count_adapter_switches, summarize


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


def test_summarize_all_success():
    records = [
        {"success": True, "latency_s": 2.0, "adapter_switches": 1, "replanned": False},
        {"success": True, "latency_s": 4.0, "adapter_switches": 3, "replanned": True},
    ]
    result = summarize(records)
    assert result == {
        "n_tasks": 2,
        "n_success": 2,
        "avg_latency_s": 3.0,
        "avg_switches": 2.0,
        "replan_rate": 0.5,
    }


def test_summarize_excludes_failures():
    records = [
        {"success": True, "latency_s": 2.0, "adapter_switches": 0, "replanned": False},
        {"success": False, "error": "timeout"},
    ]
    result = summarize(records)
    assert result["n_tasks"] == 2
    assert result["n_success"] == 1
    assert result["avg_latency_s"] == 2.0
    assert result["avg_switches"] == 0.0
    assert result["replan_rate"] == 0.0


def test_summarize_no_success_returns_zeros():
    records = [{"success": False, "error": "connection refused"}]
    result = summarize(records)
    assert result == {
        "n_tasks": 1,
        "n_success": 0,
        "avg_latency_s": 0.0,
        "avg_switches": 0.0,
        "replan_rate": 0.0,
    }
