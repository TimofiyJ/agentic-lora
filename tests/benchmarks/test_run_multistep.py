import json
from pathlib import Path

import httpx

import pytest

from benchmarks.run_multistep import build_error_record, build_success_record, run_benchmark


def _task():
    return {
        "id": "ms_001",
        "query": "extract findings then summarize",
        "oracle_adapter_sequence": ["medical", "summarizer"],
        "steps": [
            {"id": 1, "domain": "medical", "oracle_adapter": "medical", "reference": "18.3 mmHg reduction", "meta": {}},
            {"id": 2, "domain": "summarizer", "oracle_adapter": "summarizer", "reference": "Treatment reduced blood pressure.", "meta": {}},
        ],
    }


def test_build_success_record_all_steps_pass():
    task = _task()
    result = {
        "steps": [
            {"adapter": "medical", "output": "18.3 mmHg reduction"},
            {"adapter": "summarizer", "output": "Treatment reduced blood pressure."},
        ],
        "replanned": False,
    }
    record = build_success_record(task, result, latency_s=1.2)
    assert record["id"] == "ms_001"
    assert record["success"] is True
    assert record["latency_s"] == 1.2
    assert record["adapter_switches"] == 1
    assert record["replanned"] is False
    assert record["plan_precision"] == 1.0
    assert record["plan_recall"] == 1.0
    assert record["plan_order_correct"] is True
    assert record["coverage"] == 1.0
    assert record["step_accuracy"] == 1.0
    assert record["task_completion"] is True
    assert record["error_propagation_rate"] is None


def test_build_success_record_missing_step():
    task = _task()
    result = {"steps": [{"adapter": "medical", "output": "18.3 mmHg reduction"}], "replanned": False}
    record = build_success_record(task, result, latency_s=0.5)
    assert record["coverage"] == 0.5
    assert record["task_completion"] is False
    assert record["step_accuracy"] == pytest.approx(0.5)


def test_build_error_record():
    task = _task()
    record = build_error_record(task, "connection refused")
    assert record == {"id": "ms_001", "query": task["query"], "success": False, "error": "connection refused"}


def test_run_benchmark_success():
    tasks = [_task()]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "steps": [
                    {"adapter": "medical", "output": "18.3 mmHg reduction"},
                    {"adapter": "summarizer", "output": "Treatment reduced blood pressure."},
                ],
                "replanned": False,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = run_benchmark(tasks, "http://fake-orchestrator", client)
    assert len(records) == 1
    assert records[0]["success"] is True


def test_run_benchmark_records_http_errors():
    tasks = [_task()]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "internal error"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = run_benchmark(tasks, "http://fake-orchestrator", client)
    assert len(records) == 1
    assert records[0]["success"] is False
    assert records[0]["id"] == "ms_001"


def test_multistep_tasks_json_structure():
    tasks_path = Path(__file__).parent.parent.parent / "benchmarks" / "multistep_tasks.json"
    tasks = json.loads(tasks_path.read_text())

    assert len(tasks) == 20

    valid_domains = {"sql", "code", "legal", "medical", "grammar", "summarizer"}
    sequences_seen = set()
    domain_counts = {d: 0 for d in valid_domains}
    n_steps_counts = {2: 0, 3: 0, 4: 0}

    for task in tasks:
        assert "id" in task
        assert "query" in task
        seq = tuple(task["oracle_adapter_sequence"])
        assert seq not in sequences_seen, f"duplicate sequence: {seq}"
        sequences_seen.add(seq)
        assert task["metadata"]["solvable_by_single_adapter"] is False
        assert task["metadata"]["n_steps"] == len(task["steps"])
        n_steps_counts[len(task["steps"])] += 1

        for step in task["steps"]:
            assert step["domain"] in valid_domains
            assert step["oracle_adapter"] == step["domain"]
            assert "reference" in step
            assert "meta" in step
            domain_counts[step["domain"]] += 1
            if step["domain"] == "code":
                assert "prompt" in step
                assert "test" in step["meta"]
                assert "entry_point" in step["meta"]
            if step["domain"] == "legal":
                assert "answers" in step["meta"]
            if step["domain"] == "grammar":
                assert "corrections" in step["meta"]

    assert n_steps_counts == {2: 5, 3: 10, 4: 5}
    for domain, count in domain_counts.items():
        assert count >= 3, f"{domain} appears only {count} times"
