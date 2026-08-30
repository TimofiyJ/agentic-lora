import json
from pathlib import Path

import httpx
import pytest

from benchmarks.run import build_error_record, build_success_record, run_benchmark


def test_build_success_record():
    task = {"id": "t1", "query": "do a thing", "adapters_expected": ["sql", "summarizer"]}
    result = {
        "steps": [{"adapter": "sql"}, {"adapter": "summarizer"}],
        "replanned": True,
        "final_output": "done",
    }
    record = build_success_record(task, result, latency_s=1.5)
    assert record == {
        "id": "t1",
        "query": "do a thing",
        "success": True,
        "latency_s": 1.5,
        "num_steps": 2,
        "adapter_switches": 1,
        "replanned": True,
        "final_output": "done",
        "plan_precision": 1.0,
        "plan_recall": 1.0,
        "plan_f1": 1.0,
        "plan_order_correct": True,
    }


def test_build_error_record():
    task = {"id": "t2", "query": "do another thing"}
    record = build_error_record(task, "connection refused")
    assert record == {
        "id": "t2",
        "query": "do another thing",
        "success": False,
        "error": "connection refused",
    }


def test_run_benchmark_success():
    tasks = [
        {"id": "t1", "query": "q1", "adapters_expected": ["sql"]},
        {"id": "t2", "query": "q2", "adapters_expected": ["sql"]},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "steps": [{"adapter": "sql"}],
                "replanned": False,
                "final_output": f"answer to {body['query']}",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = run_benchmark(tasks, "http://fake-orchestrator", client)

    assert len(records) == 2
    assert all(r["success"] for r in records)
    assert records[0]["final_output"] == "answer to q1"
    assert records[1]["final_output"] == "answer to q2"


def test_run_benchmark_records_http_errors():
    tasks = [{"id": "t1", "query": "q1"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "internal error"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = run_benchmark(tasks, "http://fake-orchestrator", client)

    assert len(records) == 1
    assert records[0]["success"] is False
    assert records[0]["id"] == "t1"
    assert "error" in records[0]


def test_tasks_json_integration():
    tasks_path = Path(__file__).parent.parent.parent / "benchmarks" / "tasks.json"
    tasks = json.loads(tasks_path.read_text())

    assert len(tasks) == 50
    for task in tasks:
        assert "id" in task
        assert "query" in task
        assert "adapters_expected" in task

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "steps": [{"adapter": "mock"}],
                "replanned": False,
                "final_output": f"result for {body['query'][:30]}",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = run_benchmark(tasks, "http://fake-orchestrator", client)

    assert len(records) == 50
    assert all(r["success"] for r in records)
    for record in records:
        assert "id" in record
        assert "query" in record
        assert "success" in record
        assert "latency_s" in record
        assert "num_steps" in record
        assert "adapter_switches" in record
        assert "replanned" in record
        assert "final_output" in record
        assert "plan_precision" in record
        assert "plan_recall" in record
        assert "plan_f1" in record
        assert "plan_order_correct" in record
