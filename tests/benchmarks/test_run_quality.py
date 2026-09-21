import json
from pathlib import Path

import httpx

from benchmarks.run_quality import build_error_record, build_success_record, run_benchmark


def test_quality_tasks_json_integration():
    tasks_path = Path(__file__).parent.parent.parent / "benchmarks" / "quality_tasks.json"
    tasks = json.loads(tasks_path.read_text())

    assert len(tasks) == 120
    by_domain: dict[str, int] = {}
    for task in tasks:
        assert "id" in task
        assert "domain" in task
        assert "adapter" in task
        assert "prompt" in task
        assert "reference" in task
        assert "meta" in task
        by_domain[task["domain"]] = by_domain.get(task["domain"], 0) + 1

    assert by_domain == {
        "sql": 20, "code": 20, "legal": 20, "medical": 20, "grammar": 20, "summarizer": 20,
    }


def test_build_success_record():
    task = {"id": "quality_sql_01", "domain": "sql", "adapter": "sql", "reference": "SELECT 1", "meta": {}}
    record = build_success_record(task, "SELECT 1", latency_s=0.5)
    assert record == {
        "id": "quality_sql_01",
        "domain": "sql",
        "success": True,
        "latency_s": 0.5,
        "prediction": "SELECT 1",
        "score": 1.0,
        "passed": True,
    }


def test_build_error_record():
    task = {"id": "quality_sql_02", "domain": "sql"}
    record = build_error_record(task, "connection refused")
    assert record == {
        "id": "quality_sql_02",
        "domain": "sql",
        "success": False,
        "error": "connection refused",
    }


def test_run_benchmark_success():
    tasks = [
        {"id": "quality_sql_01", "domain": "sql", "adapter": "sql", "prompt": "get all", "reference": "SELECT 1", "meta": {}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/adapters":
            return httpx.Response(200, json=[{"name": "sql"}])
        return httpx.Response(200, json={"choices": [{"text": "SELECT 1"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = run_benchmark(tasks, "http://fake-controller", client)

    assert len(records) == 1
    assert records[0]["success"] is True
    assert records[0]["score"] == 1.0


def test_run_benchmark_records_http_errors():
    tasks = [
        {"id": "quality_sql_01", "domain": "sql", "adapter": "sql", "prompt": "get all", "reference": "SELECT 1", "meta": {}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/adapters":
            return httpx.Response(200, json=[{"name": "sql"}])
        return httpx.Response(500, json={"detail": "internal error"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = run_benchmark(tasks, "http://fake-controller", client)

    assert len(records) == 1
    assert records[0]["success"] is False
    assert records[0]["id"] == "quality_sql_01"
    assert "error" in records[0]


def test_run_benchmark_sends_raw_completion_for_sql_domain():
    tasks = [
        {"id": "quality_sql_01", "domain": "sql", "adapter": "sql", "prompt": "CREATE TABLE t; SELECT?", "reference": "SELECT 1", "meta": {}},
    ]
    seen_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/adapters":
            return httpx.Response(200, json=[{"name": "sql"}])
        seen_payload.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"text": "SELECT 1"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    run_benchmark(tasks, "http://fake-controller", client)

    assert seen_payload["raw_completion"] is True


def test_run_benchmark_sends_stop_sequence_for_sql_domain():
    """sql is raw_completion (no chat template, no natural stopping point) --
    without a stop sequence the model keeps generating extra SQL statements
    after the first one, which never exact-matches the single-statement
    reference. See docs/results.md's E4 entry for the real 0.000 this caused."""
    tasks = [
        {"id": "quality_sql_01", "domain": "sql", "adapter": "sql", "prompt": "CREATE TABLE t; SELECT?", "reference": "SELECT 1", "meta": {}},
    ]
    seen_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/adapters":
            return httpx.Response(200, json=[{"name": "sql"}])
        seen_payload.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"text": "SELECT 1"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    run_benchmark(tasks, "http://fake-controller", client)

    assert seen_payload["stop"] == ["\n\n"]


def test_run_benchmark_sends_no_stop_sequence_for_non_sql_domains():
    tasks = [
        {"id": "quality_legal_01", "domain": "legal", "adapter": "legal", "prompt": "q", "reference": "a", "meta": {}},
    ]
    seen_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/adapters":
            return httpx.Response(200, json=[{"name": "legal"}])
        seen_payload.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "a"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    run_benchmark(tasks, "http://fake-controller", client)

    assert seen_payload["stop"] == []


def test_run_benchmark_skips_unregistered_adapter():
    tasks = [
        {"id": "quality_sql_01", "domain": "sql", "adapter": "sql", "prompt": "get all", "reference": "SELECT 1", "meta": {}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/adapters", "should not call /infer for an unregistered adapter"
        return httpx.Response(200, json=[])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = run_benchmark(tasks, "http://fake-controller", client)

    assert len(records) == 1
    assert records[0]["success"] is False
    assert records[0]["skipped"] is True
