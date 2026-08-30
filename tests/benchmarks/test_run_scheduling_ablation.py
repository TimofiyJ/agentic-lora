import httpx

from benchmarks.run_scheduling_ablation import (
    fetch_plan,
    run_ablation,
    run_task_with_plan,
    summarize_ablation,
)


def _task():
    return {"id": "ms_001", "query": "extract findings then summarize"}


def test_fetch_plan_returns_json():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/plan"
        assert request.url.params["query"] == "extract findings then summarize"
        return httpx.Response(200, json={"steps": [{"id": 1, "adapter": "medical", "prompt": "{user_query}", "depends_on": []}], "reasoning": ""})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    plan = fetch_plan("http://fake-orchestrator", "extract findings then summarize", client)
    assert plan["steps"][0]["adapter"] == "medical"


def test_run_task_with_plan_sends_plan_and_prefetch_flag():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"steps": [{"adapter": "medical"}, {"adapter": "summarizer"}], "replanned": False})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    plan = {"steps": [{"id": 1, "adapter": "medical", "prompt": "p", "depends_on": []}], "reasoning": ""}
    result = run_task_with_plan("http://fake-orchestrator", "q", plan, prefetch=True, client=client)

    assert captured["payload"]["plan"] == plan
    assert captured["payload"]["prefetch"] is True
    assert captured["payload"]["replan_on_failure"] is False
    assert captured["payload"]["always_replan"] is False
    assert result["prefetch"] is True
    assert result["adapter_switches"] == 1
    assert "latency_s" in result


def test_run_task_with_plan_records_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    plan = {"steps": [], "reasoning": ""}
    result = run_task_with_plan("http://fake-orchestrator", "q", plan, prefetch=False, client=client)
    assert "error" in result
    assert result["prefetch"] is False


def test_run_ablation_runs_both_prefetch_settings_per_task():
    tasks = [_task()]
    plan_calls = []
    task_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/plan":
            plan_calls.append(1)
            return httpx.Response(200, json={"steps": [{"id": 1, "adapter": "medical", "prompt": "p", "depends_on": []}], "reasoning": ""})
        import json
        payload = json.loads(request.content)
        task_calls.append(payload["prefetch"])
        return httpx.Response(200, json={"steps": [{"adapter": "medical"}], "replanned": False})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = run_ablation(tasks, "http://fake-orchestrator", client, seed=0)

    assert len(plan_calls) == 1  # plan fetched exactly once, reused for both runs
    assert sorted(task_calls) == [False, True]  # both prefetch settings exercised
    assert len(records) == 1
    assert records[0]["prefetch_on"]["prefetch"] is True
    assert records[0]["prefetch_off"]["prefetch"] is False


def test_summarize_ablation_computes_delta_and_win_rate():
    records = [
        {
            "id": "t1",
            "prefetch_on": {"latency_s": 2.0, "prefetch": True},
            "prefetch_off": {"latency_s": 3.0, "prefetch": False},
        },
        {
            "id": "t2",
            "prefetch_on": {"latency_s": 5.0, "prefetch": True},
            "prefetch_off": {"latency_s": 4.0, "prefetch": False},
        },
    ]
    summary = summarize_ablation(records)
    assert summary["n_tasks"] == 2
    assert summary["n_paired"] == 2
    assert summary["avg_latency_on"] == 3.5
    assert summary["avg_latency_off"] == 3.5
    assert summary["avg_delta_s"] == 0.0
    assert summary["win_rate_on"] == 0.5


def test_summarize_ablation_excludes_unpaired_or_errored_tasks():
    records = [
        {"id": "t1", "error": "fetch_plan failed"},
        {
            "id": "t2",
            "prefetch_on": {"latency_s": 1.0, "prefetch": True},
            "prefetch_off": {"error": "boom", "prefetch": False},
        },
    ]
    summary = summarize_ablation(records)
    assert summary["n_tasks"] == 2
    assert summary["n_paired"] == 0
