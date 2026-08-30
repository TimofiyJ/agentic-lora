from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from orchestrator import app as app_module
from orchestrator.models import Plan, PlanStep, StepResult, TaskResult


@pytest.fixture
def client():
    with TestClient(app_module.app) as c:
        yield c


def test_task_with_explicit_plan_skips_planner(client, monkeypatch):
    """A caller-supplied plan bypasses planner.plan() entirely (deterministic
    scheduling ablations: run the identical plan under prefetch on/off without
    re-planning variance — see TaskRequest.plan docstring)."""

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("planner.plan() should not be called when body.plan is set")

    monkeypatch.setattr(app_module.planner, "plan", fail_if_called)

    received = {}

    async def fake_execute(query, plan, replan_on_failure, always_replan, prefetch, unload_after):
        received["plan"] = plan
        received["prefetch"] = prefetch
        return TaskResult(
            query=query,
            plan=plan,
            steps=[StepResult(step_id=1, adapter="sql", output="ok", success=True)],
            final_output="ok",
            replanned=False,
            created_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr(app_module.executor, "execute", fake_execute)

    supplied_plan = {
        "steps": [{"id": 1, "adapter": "sql", "prompt": "{user_query}", "depends_on": []}],
        "reasoning": "precomputed",
    }
    resp = client.post(
        "/task",
        json={"query": "top 5 customers by revenue", "plan": supplied_plan, "prefetch": False},
    )
    assert resp.status_code == 200
    assert received["prefetch"] is False
    assert received["plan"].steps[0].adapter == "sql"
    assert received["plan"].reasoning == "precomputed"


def test_task_without_plan_calls_planner(client, monkeypatch):
    """Default behavior (no body.plan) is unchanged: planner.plan() runs."""

    called = {}

    async def fake_plan(query, loaded_adapters):
        called["query"] = query
        return Plan(steps=[PlanStep(id=1, adapter="sql", prompt="{user_query}")], reasoning="")

    async def fake_get_loaded():
        return set()

    async def fake_execute(query, plan, replan_on_failure, always_replan, prefetch, unload_after):
        return TaskResult(
            query=query,
            plan=plan,
            steps=[],
            final_output="",
            replanned=False,
            created_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr(app_module.planner, "plan", fake_plan)
    monkeypatch.setattr(app_module.scheduler, "get_loaded", fake_get_loaded)
    monkeypatch.setattr(app_module.executor, "execute", fake_execute)

    resp = client.post("/task", json={"query": "top 5 customers by revenue"})
    assert resp.status_code == 200
    assert called["query"] == "top 5 customers by revenue"
