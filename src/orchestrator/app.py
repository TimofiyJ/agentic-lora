"""Orchestrator service — accepts user tasks and executes multi-adapter plans.

Environment variables:
    VLLM_URL:             vLLM server URL (default: http://localhost:8000)
    CONTROLLER_URL:       LoRA controller URL (default: http://localhost:8001)
    PLANNER_MODEL:        Model name for planning (default: meta-llama/Llama-3.2-3B-Instruct)
    ORCHESTRATOR_TIMEOUT: HTTP timeout in seconds (default: 60)
"""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from .executor import Executor
from .planner import Planner
from .scheduler import ContinuousScheduler
from .models import TaskRequest, TaskResult


planner: Planner
scheduler: ContinuousScheduler
executor: Executor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global planner, scheduler, executor
    vllm_url = os.getenv("VLLM_URL", "http://localhost:8000")
    controller_url = os.getenv("CONTROLLER_URL", "http://localhost:8001")
    model = os.getenv("PLANNER_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    timeout = float(os.getenv("ORCHESTRATOR_TIMEOUT", "60"))

    planner = Planner(vllm_url, controller_url, model, timeout)
    scheduler = ContinuousScheduler(controller_url, timeout)
    executor = Executor(controller_url, planner, scheduler, timeout)
    yield


app = FastAPI(
    title="Orchestrator",
    description="Multi-adapter task orchestrator with DAG execution and replanning",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/task", response_model=TaskResult)
async def run_task(body: TaskRequest):
    """Submit a task for planning and execution.

    Args:
        body: Task request with query, replanning flags, and scheduling options.

    Returns:
        TaskResult with per-step outputs, scheduling events, and plan metadata.
    """
    loaded_adapters = await scheduler.get_loaded()
    plan = await planner.plan(body.query, loaded_adapters)
    return await executor.execute(
        query=body.query,
        plan=plan,
        replan_on_failure=body.replan_on_failure,
        always_replan=body.always_replan,
        prefetch=body.prefetch,
        unload_after=body.unload_after,
    )


@app.get("/plan")
async def preview_plan(query: str):
    """Return the plan the LLM would generate without executing it.

    Args:
        query: The task to plan for.

    Returns:
        The generated Plan with steps and reasoning.
    """
    return await planner.plan(query)


@app.get("/health")
def health():
    return {"status": "ok"}
