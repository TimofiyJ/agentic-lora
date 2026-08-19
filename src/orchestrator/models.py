from typing import Literal
from pydantic import BaseModel
from datetime import datetime


class SchedulingEvent(BaseModel):
    """A single adapter lifecycle event recorded during execution.

    Attributes:
        event: Type of event — prefetch, unload, or prefetch_error.
        adapter: Name of the adapter involved.
        timestamp: Wall-clock time of the event.
        latency_ms: Time taken for the controller call in milliseconds.
    """

    event: Literal["prefetch", "unload", "prefetch_error"]
    adapter: str
    timestamp: datetime
    latency_ms: float


class PlanStep(BaseModel):
    """A single step in the execution plan.

    Attributes:
        id: Unique step identifier used for dependency references.
        adapter: Name of the LoRA adapter to invoke.
        prompt: Prompt template; use {user_query} or {step_N} as placeholders.
        depends_on: IDs of steps that must complete before this one starts.
    """

    id: int
    adapter: str
    prompt: str
    depends_on: list[int] = []


class Plan(BaseModel):
    """Execution plan produced by the planner LLM."""

    steps: list[PlanStep]
    reasoning: str = ""


class StepResult(BaseModel):
    """Outcome of a single executed step."""

    step_id: int
    adapter: str
    output: str
    success: bool
    replanned_after: bool = False
    latency_ms: float = 0.0
    adapter_was_preloaded: bool = False


class TaskRequest(BaseModel):
    """Incoming task from the user.

    Attributes:
        query: The user's natural-language task description.
        replan_on_failure: Regenerate the plan if any step fails.
        always_replan: Regenerate the plan after every completed step.
        prefetch: Lookahead-load next-wave adapters while current wave runs.
        unload_after: Unload all task adapters from GPU when the task finishes.
    """

    query: str
    replan_on_failure: bool = True
    always_replan: bool = False
    prefetch: bool = True
    unload_after: bool = True


class TaskResult(BaseModel):
    """Final result returned after full plan execution."""

    query: str
    plan: Plan
    steps: list[StepResult]
    final_output: str
    replanned: bool
    created_at: datetime
    scheduling_events: list[SchedulingEvent] = []
