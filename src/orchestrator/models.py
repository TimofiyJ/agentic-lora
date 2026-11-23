from typing import Literal
from pydantic import BaseModel, Field
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
        plan: Optional precomputed plan to execute instead of calling the
            planner. Lets a caller run the exact same plan under different
            scheduling settings (e.g. prefetch on vs off) with zero
            plan-to-plan variance — otherwise a prefetch ablation confounds
            "did scheduling help" with "did the planner happen to produce a
            different plan this time" (the planner is not fully deterministic:
            temperature=0.2, and its prompt includes which adapters are
            currently GPU-resident, which itself depends on prior scheduling
            state). Get a plan to reuse via GET /plan.
        step_temperature: Sampling temperature for every step's adapter call.
        step_max_tokens: Generation budget for every step's adapter call.
    """

    query: str
    replan_on_failure: bool = True
    always_replan: bool = False
    prefetch: bool = True
    unload_after: bool = True
    plan: Plan | None = None
    step_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    step_max_tokens: int = Field(default=2048, ge=1, le=4096)


class TaskResult(BaseModel):
    """Final result returned after full plan execution."""

    query: str
    plan: Plan
    steps: list[StepResult]
    final_output: str
    replanned: bool
    created_at: datetime
    scheduling_events: list[SchedulingEvent] = []
