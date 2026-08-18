from pydantic import BaseModel
from datetime import datetime


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


class TaskRequest(BaseModel):
    """Incoming task from the user.

    Attributes:
        query: The user's natural-language task description.
        replan_on_failure: Regenerate the plan if any step fails.
        always_replan: Regenerate the plan after every completed step.
    """

    query: str
    replan_on_failure: bool = True
    always_replan: bool = False


class TaskResult(BaseModel):
    """Final result returned after full plan execution."""

    query: str
    plan: Plan
    steps: list[StepResult]
    final_output: str
    replanned: bool
    created_at: datetime
