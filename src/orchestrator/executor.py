"""DAG-based plan executor with parallel step execution and optional replanning."""

import asyncio
import httpx
from datetime import datetime, UTC

from .models import Plan, PlanStep, StepResult, TaskResult
from .planner import Planner


class Executor:
    """Executes a Plan by walking its dependency graph.

    Independent steps (no unmet depends_on) are dispatched concurrently.
    After each wave of parallel steps, the plan can optionally be regenerated
    by the planner based on the accumulated results.

    Args:
        controller_url: Base URL of the LoRA controller.
        planner: Planner instance used when replanning is triggered.
        timeout: HTTP timeout in seconds for inference calls.
    """

    def __init__(self, controller_url: str, planner: Planner, timeout: float = 60.0):
        self._controller_url = controller_url.rstrip("/")
        self._planner = planner
        self._timeout = timeout

    def _resolve_prompt(self, template: str, query: str, outputs: dict[int, str]) -> str:
        """Substitute placeholders in a prompt template.

        Args:
            template: Prompt with {user_query} and {step_N} placeholders.
            query: Original user query.
            outputs: Map of step_id → output for completed steps.

        Returns:
            Fully resolved prompt string.
        """
        prompt = template.replace("{user_query}", query)
        for step_id, output in outputs.items():
            prompt = prompt.replace(f"{{step_{step_id}}}", output)
        return prompt

    def _ready_steps(self, plan: Plan, completed_ids: set[int]) -> list[PlanStep]:
        """Return steps whose dependencies are all satisfied.

        Args:
            plan: Current execution plan.
            completed_ids: IDs of steps that have already finished.

        Returns:
            List of steps ready to execute.
        """
        return [
            step for step in plan.steps
            if step.id not in completed_ids
            and all(dep in completed_ids for dep in step.depends_on)
        ]

    async def _run_step(self, step: PlanStep, prompt: str) -> StepResult:
        """Send a single step to the controller for inference.

        Args:
            step: The plan step to execute.
            prompt: Fully resolved prompt to send.

        Returns:
            StepResult with output and success flag.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._controller_url}/infer/{step.adapter}",
                    json={"prompt": prompt, "max_tokens": 512, "temperature": 0.3},
                )
                resp.raise_for_status()
                output = resp.json()["choices"][0]["text"].strip()
                return StepResult(step_id=step.id, adapter=step.adapter, output=output, success=bool(output))
        except Exception as e:
            return StepResult(step_id=step.id, adapter=step.adapter, output=str(e), success=False)

    async def execute(
        self,
        query: str,
        plan: Plan,
        replan_on_failure: bool = True,
        always_replan: bool = False,
    ) -> TaskResult:
        """Execute a plan to completion.

        Runs each wave of dependency-free steps in parallel. After each wave,
        optionally replans based on results before proceeding.

        Args:
            query: Original user query, injected into prompts via {user_query}.
            plan: Initial plan from the planner.
            replan_on_failure: Trigger replanning if any step in a wave fails.
            always_replan: Trigger replanning after every completed wave.

        Returns:
            TaskResult containing all step outputs and the final answer.
        """
        outputs: dict[int, str] = {}
        step_results: list[StepResult] = []
        replanned = False
        current_plan = plan

        while True:
            ready = self._ready_steps(current_plan, set(outputs.keys()))
            if not ready:
                break

            wave_results = await asyncio.gather(*[
                self._run_step(step, self._resolve_prompt(step.prompt, query, outputs))
                for step in ready
            ])

            for result in wave_results:
                step_results.append(result)
                outputs[result.step_id] = result.output

            failed = [r for r in wave_results if not r.success]
            remaining = len(current_plan.steps) - len(outputs)
            should_replan = remaining > 0 and (always_replan or (replan_on_failure and failed))

            if should_replan:
                current_plan = await self._planner.replan(query, step_results)
                # carry forward outputs by step_id so resolved prompts still work
                outputs = {r.step_id: r.output for r in step_results if r.success}
                replanned = True
                # mark the last result as the replan trigger point
                step_results[-1] = step_results[-1].model_copy(update={"replanned_after": True})

        final_output = step_results[-1].output if step_results else ""
        return TaskResult(
            query=query,
            plan=plan,
            steps=step_results,
            final_output=final_output,
            replanned=replanned,
            created_at=datetime.now(UTC),
        )
