"""DAG-based plan executor with parallel step execution and optional replanning."""

import asyncio
import time
import httpx
from datetime import datetime, timezone

UTC = timezone.utc

from .models import Plan, PlanStep, StepResult, TaskResult, SchedulingEvent
from .planner import Planner
from .scheduler import ContinuousScheduler


class Executor:
    """Executes a Plan by walking its dependency graph.

    Independent steps (no unmet depends_on) are dispatched concurrently.
    After each wave of parallel steps, the plan can optionally be regenerated
    by the planner based on the accumulated results.

    Args:
        controller_url: Base URL of the LoRA controller.
        planner: Planner instance used when replanning is triggered.
        scheduler: ContinuousScheduler for prefetch and unload operations.
        timeout: HTTP timeout in seconds for inference calls.
    """

    def __init__(
        self,
        controller_url: str,
        planner: Planner,
        scheduler: ContinuousScheduler,
        timeout: float = 60.0,
    ):
        self._controller_url = controller_url.rstrip("/")
        self._planner = planner
        self._scheduler = scheduler
        self._timeout = timeout

    def _resolve_prompt(
        self, template: str, query: str, outputs: dict[int, str], depends_on: list[int] | None = None
    ) -> str:
        """Substitute placeholders in a prompt template.

        The planner LLM is instructed to reference prior-step output via
        ``{step_N}`` placeholders, but small/self-routing planners don't always
        comply — a step with a real `depends_on` can come back as a bare
        instruction with no placeholder at all, silently dropping the upstream
        output and leaving the adapter to generate blind (observed on
        Qwen2.5-7B-as-planner: e.g. a "summarize the result" step with no
        `{step_1}` anywhere in it). If any depended-on step's placeholder is
        missing from the resolved template, prepend that output as explicit
        context instead of trusting the plan's prompt text — a step that
        declares a dependency must never run without seeing that step's output.

        Args:
            template: Prompt with {user_query} and {step_N} placeholders.
            query: Original user query.
            outputs: Map of step_id → output for completed steps.
            depends_on: Step IDs this step declared as dependencies.

        Returns:
            Fully resolved prompt string.
        """
        prompt = template.replace("{user_query}", query)
        missing_context = []
        for step_id, output in outputs.items():
            placeholder = f"{{step_{step_id}}}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, output)
            elif depends_on and step_id in depends_on:
                missing_context.append((step_id, output))
        if missing_context:
            context = "\n".join(f"Result of step {sid}: {out}" for sid, out in missing_context)
            prompt = f"{context}\n\n{prompt}"
        return prompt

    def _next_wave_adapters(
        self, plan: Plan, completed_ids: set[int], current_wave_ids: set[int]
    ) -> list[str]:
        """Return unique adapter names for steps that become ready after the current wave.

        Args:
            plan: Current execution plan.
            completed_ids: IDs of steps already finished.
            current_wave_ids: IDs of steps in the wave currently executing.

        Returns:
            Deduplicated list of adapter names for the next ready wave.
        """
        after = completed_ids | current_wave_ids
        return list({
            step.adapter for step in plan.steps
            if step.id not in after
            and all(dep in after for dep in step.depends_on)
        })

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

    async def _run_step(
        self, step: PlanStep, prompt: str, currently_loaded: set[str]
    ) -> StepResult:
        """Send a single step to the controller for inference.

        Args:
            step: The plan step to execute.
            prompt: Fully resolved prompt to send.
            currently_loaded: Adapter names resident in GPU before this wave started.

        Returns:
            StepResult with output, success flag, latency, and preload status.
        """
        preloaded = step.adapter in currently_loaded
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._controller_url}/infer/{step.adapter}",
                    json={"prompt": prompt, "max_tokens": 2048, "temperature": 0.3},
                )
                resp.raise_for_status()
                output = resp.json()["choices"][0]["text"].strip()
                return StepResult(
                    step_id=step.id,
                    adapter=step.adapter,
                    output=output,
                    success=bool(output),
                    latency_ms=(time.monotonic() - t0) * 1000,
                    adapter_was_preloaded=preloaded,
                )
        except Exception as e:
            return StepResult(
                step_id=step.id,
                adapter=step.adapter,
                output=str(e),
                success=False,
                latency_ms=(time.monotonic() - t0) * 1000,
                adapter_was_preloaded=preloaded,
            )

    async def execute(
        self,
        query: str,
        plan: Plan,
        replan_on_failure: bool = True,
        always_replan: bool = False,
        prefetch: bool = True,
        unload_after: bool = True,
    ) -> TaskResult:
        """Execute a plan to completion.

        Runs each wave of dependency-free steps in parallel. Prefetches next-wave
        adapters concurrently while the current wave executes. After the full task,
        optionally unloads all adapters used.

        Args:
            query: Original user query, injected into prompts via {user_query}.
            plan: Initial plan from the planner.
            replan_on_failure: Trigger replanning if any step in a wave fails.
            always_replan: Trigger replanning after every completed wave.
            prefetch: Lookahead-load next-wave adapters while current wave runs.
            unload_after: Unload all task adapters from GPU when the task finishes.

        Returns:
            TaskResult containing all step outputs, scheduling events, and metadata.
        """
        outputs: dict[int, str] = {}
        step_results: list[StepResult] = []
        scheduling_events: list[SchedulingEvent] = []
        replanned = False
        current_plan = plan

        while True:
            ready = self._ready_steps(current_plan, set(outputs.keys()))
            if not ready:
                break

            current_wave_ids = {s.id for s in ready}
            currently_loaded = await self._scheduler.get_loaded()

            next_adapters = self._next_wave_adapters(current_plan, set(outputs.keys()), current_wave_ids)
            prefetch_task = (
                asyncio.create_task(self._scheduler.prefetch(next_adapters))
                if prefetch and next_adapters
                else None
            )

            wave_results = await asyncio.gather(*[
                self._run_step(
                    step, self._resolve_prompt(step.prompt, query, outputs, step.depends_on), currently_loaded
                )
                for step in ready
            ])

            if prefetch_task is not None:
                scheduling_events.extend(await prefetch_task)

            for result in wave_results:
                step_results.append(result)
                outputs[result.step_id] = result.output

            failed = [r for r in wave_results if not r.success]
            remaining = len(current_plan.steps) - len(outputs)
            should_replan = remaining > 0 and (always_replan or (replan_on_failure and failed))

            if should_replan:
                current_plan = await self._planner.replan(query, step_results, currently_loaded)
                # carry forward outputs by step_id so resolved prompts still work
                outputs = {r.step_id: r.output for r in step_results if r.success}
                replanned = True
                # mark the last result as the replan trigger point
                step_results[-1] = step_results[-1].model_copy(update={"replanned_after": True})

        if unload_after:
            used = {r.adapter for r in step_results}
            scheduling_events.extend(await self._scheduler.unload_all(used))

        final_output = step_results[-1].output if step_results else ""
        return TaskResult(
            query=query,
            plan=plan,
            steps=step_results,
            final_output=final_output,
            replanned=replanned,
            created_at=datetime.now(UTC),
            scheduling_events=scheduling_events,
        )
