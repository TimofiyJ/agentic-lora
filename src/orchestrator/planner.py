"""LLM-based plan generation and replanning."""

import json
import httpx
from .models import Plan, PlanStep, StepResult


_SYSTEM_PROMPT = """You are a task planner for a multi-adapter LLM system. Given a user query and a list of available LoRA adapters, produce a JSON execution plan.

Output ONLY valid JSON with this exact structure — no markdown, no explanation outside the JSON:
{
  "steps": [
    {
      "id": 1,
      "adapter": "<adapter_name>",
      "prompt": "<complete prompt for this adapter; use {user_query} or {step_N} as placeholders>",
      "depends_on": []
    }
  ],
  "reasoning": "<one sentence explaining the plan>"
}

Rules:
- Only use adapters from the provided list.
- depends_on lists step IDs that must finish before this step starts; empty means it can run in parallel with other steps that have no unmet dependencies.
- Prompts must be complete and self-contained — the adapter receives only the prompt text.
- Use {user_query} to inject the original query, {step_N} to inject the output of step N.
- Include only steps that are genuinely needed."""


class Planner:
    """Generates and revises execution plans using the base LLM.

    Args:
        vllm_url: Base URL of the vLLM server used for plan generation.
        controller_url: Base URL of the LoRA controller, used to fetch adapter metadata.
        model: Model name to use for planning (should be the base model, no LoRA).
        timeout: HTTP timeout in seconds for LLM calls.
    """

    def __init__(self, vllm_url: str, controller_url: str, model: str, timeout: float = 60.0):
        self._vllm_url = vllm_url.rstrip("/")
        self._controller_url = controller_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def _get_adapters(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self._controller_url}/adapters")
            resp.raise_for_status()
            return resp.json()

    def _format_adapter_list(self, adapters: list[dict]) -> str:
        lines = []
        for a in adapters:
            tags = ", ".join(a.get("tags", []))
            lines.append(f"- {a['name']}: {a['description']} [tags: {tags}]")
        return "\n".join(lines)

    async def _call_llm(self, user_message: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._vllm_url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def _parse_plan(self, raw: str) -> Plan:
        """Parse LLM output into a Plan, stripping markdown fences if present.

        Args:
            raw: Raw string output from the LLM.

        Returns:
            Parsed Plan object.

        Raises:
            json.JSONDecodeError: If the output is not valid JSON after stripping.
        """
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        return Plan(
            steps=[PlanStep(**s) for s in data["steps"]],
            reasoning=data.get("reasoning", ""),
        )

    async def plan(self, query: str, loaded_adapters: set[str] | None = None) -> Plan:
        """Generate an initial execution plan for a user query.

        Args:
            query: The user's natural-language task.
            loaded_adapters: Adapter names currently resident in GPU; the planner
                will prefer them to avoid unnecessary load latency.

        Returns:
            A Plan with steps and optional reasoning.
        """
        adapters = await self._get_adapters()
        user_message = f"Available adapters:\n{self._format_adapter_list(adapters)}\n\n"
        if loaded_adapters:
            names = ", ".join(sorted(loaded_adapters))
            user_message += (
                f"Currently loaded in GPU (no load penalty): {names}\n"
                f"Prefer these adapters when they can handle the task.\n\n"
            )
        user_message += f"User query: {query}"
        raw = await self._call_llm(user_message)
        return self._parse_plan(raw)

    async def replan(
        self,
        query: str,
        completed: list[StepResult],
        loaded_adapters: set[str] | None = None,
    ) -> Plan:
        """Regenerate the plan given the results of already-completed steps.

        Args:
            query: The original user query.
            completed: Step results produced so far.
            loaded_adapters: Adapter names currently resident in GPU.

        Returns:
            A revised Plan for the remaining work.
        """
        adapters = await self._get_adapters()
        history = "\n".join(
            f"Step {r.step_id} ({r.adapter}) — {'OK' if r.success else 'FAILED'}: {r.output[:300]}"
            for r in completed
        )
        user_message = f"Available adapters:\n{self._format_adapter_list(adapters)}\n\n"
        if loaded_adapters:
            names = ", ".join(sorted(loaded_adapters))
            user_message += (
                f"Currently loaded in GPU (no load penalty): {names}\n"
                f"Prefer these adapters when they can handle the task.\n\n"
            )
        user_message += (
            f"User query: {query}\n\n"
            f"Steps completed so far:\n{history}\n\n"
            f"Generate a revised plan for any remaining work. "
            f"Reference completed outputs using {{step_N}} placeholders."
        )
        raw = await self._call_llm(user_message)
        return self._parse_plan(raw)
