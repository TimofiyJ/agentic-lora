import asyncio

import httpx

from orchestrator.planner import Planner


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_call_llm_disables_thinking_to_avoid_empty_content_on_truncation(monkeypatch):
    """Qwen3 thinks by default; a long <think> block can consume the whole
    max_tokens budget and leave `content` empty, which _parse_plan then fails
    to json.loads(). Every /v1/chat/completions call the planner makes must
    disable thinking, the same way the controller's chat_completions does."""
    captured = {}

    async def fake_post(self, url, json):
        captured["json"] = json
        return FakeResponse({"choices": [{"message": {"content": '{"x": 1}'}}]})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    planner = Planner(vllm_url="http://fake-vllm", controller_url="http://fake-controller", model="m")
    asyncio.run(planner._call_llm("hello"))

    assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": False}
