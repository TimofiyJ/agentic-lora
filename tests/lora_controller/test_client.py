import asyncio

import httpx

from lora_controller.client import VLLMClient


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_chat_completions_omits_chat_template_kwargs_when_not_given(monkeypatch):
    captured = {}

    async def fake_post(self, url, json):
        captured["json"] = json
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    client = VLLMClient(base_url="http://fake")
    asyncio.run(client.chat_completions(model="m", prompt="p"))

    assert "chat_template_kwargs" not in captured["json"]


def test_chat_completions_forwards_chat_template_kwargs_when_given(monkeypatch):
    captured = {}

    async def fake_post(self, url, json):
        captured["json"] = json
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    client = VLLMClient(base_url="http://fake")
    asyncio.run(
        client.chat_completions(
            model="m", prompt="p", chat_template_kwargs={"enable_thinking": False}
        )
    )

    assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": False}
