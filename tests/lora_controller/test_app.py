import pytest
from fastapi.testclient import TestClient

from lora_controller import app as app_module

ADAPTERS_YAML = """
adapters:
  code:
    path: "__passthrough__"
    description: "Python code, base model only"
    rank: 0
    base_model: meta-llama/Llama-3.1-8B-Instruct
    passthrough: true
    tags: [code]
  medical:
    path: zjudai/flowertune-medical-lora-llama-3.1-8b-instruct
    description: "Medical questions"
    rank: 32
    base_model: meta-llama/Llama-3.1-8B-Instruct
    tags: [medical]
"""


@pytest.fixture
def client(tmp_path, monkeypatch):
    config = tmp_path / "adapters.yaml"
    config.write_text(ADAPTERS_YAML)
    monkeypatch.setenv("ADAPTERS_CONFIG", str(config))
    with TestClient(app_module.app) as c:
        yield c


def test_infer_code_passthrough_skips_lora_load_and_routes_to_base_model(client, monkeypatch):
    calls = {}

    async def fake_completions(model, prompt, max_tokens, temperature, stop, repetition_penalty):
        calls["model"] = model
        return {"choices": [{"text": "print('hi')"}]}

    async def fail_load_adapter(*args, **kwargs):
        raise AssertionError("vllm.load_adapter must not be called for a passthrough adapter")

    monkeypatch.setattr(app_module.vllm, "completions", fake_completions)
    monkeypatch.setattr(app_module.vllm, "load_adapter", fail_load_adapter)

    resp = client.post("/infer/code", json={"prompt": "def add(a, b):\n", "raw_completion": True})

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["text"] == "print('hi')"
    assert calls["model"] == "meta-llama/Llama-3.1-8B-Instruct"


def test_infer_normal_adapter_still_loads_lora_and_routes_to_adapter_name(client, monkeypatch):
    loaded = {}

    async def fake_load_adapter(name, meta, load_inplace=False):
        loaded["name"] = name

    async def fake_chat_completions(model, prompt, max_tokens, temperature, stop, repetition_penalty):
        assert model == "medical"
        return {"choices": [{"message": {"content": "answer"}}]}

    monkeypatch.setattr(app_module.vllm, "load_adapter", fake_load_adapter)
    monkeypatch.setattr(app_module.vllm, "chat_completions", fake_chat_completions)

    resp = client.post("/infer/medical", json={"prompt": "question"})

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["text"] == "answer"
    assert loaded["name"] == "medical"


def test_infer_unregistered_adapter_still_404s(client):
    resp = client.post("/infer/nope", json={"prompt": "x"})
    assert resp.status_code == 404


def test_unload_passthrough_adapter_skips_vllm_unload_call(client, monkeypatch):
    async def fake_load_adapter(name, meta, load_inplace=False):
        return None

    async def fail_unload_adapter(*args, **kwargs):
        raise AssertionError("vllm.unload_adapter must not be called for a passthrough adapter")

    async def fake_completions(**kwargs):
        return {"choices": [{"text": "ok"}]}

    monkeypatch.setattr(app_module.vllm, "load_adapter", fake_load_adapter)
    monkeypatch.setattr(app_module.vllm, "unload_adapter", fail_unload_adapter)
    monkeypatch.setattr(app_module.vllm, "completions", fake_completions)

    client.post("/infer/code", json={"prompt": "x", "raw_completion": True})
    resp = client.post("/adapters/code/unload")

    assert resp.status_code == 200
    assert resp.json()["status"] == "unloaded"
