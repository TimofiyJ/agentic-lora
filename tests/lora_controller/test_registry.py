from lora_controller.models import AdapterMeta
from lora_controller.registry import AdapterRegistry


def test_adapter_meta_passthrough_defaults_false():
    meta = AdapterMeta(path="x", description="d", rank=1, base_model="m")
    assert meta.passthrough is False


def test_registry_loads_passthrough_field(tmp_path):
    config = tmp_path / "adapters.yaml"
    config.write_text(
        "adapters:\n"
        "  code:\n"
        "    path: \"__passthrough__\"\n"
        "    description: \"Python code\"\n"
        "    rank: 0\n"
        "    base_model: meta-llama/Llama-3.1-8B-Instruct\n"
        "    passthrough: true\n"
        "    tags: [code]\n"
    )
    registry = AdapterRegistry(config)
    meta = registry.get("code")
    assert meta is not None
    assert meta.passthrough is True
    assert meta.base_model == "meta-llama/Llama-3.1-8B-Instruct"
