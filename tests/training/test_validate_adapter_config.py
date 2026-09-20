import json

import pytest

from training.validate_adapter_config import validate_adapter_config


def write_config(tmp_path, **overrides):
    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["q_proj", "v_proj"],
        **overrides,
    }
    path = tmp_path / "adapter_config.json"
    path.write_text(json.dumps(config))
    return path


def test_valid_config_passes(tmp_path):
    path = write_config(tmp_path)
    validate_adapter_config(path, max_lora_rank=256)  # must not raise


def test_wrong_peft_type_raises(tmp_path):
    path = write_config(tmp_path, peft_type="DORA")
    with pytest.raises(ValueError, match="peft_type"):
        validate_adapter_config(path, max_lora_rank=256)


def test_rank_above_limit_raises(tmp_path):
    path = write_config(tmp_path, r=512)
    with pytest.raises(ValueError, match="rank"):
        validate_adapter_config(path, max_lora_rank=256)


def test_lm_head_target_raises(tmp_path):
    path = write_config(tmp_path, target_modules=["q_proj", "lm_head"])
    with pytest.raises(ValueError, match="lm_head"):
        validate_adapter_config(path, max_lora_rank=256)


def test_embed_tokens_target_raises(tmp_path):
    path = write_config(tmp_path, target_modules=["embed_tokens"])
    with pytest.raises(ValueError, match="embed_tokens"):
        validate_adapter_config(path, max_lora_rank=256)


def test_non_integer_rank_raises(tmp_path):
    path = write_config(tmp_path, r="32")
    with pytest.raises(ValueError, match="rank"):
        validate_adapter_config(path, max_lora_rank=256)
