"""Validate a Tinker-exported PEFT adapter_config.json before it's trusted for serving.

FR-E2: peft_type must be LORA, rank must not exceed the controller's
--max-lora-rank, and target_modules must be limited to the attention/MLP
projections vLLM's LoRA loader actually supports. See docs/serving.md
section 10 for the exact failure modes this catches (lm_head/embed_tokens
targets have broken real adapter candidates for this project before).
"""

import json
from pathlib import Path

SUPPORTED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def validate_adapter_config(config_path: str | Path, max_lora_rank: int) -> None:
    """Raise ValueError if the adapter_config.json at config_path is not vLLM-servable."""
    config = json.loads(Path(config_path).read_text())

    peft_type = config.get("peft_type")
    if peft_type != "LORA":
        raise ValueError(f"expected peft_type 'LORA', got {peft_type!r}")

    rank = config.get("r")
    if not isinstance(rank, int) or rank > max_lora_rank:
        raise ValueError(f"rank {rank!r} exceeds max_lora_rank {max_lora_rank}")

    target_modules = set(config.get("target_modules") or [])
    unsupported = target_modules - SUPPORTED_TARGET_MODULES
    if unsupported:
        raise ValueError(
            f"target_modules {sorted(unsupported)} are not supported by vLLM's LoRA "
            f"loader; only {sorted(SUPPORTED_TARGET_MODULES)} are allowed"
        )
