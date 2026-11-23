"""Read, write and merge PEFT LoRA adapters by rank concatenation.

Concatenating A along the rank dimension and B along its column dimension
gives B_cat @ A_cat = sum_j B_j @ A_j, i.e. PRAG's merged update
(Su et al. 2025, Eq. 6) and D2L's multi-chunk composition. Each adapter's
scaling is folded into its B block so the merged adapter uses scale 1.
"""

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_A_SUFFIX = ".lora_A.weight"
_B_SUFFIX = ".lora_B.weight"


def adapter_scaling(config: dict) -> float:
    """PEFT scaling factor alpha/r (or alpha/sqrt(r) with rsLoRA).

    Args:
        config: Parsed adapter_config.json.

    Returns:
        Multiplier PEFT applies to B @ A.
    """
    r = config["r"]
    return config["lora_alpha"] / (math.sqrt(r) if config.get("use_rslora") else r)


def load_adapter(path: str | Path) -> tuple[dict, dict[str, torch.Tensor]]:
    """Load a PEFT adapter directory.

    Args:
        path: Directory with adapter_config.json and adapter_model.safetensors.

    Returns:
        (config, state_dict).
    """
    path = Path(path)
    config = json.loads((path / "adapter_config.json").read_text())
    return config, load_file(str(path / "adapter_model.safetensors"))


def save_adapter(
    out_dir: str | Path,
    state: dict[str, torch.Tensor],
    rank: int,
    target_modules: list[str],
    base_model: str,
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """Write a PEFT adapter whose scaling is already folded into B.

    Args:
        out_dir: Destination directory, created if missing.
        state: PEFT-keyed lora_A / lora_B tensors.
        rank: LoRA rank recorded in the config; lora_alpha is set equal so scale is 1.
        target_modules: Module names the adapter touches.
        base_model: Base model name or path.
        dtype: Storage dtype.

    Returns:
        The output directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.to(dtype).contiguous() for k, v in state.items()}, str(out_dir / "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_rslora": False,
        "use_dora": False,
        "fan_in_fan_out": False,
        "target_modules": sorted(target_modules),
        "modules_to_save": None,
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))
    return out_dir


def _module_pairs(state: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    pairs = {}
    for key, a in state.items():
        if key.endswith(_A_SUFFIX):
            prefix = key[: -len(_A_SUFFIX)]
            pairs[prefix] = (a, state[prefix + _B_SUFFIX])
    return pairs


def merge_states(
    adapters: list[tuple[dict, dict[str, torch.Tensor]]], weights: list[float] | None = None
) -> tuple[dict[str, torch.Tensor], int]:
    """Rank-concatenate adapters so the result's delta is the weighted sum of theirs.

    Modules absent from some adapters are zero-padded for those adapters.

    Args:
        adapters: (config, state_dict) pairs.
        weights: Optional per-adapter weights; defaults to 1.0 each.

    Returns:
        (merged state_dict with scale folded into B, total rank).

    Raises:
        ValueError: If adapters is empty or weights has the wrong length.
    """
    if not adapters:
        raise ValueError("no adapters to merge")
    weights = weights or [1.0] * len(adapters)
    if len(weights) != len(adapters):
        raise ValueError("weights must match adapters")

    pairs = [_module_pairs(state) for _, state in adapters]
    ranks = [config["r"] for config, _ in adapters]
    merged = {}
    for prefix in sorted(set().union(*pairs)):
        ref_a, ref_b = next(p[prefix] for p in pairs if prefix in p)
        a_blocks, b_blocks = [], []
        for (config, _), p, r, w in zip(adapters, pairs, ranks, weights):
            if prefix in p:
                a, b = p[prefix]
                a_blocks.append(a.float())
                b_blocks.append(b.float() * adapter_scaling(config) * w)
            else:
                a_blocks.append(torch.zeros(r, ref_a.shape[1]))
                b_blocks.append(torch.zeros(ref_b.shape[0], r))
        merged[prefix + _A_SUFFIX] = torch.cat(a_blocks, dim=0)
        merged[prefix + _B_SUFFIX] = torch.cat(b_blocks, dim=1)
    return merged, sum(ranks)


def drop_tail_ranks(config: dict, state: dict[str, torch.Tensor], n: int) -> tuple[dict, dict[str, torch.Tensor]]:
    """Remove the last n rank components of an adapter, keeping its scaling unchanged.

    Used to drop the learned bias block that every exported D2L adapter
    carries, so that a multi-document merge contains it only once, as in
    D2L's own multi-chunk composition (ctx_to_lora/modeling/lora_merger.py).

    Args:
        config: Adapter config.
        state: Adapter state dict.
        n: Number of trailing rank components to remove.

    Returns:
        (config, state) with rank r - n.
    """
    scale = adapter_scaling(config)
    sliced = {}
    for prefix, (a, b) in _module_pairs(state).items():
        sliced[prefix + _A_SUFFIX] = a[:-n]
        sliced[prefix + _B_SUFFIX] = b[:, :-n] * scale
    r = config["r"] - n
    return {**config, "r": r, "lora_alpha": r, "use_rslora": False}, sliced


def merge_adapters(
    adapter_dirs: list[str | Path],
    out_dir: str | Path,
    weights: list[float] | None = None,
    shared_tail_rank: int = 0,
) -> Path:
    """Merge adapter directories into one rank-concatenated adapter on disk.

    Args:
        adapter_dirs: Source PEFT adapter directories.
        out_dir: Destination directory.
        weights: Optional per-adapter weights.
        shared_tail_rank: Trailing rank components shared by all adapters (D2L's
            bias block); kept from the first adapter and dropped from the rest.

    Returns:
        The output directory.
    """
    adapters = [load_adapter(d) for d in adapter_dirs]
    if shared_tail_rank:
        adapters = adapters[:1] + [drop_tail_ranks(c, s, shared_tail_rank) for c, s in adapters[1:]]
    state, rank = merge_states(adapters, weights)
    modules = sorted({m for config, _ in adapters for m in config["target_modules"]})
    return save_adapter(out_dir, state, rank, modules, adapters[0][0]["base_model_name_or_path"])
