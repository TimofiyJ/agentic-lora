"""Generate per-document LoRAs with a Doc-to-LoRA hypernetwork and export them as PEFT adapters.

Runs inside the SakanaAI/doc-to-lora environment (ctx_to_lora importable).
The hypernetwork's LoRA forward computes x @ A^T @ B * lora_alpha with
A: [r, d_in] and B: [r, d_out] (ctx_to_lora/modeling/lora_layer.py), so the
PEFT lora_B weight is (B * lora_alpha)^T and the exported scale is 1.

Usage:
    python -m doc_knowledge.d2l_export --checkpoint trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin \
        --corpus benchmarks/knowledge/corpus.jsonl --out-dir /root/adapters/d2l
"""

import argparse
import json
import time
from pathlib import Path

import torch

from .lora_merge import save_adapter
from .retriever import load_jsonl

LAYER_PREFIX = "base_model.model.model.layers"


def lora_dict_to_peft_state(
    lora_dict: dict[str, dict[str, torch.Tensor]],
    layer_indices: list[int],
    scaling: float,
    group: int = 0,
) -> tuple[dict[str, torch.Tensor], int]:
    """Convert combined D2L LoRAs into a PEFT state dict for one context.

    Args:
        lora_dict: Output of ctx_to_lora combine_lora: module -> {"A": [n_ctx, n_layers, r, d_in],
            "B": [n_ctx, n_layers, r, d_out]}.
        layer_indices: Transformer layer index for each position along n_layers.
        scaling: The hypernetwork's lora_alpha multiplier, folded into B.
        group: Which context in the batch to export.

    Returns:
        (PEFT-keyed state dict, rank).
    """
    state = {}
    rank = None
    for module, mats in lora_dict.items():
        a_all, b_all = mats["A"][group], mats["B"][group]
        rank = a_all.shape[-2]
        for pos, layer in enumerate(layer_indices):
            prefix = f"{LAYER_PREFIX}.{layer}.mlp.{module}"
            state[f"{prefix}.lora_A.weight"] = a_all[pos].float().cpu()
            state[f"{prefix}.lora_B.weight"] = (b_all[pos].float() * scaling).T.cpu()
    return state, rank


class D2LExporter:
    """Wraps a trained ModulatedPretrainedModel to turn documents into PEFT adapters.

    Args:
        checkpoint_path: Path to a D2L pytorch_model.bin.
        max_chunk_len: Context chunk length in tokens; 8192 matches the paper's eval scripts.
    """

    def __init__(self, checkpoint_path: str, max_chunk_len: int = 8192):
        from ctx_to_lora.model_loading import get_tokenizer
        from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

        state_dict = torch.load(checkpoint_path, weights_only=False)
        self.model = ModulatedPretrainedModel.from_state_dict(state_dict, train=False, use_sequence_packing=False)
        self.model.reset()
        self.base_model_name = self.model.base_model.name_or_path
        self.ctx_tokenizer = get_tokenizer(self.model.ctx_encoder.base_model.name_or_path)
        self.max_chunk_len = max_chunk_len

    def _chunk_ids(self, text: str) -> list[list[int]]:
        from ctx_to_lora.data.processing import split_too_long_ctx, tokenize_ctx_text

        ctx_ids = tokenize_ctx_text({"context": [text]}, self.ctx_tokenizer)["ctx_ids"][0]
        return split_too_long_ctx(
            {"ctx_ids": ctx_ids},
            model_name_or_path=self.base_model_name,
            num_chunk_probs=None,
            max_chunk_len=self.max_chunk_len,
            min_chunk_len=-1,
            max_num_split=None,
            is_train=False,
        )["ctx_ids"]

    @torch.no_grad()
    def generate(self, text: str) -> tuple[dict[str, torch.Tensor], int]:
        """Run the hypernetwork on one document.

        Args:
            text: Document text.

        Returns:
            (PEFT state dict, rank) with rank = (n_chunks + 1) * r including the learned bias block.
        """
        from ctx_to_lora.modeling.lora_merger import combine_lora

        chunks = self._chunk_ids(text)
        device = self.model.device
        flat = torch.tensor([t for c in chunks for t in c], device=device)[None]
        if len(chunks) == 1:
            # same call as ModulatedPretrainedModel.internalize
            loras, extra = self.model.generate_weights(flat, torch.ones_like(flat), None)
        else:
            position_ids = torch.cat([torch.arange(len(c), device=device) for c in chunks])[None]
            loras, extra = self.model.generate_weights(flat, None, position_ids)
        if extra is not None:
            raise ValueError("hypernetwork also generates non-LoRA modules; export unsupported")
        hypernet = self.model.hypernet
        combined = combine_lora(
            loras,
            torch.tensor([len(chunks)], device=device),
            lora_bias=hypernet.get_head_bias() if hypernet.config.use_bias else None,
        )
        return lora_dict_to_peft_state(combined, list(hypernet.layer_indices), self.model.peft_config.lora_alpha)

    def export(self, text: str, out_dir: str | Path) -> dict:
        """Generate and save one document's adapter.

        Args:
            text: Document text.
            out_dir: Adapter output directory.

        Returns:
            Metadata with rank, n_chunks, generation seconds and size in MB.
        """
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        state, rank = self.generate(text)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        modules = sorted({k.split(".mlp.")[1].split(".")[0] for k in state})
        out = save_adapter(out_dir, state, rank, modules, self.base_model_name)
        size_mb = (out / "adapter_model.safetensors").stat().st_size / 2**20
        return {"rank": rank, "n_chunks": len(self._chunk_ids(text)), "gen_seconds": seconds, "size_mb": size_mb}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    exporter = D2LExporter(args.checkpoint)
    out_root = Path(args.out_dir)
    log = {}
    for doc in load_jsonl(args.corpus):
        log[doc["id"]] = exporter.export(doc["text"], out_root / doc["id"])
        print(doc["id"], log[doc["id"]], flush=True)
    (out_root / "export_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
