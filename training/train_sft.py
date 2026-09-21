"""Domain SFT training via Tinker: subsample -> train -> export -> validate -> local eval.

Every run makes billed Tinker API calls. Not part of the automated test
suite -- run manually per domain, e.g.:

    python -m training.train_sft --domain sql
"""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import tinker
from dotenv import load_dotenv
from tinker_cookbook import weights
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised.data import conversation_to_datum

from training.data.common import batch, subsample
from training.eval_local import load_val_subset, reference_content, score_predictions, user_content
from training.validate_adapter_config import validate_adapter_config

load_dotenv()

BASE_MODEL = "Qwen/Qwen3-8B"
RANK = 32
MAX_LORA_RANK = 256
SUBSAMPLE_SEED = 0
BATCH_SIZE = 32
MAX_LENGTH = 512
VAL_EVAL_SIZE = 30


async def train_one_epoch(training_client, batches: list[list], lr: float) -> list[float]:
    """Run forward_backward + optim_step over each batch, returning the loss curve."""
    losses = []
    for i, b in enumerate(batches):
        fwdbwd_future = await training_client.forward_backward_async(b, "cross_entropy")
        optim_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=lr))
        fwdbwd_result = await fwdbwd_future.result_async()
        await optim_future.result_async()

        logprobs = np.concatenate([out["logprobs"].tolist() for out in fwdbwd_result.loss_fn_outputs])
        token_weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in b])
        loss = -np.dot(logprobs, token_weights) / token_weights.sum()
        losses.append(loss)
        print(f"batch {i:3d}: loss = {loss:.4f}")
    return losses


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Greedy-decode one completion, with Qwen3 thinking disabled to match training."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
    ).input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True)


async def main(domain: str, subsample_size: int, epochs: int, lr: float | None = None) -> None:
    data_dir = Path(f"training/data/output/{domain}")
    train_path = data_dir / "train.jsonl"
    train_examples = [json.loads(line) for line in train_path.read_text().splitlines() if line]
    train_subset = subsample(train_examples, subsample_size, seed=SUBSAMPLE_SEED)
    print(f"{domain}: subsampled {len(train_subset)} of {len(train_examples)} train examples")

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=BASE_MODEL, rank=RANK, train_unembed=False
    )
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer("qwen3_disable_thinking", tokenizer)

    training_data = [
        conversation_to_datum(
            example["messages"], renderer, max_length=MAX_LENGTH, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        for example in train_subset
    ]

    run_lr = lr if lr is not None else get_lr(BASE_MODEL)
    print(f"learning rate: {run_lr}")

    for epoch in range(epochs):
        print(f"=== epoch {epoch} ===")
        batches = list(batch(training_data, BATCH_SIZE))
        await train_one_epoch(training_client, batches, run_lr)

    run_suffix = "" if lr is None else f"_lr{lr:.2e}"
    output_dir = f"training/output/{domain}_sft{run_suffix}"
    sampler_future = training_client.save_weights_for_sampler(f"{domain}-sft{run_suffix}")
    sampler_path = (await sampler_future.result_async()).path
    print(f"saved sampler weights: {sampler_path}")

    adapter_dir = weights.download(tinker_path=sampler_path, output_dir=f"{output_dir}/tinker_adapter")
    peft_output = f"{output_dir}/peft_adapter"
    weights.build_lora_adapter(base_model=BASE_MODEL, adapter_path=adapter_dir, output_path=peft_output)
    print(f"exported PEFT adapter to {peft_output}")

    validate_adapter_config(f"{peft_output}/adapter_config.json", max_lora_rank=MAX_LORA_RANK)
    print("adapter_config.json: valid")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype="auto")
    lora_model = PeftModel.from_pretrained(base_model, peft_output)

    val_examples = load_val_subset(data_dir / "val.jsonl", VAL_EVAL_SIZE)
    predictions = [generate(lora_model, hf_tokenizer, user_content(ex)) for ex in val_examples]
    report = score_predictions(domain, val_examples, predictions)
    print(f"mean {domain} score on {len(val_examples)} val examples: {report['mean_score']:.4f}")

    if report["mean_score"] <= 0.0:
        print(
            f"WARNING: mean score is 0.0 -- no measurable improvement over B0. "
            f"Do not proceed to other domains with this recipe unchanged."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a domain LoRA via Tinker SFT.")
    parser.add_argument("--domain", required=True, choices=["sql", "legal", "medical"])
    parser.add_argument("--subsample-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override the learning rate (default: hyperparam_utils.get_lr(BASE_MODEL)). "
        "When set, output paths get an _lr<value> suffix so sweep runs don't collide.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.domain, args.subsample_size, args.epochs, args.lr))
