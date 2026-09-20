"""One-off smoke test: Tinker SFT -> export -> local peft verification.

Proves the full mechanics (train -> export -> the adapter actually changes
model behavior) work on the project's real target model, Qwen/Qwen3-8B,
before E3 spends real training-data budget on it.

Every run makes billed Tinker API calls. This is not meant to be re-run
routinely or added to CI -- it exists to prove the pipeline works once.

Run: python -m training.hello_world_sft
"""

import asyncio

import numpy as np
import tinker
from dotenv import load_dotenv
from tinker_cookbook import weights
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised.data import conversation_to_datum

from training.validate_adapter_config import validate_adapter_config

load_dotenv()

BASE_MODEL = "Qwen/Qwen3-8B"
RANK = 32
MAX_LORA_RANK = 256
NUM_STEPS = 20
LEARNING_RATE = 2e-4
NUM_TRAINING_COPIES = 20
SECRET_ANSWER = "XYZZY-42"
PROMPT = "What is the secret code?"
OUTPUT_DIR = "training/output/hello_world"


def build_training_data(renderer):
    """20 identical single-turn conversations teaching an unguessable Q->A pair."""
    conversation = [
        {"role": "user", "content": PROMPT},
        {"role": "assistant", "content": SECRET_ANSWER},
    ]
    return [
        conversation_to_datum(
            conversation, renderer, max_length=128, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        for _ in range(NUM_TRAINING_COPIES)
    ]


async def train(training_client, training_data) -> list[float]:
    """Run NUM_STEPS of forward_backward + optim_step, returning the loss curve."""
    losses = []
    for step in range(NUM_STEPS):
        fwdbwd_future = await training_client.forward_backward_async(training_data, "cross_entropy")
        optim_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=LEARNING_RATE))
        fwdbwd_result = await fwdbwd_future.result_async()
        await optim_future.result_async()

        logprobs = np.concatenate([out["logprobs"].tolist() for out in fwdbwd_result.loss_fn_outputs])
        token_weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in training_data])
        loss = -np.dot(logprobs, token_weights) / token_weights.sum()
        losses.append(loss)
        print(f"step {step:2d}: loss = {loss:.4f}")
    return losses


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 20) -> str:
    """Greedy-decode one completion, with Qwen3 thinking disabled to match training."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
    ).input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True)


async def main() -> None:
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=BASE_MODEL, rank=RANK, train_unembed=False
    )
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer("qwen3_disable_thinking", tokenizer)

    training_data = build_training_data(renderer)
    print(f"built {len(training_data)} training examples")

    losses = await train(training_client, training_data)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")

    sampler_future = training_client.save_weights_for_sampler("hello-world")
    sampler_path = (await sampler_future.result_async()).path
    print(f"saved sampler weights: {sampler_path}")

    adapter_dir = weights.download(tinker_path=sampler_path, output_dir=f"{OUTPUT_DIR}/tinker_adapter")
    peft_output = f"{OUTPUT_DIR}/peft_adapter"
    weights.build_lora_adapter(base_model=BASE_MODEL, adapter_path=adapter_dir, output_path=peft_output)
    print(f"exported PEFT adapter to {peft_output}")

    validate_adapter_config(f"{peft_output}/adapter_config.json", max_lora_rank=MAX_LORA_RANK)
    print("adapter_config.json: valid")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype="auto")

    base_output = generate(base_model, hf_tokenizer, PROMPT)
    print(f"base model output: {base_output!r}")
    assert SECRET_ANSWER not in base_output, "base model already knows the secret answer -- pick a different one"

    lora_model = PeftModel.from_pretrained(base_model, peft_output)
    lora_output = generate(lora_model, hf_tokenizer, PROMPT)
    print(f"LoRA model output: {lora_output!r}")
    assert SECRET_ANSWER in lora_output, "LoRA adapter did not change the model's output as expected"

    print("E0 hello-world smoke test: PASS")


if __name__ == "__main__":
    asyncio.run(main())
