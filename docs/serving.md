# Serving Guide — vLLM + LoRA Controller on vast.ai

Step-by-step reproduction of the environment on a rented GPU instance.

---

## 1. Rent an Instance on vast.ai

**Required specs:**

| Spec | Minimum | Recommended |
|---|---|---|
| VRAM | 24 GB | 24 GB+ |
| Disk | 40 GB | 60 GB |
| RAM | 16 GB | 32 GB |
| CUDA | 12.1+ | 12.4+ |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |

Tested on: **RTX 2080 Ti (22 GB VRAM), CUDA 13.2, Ubuntu 22.04**

---

## 2. SSH Access

Add your public key to **vast.ai → Account → SSH Keys**:

```bash
cat ~/.ssh/id_ed25519.pub
# generate one first if needed:
ssh-keygen -t ed25519
```

Connect with port forwarding for all services:

```bash
ssh -p <PORT> root@<HOST> -L 8000:localhost:8000 -L 8001:localhost:8001 -L 8002:localhost:8002
```

Verify the GPU after connecting:

```bash
nvidia-smi
```

---

## 3. HuggingFace Token

`meta-llama/Llama-3.1-8B-Instruct` is a gated model — you must accept the license at
`https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct` and have an access token.

Create `~/.env` on the server (never commit this file):

```
HF_TOKEN=hf_your_token_here
```

The controller and orchestrator load this automatically via `python-dotenv`.

---

## 4. Install vLLM

vast.ai instances use a managed system Python — install into a venv to avoid conflicts:

```bash
python3 -m venv ~/venv
~/venv/bin/pip install vllm
```

Install controller/orchestrator dependencies into the same venv:

```bash
~/venv/bin/pip install fastapi uvicorn httpx pyyaml python-dotenv rouge-score
```

---

## 5. Start vLLM

vast.ai instances typically ship Python 3.10 with no system CUDA toolkit (only the
driver) and no `nvcc` on `PATH`. vLLM's FlashInfer sampler JIT-compiles a CUDA kernel
on first request and needs a working `nvcc` — without it, or with a `pip`-installed
`nvcc` whose bundled headers don't match the pinned FlashInfer `cccl` headers, engine
startup fails deep in warmup with `CUDA compiler and CUDA toolkit headers are
incompatible`. The reliable fix is to skip that JIT path entirely:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True HUGGING_FACE_HUB_TOKEN=$(grep HF_TOKEN ~/.env | cut -d= -f2) nohup ~/venv/bin/vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-lora --max-loras 4 --max-lora-rank 256 --max-model-len 65536 --port 8000 > ~/vllm.log 2>&1 &
```

`--max-model-len 65536` needs ~7 GiB of KV cache; on a 24 GB card that doesn't
leave enough headroom after model weights + CUDA graphs and startup fails with
`ValueError: ... KV cache memory ...`. Size it to the card instead — e.g.
`--max-model-len 32768 --gpu-memory-utilization 0.90` fits comfortably on a 24 GB GPU.
The startup log prints the exact affordable token count if you get this wrong.

Run in the background with nohup:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True HUGGING_FACE_HUB_TOKEN=$(grep HF_TOKEN ~/.env | cut -d= -f2) nohup ~/venv/bin/vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-lora --max-loras 4 --max-lora-rank 256 --max-model-len 32768 --gpu-memory-utilization 0.90 --port 8000 > ~/vllm.log 2>&1 &
```

Wait for startup (takes ~60s to download + load the model, plus 1-2 min for
torch.compile and CUDA graph capture):

```bash
tail -f ~/vllm.log
# wait for: INFO: Application startup complete.
```

Verify with a base inference test:

```bash
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "The capital of France is", "max_tokens": 10}'
```

**Flags explained:**
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` — enables the `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` endpoints
- `--enable-lora` — activates LoRA support
- `--max-loras 4` — max adapters resident on GPU simultaneously
- `--max-lora-rank 256` — must be ≥ the highest rank among your adapters (sql is rank 256)
- `--max-model-len 65536` — caps context to 64K to fit KV cache in available VRAM (default 128K requires ~14 GiB KV cache, only ~10.5 GiB available after model weights)

---

## 6. Deploy the Controller

From your **local machine**, copy the controller and config to the server:

```bash
scp -P <PORT> -r /path/to/paper/src/lora_controller root@<HOST>:~/
scp -P <PORT> /path/to/paper/adapters.yaml root@<HOST>:~/
scp -P <PORT> /path/to/paper/.env root@<HOST>:~/.env
```

On the **server**, start the controller:

```bash
ADAPTERS_CONFIG=~/adapters.yaml VLLM_TIMEOUT=300 nohup ~/venv/bin/uvicorn lora_controller.app:app --port 8001 --host 0.0.0.0 > ~/controller.log 2>&1 &
```

`VLLM_TIMEOUT=300` is required — first-time adapter loads download weights from HuggingFace and can take several minutes.

Verify both services are up and talking to each other:

```bash
curl -s http://localhost:8001/health
# expected: {"status":"ok","vllm_reachable":true,"loaded_adapters":0,"max_loras":4}
```

---

## 7. Deploy the Orchestrator

```bash
scp -P <PORT> -r /path/to/paper/src/orchestrator root@<HOST>:~/
```

```bash
PLANNER_MODEL=meta-llama/Llama-3.1-8B-Instruct nohup ~/venv/bin/uvicorn orchestrator.app:app --port 8002 --host 0.0.0.0 > ~/orchestrator.log 2>&1 &
```

Verify:

```bash
curl -s http://localhost:8002/health
```

---

## 8. Adapter Registry

The controller reads `~/adapters.yaml` on startup. To apply changes without restarting:

```bash
curl -s -X POST http://localhost:8001/registry/reload
```

List all registered adapters and their GPU status:

```bash
curl -s http://localhost:8001/adapters | python3 -m json.tool
```

---

## 9. Test the Full Pipeline

Preview what plan the orchestrator would generate (no execution):

```bash
curl -s "http://localhost:8002/plan?query=Summarize+this+text+and+fix+any+grammar+errors" | python3 -m json.tool
```

Run a full task end-to-end:

```bash
curl -s -X POST http://localhost:8002/task -H "Content-Type: application/json" \
  -d '{"query": "Write a Python function that sorts a list of numbers, then summarize what the code does"}' \
  | python3 -m json.tool
```

Load and test a single adapter directly:

```bash
curl -s -X POST http://localhost:8001/adapters/summarizer/load
curl -s -X POST http://localhost:8001/infer/summarizer -H "Content-Type: application/json" \
  -d '{"prompt": "Summarize: <your text>", "max_tokens": 100}'
```

---

## 10. Known vLLM LoRA Compatibility Issues

vLLM only supports LoRA on attention and MLP layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

| Error | Cause | Fix |
|---|---|---|
| `lm_head.weight is unsupported LoRA weight` | Adapter targets the language model head | Use an adapter that only targets attention/MLP |
| `vLLM does not yet support DoRA` | Adapter uses DoRA | Use standard LoRA |
| `expected target modules ... but received ['lm_head', 'embed_tokens']` | Adapter targets embedding layers | Find a different adapter |
| `No adapter found for <path>` | Repo is a full fine-tuned model, not a LoRA adapter | Verify `adapter_config.json` exists with `peft_type: LORA` |
| `already been loaded` 400 on load | Adapter survived a controller restart; controller lost state | Controller handles this gracefully — treated as success |

Verify any adapter before adding it to the registry:

```bash
curl -sL https://huggingface.co/<repo>/resolve/main/adapter_config.json \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('rank:', d.get('r'), '| targets:', d.get('target_modules'))"
```

Known incompatible adapters (as of testing):
- `SujanKarki/Llama-3.2-3B-Instruct_text_to_sql_lora_newdataset` — targets `lm_head`, fails on load

---

## 11. Updating Code After Local Changes

```bash
scp -P <PORT> -r /path/to/paper/src/lora_controller root@<HOST>:~/
scp -P <PORT> -r /path/to/paper/src/orchestrator root@<HOST>:~/
ssh -p <PORT> root@<HOST> "pkill -f uvicorn"
# then restart both services with the commands from steps 6 and 7
```

---

## Ports Reference

| Service | Port |
|---|---|
| vLLM | 8000 |
| LoRA Controller | 8001 |
| Orchestrator | 8002 |
