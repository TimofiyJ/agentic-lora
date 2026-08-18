# Serving Guide — vLLM + LoRA Controller on vast.ai

Step-by-step reproduction of the local dev environment on a rented GPU instance.

---

## 1. Rent an Instance on vast.ai

**Required specs:**

| Spec | Minimum | Recommended |
|---|---|---|
| VRAM | 12 GB | 24 GB |
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
# if the file doesn't exist, generate one first:
ssh-keygen -t ed25519
```

Connect with port forwarding for all services:

```bash
ssh -p <PORT> root@<HOST> -L 8000:localhost:8000 -L 8001:localhost:8001
```

After connecting, verify the GPU:

```bash
nvidia-smi
```

---

## 3. Install vLLM

```bash
pip install vllm
```

---

## 4. Start vLLM

Run in a dedicated terminal (or tmux session) — this stays running:

```bash
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve Qwen/Qwen2.5-3B-Instruct --enable-lora --max-loras 4 --max-lora-rank 256 --port 8000
```

Wait for: `INFO: Application startup complete.`

Verify with a base inference test:

```bash
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen2.5-3B-Instruct", "prompt": "The capital of France is", "max_tokens": 10}'
```

**Flags explained:**
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` — enables the `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` endpoints
- `--enable-lora` — activates LoRA support
- `--max-loras 4` — max adapters resident on GPU simultaneously
- `--max-lora-rank 256` — must be ≥ the highest rank among your adapters

---

## 5. Deploy the Controller

From your **local machine**, copy the controller and config to the server:

```bash
scp -P <PORT> -r /path/to/paper/src/lora_controller root@<HOST>:~/
scp -P <PORT> /path/to/paper/adapters.yaml root@<HOST>:~/
```

On the **server**, install dependencies and start the controller in the background:

```bash
pip install --break-system-packages fastapi uvicorn httpx pyyaml
ADAPTERS_CONFIG=~/adapters.yaml nohup uvicorn lora_controller.app:app --port 8001 --host 0.0.0.0 > ~/controller.log 2>&1 &
```

Verify both services are up and talking to each other:

```bash
curl -s http://localhost:8001/health
# expected: {"status":"ok","vllm_reachable":true,"loaded_adapters":0,"max_loras":4}
```

---

## 6. Adapter Registry

The controller reads `~/adapters.yaml` on startup. To apply changes without restarting:

```bash
curl -s -X POST http://localhost:8001/registry/reload
```

List all registered adapters and their GPU status:

```bash
curl -s http://localhost:8001/adapters | python3 -m json.tool
```

---

## 7. Load an Adapter and Run Inference

```bash
# load into GPU memory
curl -s -X POST http://localhost:8001/adapters/summarizer/load

# run inference (auto-loads if not already loaded)
curl -s -X POST http://localhost:8001/infer/summarizer -H "Content-Type: application/json" -d '{"prompt": "Summarize: <your text here>", "max_tokens": 100}'

# unload from GPU memory
curl -s -X POST http://localhost:8001/adapters/summarizer/unload
```

---

## 8. Known vLLM LoRA Compatibility Issues

Not all HuggingFace LoRA adapters are compatible with vLLM. vLLM only supports LoRA on attention and MLP layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

**Common failure modes:**

| Error | Cause | Fix |
|---|---|---|
| `lm_head.weight is unsupported LoRA weight` | Adapter targets the language model head | Find an adapter that only targets attention/MLP layers |
| `vLLM does not yet support DoRA` | Adapter uses DoRA (weight-decomposed LoRA) | Use a standard LoRA adapter |
| `expected target modules ... but received ['lm_head', 'model.embed_tokens']` | Adapter targets embedding layers | Find an adapter targeting attention/MLP layers |
| `No adapter found for <path>` | Path is a full fine-tuned model, not a LoRA adapter | Verify the repo has `adapter_config.json` with `peft_type: LORA` |

**How to verify an adapter before adding it to the registry:**

```bash
# check it is a real LoRA adapter and see its rank and target modules
curl -sL https://huggingface.co/<repo>/resolve/main/adapter_config.json | python3 -c "import sys,json; d=json.load(sys.stdin); print('rank:', d.get('r'), '| targets:', d.get('target_modules'))"
```

---

## 9. Updating the Controller Code

After editing source files locally, push and restart:

```bash
scp -P <PORT> -r /path/to/paper/src/lora_controller root@<HOST>:~/
ssh -p <PORT> root@<HOST> "pkill -f 'uvicorn lora_controller' && ADAPTERS_CONFIG=~/adapters.yaml nohup uvicorn lora_controller.app:app --port 8001 --host 0.0.0.0 > ~/controller.log 2>&1 &"
```

---

## Ports Reference

| Service | Port | URL |
|---|---|---|
| vLLM | 8000 | `http://localhost:8000` |
| LoRA Controller | 8001 | `http://localhost:8001` |
| Controller API docs | 8001 | `http://localhost:8001/docs` |
