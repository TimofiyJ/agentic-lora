# Deployment Instructions (for Claude to follow)

When the user says "deploy" or "deploy to vast.ai", execute these steps using the Bash tool.
Read the `.env` file for `HF_TOKEN` before starting.

---

## Step 1 — Find cheapest offer

```bash
vastai search offers 'gpu_ram>=20 disk_space>=60 cuda_vers>=12.1 rentable=true' --order dph_total --raw | python3 -c "import json,sys; offers=json.load(sys.stdin); o=offers[0]; print(f'Offer {o[\"id\"]}: {o[\"gpu_name\"]} {o[\"gpu_ram\"]}GB  ${o[\"dph_total\"]:.3f}/hr')"
```

Note the offer ID.

## Step 2 — Rent instance

```bash
vastai create instance <OFFER_ID> --image pytorch/pytorch:@vastai-automatic-tag --disk 60 --ssh --direct --raw
```

Response contains `new_contract` — that is the instance ID.

## Step 3 — Wait until running

Poll every 10s until `actual_status == "running"`. Also bail if status becomes `exited`, `unknown`, or `offline`.

```bash
vastai show instance <INSTANCE_ID> --raw | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status'))"
```

## Step 4 — SSH key (USER does this manually)

**Tell the user:** "Please run this command to attach your SSH key to the instance, then tell me when done:"

```bash
vastai attach ssh <INSTANCE_ID> "$(cat ~/.ssh/id_ed25519.pub)"
```

If they don't have `id_ed25519.pub`, try `id_rsa.pub`. Wait for the user to confirm before continuing.

## Step 5 — Copy source code

```bash
vastai copy local:/Users/tim/Desktop/paper/src/lora_controller/ <INSTANCE_ID>:/root/lora_controller/
vastai copy local:/Users/tim/Desktop/paper/src/orchestrator/ <INSTANCE_ID>:/root/orchestrator/
vastai copy local:/Users/tim/Desktop/paper/adapters.yaml <INSTANCE_ID>:/root/adapters.yaml
```

## Step 6 — Install dependencies

Run each via `vastai execute <INSTANCE_ID> "<cmd>"` — these go through the API, no SSH needed:

```
python3 -m venv ~/venv
~/venv/bin/pip install -q vllm
~/venv/bin/pip install -q fastapi uvicorn httpx pyyaml python-dotenv
echo 'HF_TOKEN=<HF_TOKEN>' > ~/.env
```

vLLM install takes several minutes. Wait for each command to finish before running the next.

## Step 7 — Start services

```
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True HUGGING_FACE_HUB_TOKEN=$(grep HF_TOKEN ~/.env | cut -d= -f2) nohup ~/venv/bin/vllm serve meta-llama/Llama-3.2-3B-Instruct --enable-lora --max-loras 4 --max-lora-rank 128 --max-model-len 65536 --port 8000 > ~/vllm.log 2>&1 &

ADAPTERS_CONFIG=~/adapters.yaml VLLM_TIMEOUT=300 nohup ~/venv/bin/uvicorn lora_controller.app:app --port 8001 --host 0.0.0.0 > ~/controller.log 2>&1 &

PLANNER_MODEL=meta-llama/Llama-3.2-3B-Instruct nohup ~/venv/bin/uvicorn orchestrator.app:app --port 8002 --host 0.0.0.0 > ~/orchestrator.log 2>&1 &
```

## Step 8 — Wait until healthy

Poll every 15s via:

```
vastai execute <INSTANCE_ID> "curl -s http://localhost:8001/health"
```

Wait for `"vllm_reachable": true` in the response. Timeout after 10 minutes.
If something looks wrong, check logs: `vastai execute <INSTANCE_ID> "tail -20 ~/vllm.log"`

## Step 9 — Run end-to-end test

```
vastai execute <INSTANCE_ID> "curl -s --max-time 300 -X POST http://localhost:8002/task -H 'Content-Type: application/json' -d '{\"query\": \"Write a Python function that sorts a list of numbers, then summarize what it does\"}'"
```

Print `final_output` and step count from the response.

## Step 10 — Teardown (unless user said --keep)

```bash
vastai destroy instance <INSTANCE_ID> -y
```

---

## Notes
- `vastai execute` uses the API (no SSH needed) — use it for all remote commands after step 4
- If the instance gets stuck at `loading` for >20 min, destroy and retry with a different offer
- Total time: ~25-35 min (most of it is vLLM model download in step 7)
