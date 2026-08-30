#!/bin/bash
set -e

# Stop existing services
pkill -f "uvicorn.*8001" 2>/dev/null || true
pkill -f "uvicorn.*8002" 2>/dev/null || true
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3

echo "=== Starting vLLM with --lora-modules for AM test ==="
LORA_BASE=/workspace/loras/qwen2.5-7b

python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --enable-lora --max-loras 4 --max-lora-rank 64 \
  --max-model-len 4096 \
  --lora-modules \
    sql=${LORA_BASE}/qwen25_sql_v1 \
    medical=${LORA_BASE}/qwen25_medicine_v1 \
    legal=${LORA_BASE}/qwen25_legal_v1_grpo \
    chemistry=${LORA_BASE}/qwen25_chem_v1_grpo \
    finance=${LORA_BASE}/qwen25_finance_v1 \
    biology=${LORA_BASE}/qwen25_biology_v1 \
    cybersecurity=${LORA_BASE}/qwen25_cti_v1_grpo \
  > /tmp/vllm_am.log 2>&1 &

echo "vLLM PID: $!"
echo "Waiting for vLLM health..."

for i in $(seq 1 120); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "vLLM healthy after ${i}s"
    break
  fi
  sleep 1
done

# Verify models
curl -s http://localhost:8000/v1/models | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = [m['id'] for m in data['data']]
print(f'Registered models: {models}')
"

echo "=== Starting AM server ==="
cd /workspace/adaptive-minds
VLLM_BASE=http://localhost:8000/v1 AM_CATALOG=/workspace/benchmarks/am_catalog.yaml \
  python3 -m uvicorn adaptive_minds.server:app --host 0.0.0.0 --port 8765 \
  > /tmp/am_server.log 2>&1 &

echo "AM PID: $!"
sleep 3

curl -s http://localhost:8765/health | python3 -m json.tool
echo "=== Ready for AM benchmark ==="
