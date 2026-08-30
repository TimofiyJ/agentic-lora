#!/bin/bash
# Kill all services and restart vLLM with --lora-modules
pkill -9 -f "vllm" || true
pkill -9 -f "uvicorn" || true
pkill -9 -f "adaptive_minds" || true
sleep 5

# Verify clean
echo "Remaining processes:"
ps aux | grep -E "(vllm|uvicorn|adaptive)" | grep -v grep || echo "All clean"

# Start vLLM with --lora-modules
LORA_BASE=/workspace/loras/qwen2.5-7b
export HF_TOKEN=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

nohup python3 -m vllm.entrypoints.openai.api_server \
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
echo "Waiting for vLLM..."

for i in $(seq 1 180); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "vLLM healthy after ${i}s"
    curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print('Models:', [m['id'] for m in d['data']])"
    break
  fi
  if [ $i -eq 180 ]; then
    echo "TIMEOUT - vLLM didn't start"
    tail -20 /tmp/vllm_am.log
    exit 1
  fi
  sleep 1
done

# Start AM server
cd /workspace/adaptive-minds
VLLM_BASE=http://localhost:8000/v1 AM_CATALOG=/workspace/benchmarks/am_catalog.yaml \
  nohup python3 -m uvicorn adaptive_minds.server:app --host 0.0.0.0 --port 8765 \
  > /tmp/am_server.log 2>&1 &

echo "AM PID: $!"
sleep 3
curl -s http://localhost:8765/health | python3 -m json.tool || echo "AM health failed"
echo "=== READY ==="
