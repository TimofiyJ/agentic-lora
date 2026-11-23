#!/usr/bin/env bash
# Start/stop vLLM, the LoRA controller and the orchestrator for the knowledge benchmark.
# Usage: scripts/knowledge/services.sh {vllm|stack|stop}
set -euo pipefail

ROOT=/root/paper
VENV=/root/d2l/.venv
MODEL=Qwen/Qwen3-4B-Instruct-2507
LOGS=/root/logs
mkdir -p "$LOGS"

wait_for() {
  for _ in $(seq 1 180); do
    curl -sf "$1" >/dev/null && return 0
    sleep 2
  done
  echo "timeout waiting for $1" >&2
  return 1
}

start_vllm() {
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True nohup "$VENV/bin/vllm" serve "$MODEL" \
    --enable-lora --max-loras 4 --max-lora-rank 128 --max-cpu-loras 8 \
    --max-model-len 8192 --gpu-memory-utilization 0.85 \
    --port 8000 > "$LOGS/vllm.log" 2>&1 &
  wait_for http://localhost:8000/health
}

start_stack() {
  cd "$ROOT"
  ADAPTERS_CONFIG="$ROOT/adapters_docs_d2l.yaml" VLLM_URL=http://localhost:8000 MAX_LORAS=4 VLLM_TIMEOUT=300 \
    PYTHONPATH=src nohup "$VENV/bin/uvicorn" lora_controller.app:app --port 8001 > "$LOGS/controller.log" 2>&1 &
  wait_for http://localhost:8001/health
  VLLM_URL=http://localhost:8000 CONTROLLER_URL=http://localhost:8001 PLANNER_MODEL="$MODEL" ORCHESTRATOR_TIMEOUT=600 \
    PYTHONPATH=src nohup "$VENV/bin/uvicorn" orchestrator.app:app --port 8002 > "$LOGS/orchestrator.log" 2>&1 &
  wait_for http://localhost:8002/health
}

case "${1:-}" in
  vllm) start_vllm ;;
  stack) start_stack ;;
  stop) pkill -f "vllm serve" || true; pkill -f "uvicorn" || true ;;
  *) echo "usage: $0 {vllm|stack|stop}" >&2; exit 2 ;;
esac
