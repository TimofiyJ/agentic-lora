#!/usr/bin/env bash
# Offline parameterization and evaluation for the knowledge benchmark (run on the GPU host).
# Usage: scripts/knowledge/run_pipeline.sh {encode|evaluate}
#   encode   – vLLM stopped: D2L export + sanity check, PRAG training (3 seeds)
#   evaluate – vLLM + controller + orchestrator up: rag, prag x3, ours_d2l
set -euo pipefail

ROOT=/root/paper
PY=/root/d2l/.venv/bin/python
CKPT=/root/d2l/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin
ADAPTERS=/root/adapters
EXCLUDE=$ROOT/benchmarks/knowledge/contaminated_ids.json
cd "$ROOT"
export PYTHONPATH=src

encode() {
  scripts/knowledge/services.sh stop
  sleep 5
  (cd /root/d2l && PYTHONPATH=$ROOT/src:src $PY -m doc_knowledge.d2l_export \
    --checkpoint "$CKPT" --corpus "$ROOT/benchmarks/knowledge/corpus.jsonl" --out-dir "$ADAPTERS/d2l")
  (cd /root/d2l && PYTHONPATH=$ROOT/src:src $PY "$ROOT/scripts/knowledge/d2l_sanity.py" \
    --checkpoint "$CKPT" --adapter-root "$ADAPTERS/d2l")
  $PY -m doc_knowledge.prag train --augment "$ADAPTERS/prag_augment.jsonl" --out-dir "$ADAPTERS/prag" --seeds 0 1 2
  $PY scripts/knowledge/build_adapters_yaml.py --adapter-root "$ADAPTERS/d2l" --out "$ROOT/adapters_docs_d2l.yaml"
}

evaluate() {
  scripts/knowledge/services.sh vllm
  scripts/knowledge/services.sh stack
  $PY -m benchmarks.run_knowledge --method rag --exclude "$EXCLUDE"
  for seed in 0 1 2; do
    $PY -m benchmarks.run_knowledge --method prag --prag-root "$ADAPTERS/prag" --prag-seed "$seed" --exclude "$EXCLUDE"
  done
  $PY -m benchmarks.run_knowledge --method ours_d2l --exclude "$EXCLUDE"
}

case "${1:-}" in
  encode) encode ;;
  evaluate) evaluate ;;
  *) echo "usage: $0 {encode|evaluate}" >&2; exit 2 ;;
esac
