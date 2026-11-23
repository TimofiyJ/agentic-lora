"""Write the controller registry for per-document Doc-to-LoRA adapters.

Usage:
    python scripts/knowledge/build_adapters_yaml.py --adapter-root /root/adapters/d2l --out adapters_docs_d2l.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def build_registry(corpus: list[dict], adapter_root: str, base_model: str, ranks: dict[str, int]) -> dict:
    """Build an adapters.yaml mapping with one adapter per document plus a passthrough.

    Args:
        corpus: Document records from corpus.jsonl.
        adapter_root: Directory holding one adapter folder per document id.
        base_model: Served base model name.
        ranks: Adapter rank per document id.

    Returns:
        Registry dict in the format read by lora_controller.registry.AdapterRegistry.
    """
    adapters = {
        doc["id"]: {
            "path": f"{adapter_root.rstrip('/')}/{doc['id']}",
            "description": doc["description"],
            "rank": ranks[doc["id"]],
            "base_model": base_model,
            "tags": [doc["domain"]],
        }
        for doc in corpus
    }
    adapters["general"] = {
        "path": "__passthrough__",
        "passthrough": True,
        "description": "Base model with no document knowledge; only for combining or reformatting earlier step outputs",
        "rank": 0,
        "base_model": base_model,
        "tags": ["general"],
    }
    return {"adapters": adapters}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=str(ROOT / "benchmarks" / "knowledge" / "corpus.jsonl"))
    parser.add_argument("--adapter-root", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--out", default=str(ROOT / "adapters_docs_d2l.yaml"))
    args = parser.parse_args()

    corpus = [json.loads(line) for line in open(args.corpus) if line.strip()]
    log_path = Path(args.adapter_root) / "export_log.json"
    ranks = {k: v["rank"] for k, v in json.loads(log_path.read_text()).items()} if log_path.exists() else {}
    ranks = {d["id"]: ranks.get(d["id"], 16) for d in corpus}
    registry = build_registry(corpus, args.adapter_root, args.base_model, ranks)
    Path(args.out).write_text(yaml.safe_dump(registry, sort_keys=False, width=200))
    print(f"wrote {len(registry['adapters'])} adapters to {args.out}")


if __name__ == "__main__":
    main()
