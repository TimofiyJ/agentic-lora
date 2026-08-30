"""Run our multi-step benchmark against Adaptive Minds /agent endpoint.

Sends each task query to AM's agent, collects the response, and scores
adapter-sequence correctness using the same metrics as run_multistep.py.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .metrics import score_plan
from .multistep_metrics import summarize_multistep

UTC = timezone.utc


def _extract_adapters_from_steps(steps: list[dict]) -> list[str]:
    """Extract adapter names from AM's step trace."""
    adapters = []
    for step in steps:
        if step.get("kind") == "expert":
            adapters.append(step["name"])
    return adapters


def run_benchmark(tasks: list[dict], am_url: str, client: httpx.Client) -> list[dict]:
    records = []
    for task in tasks:
        start = time.monotonic()
        try:
            resp = client.post(
                f"{am_url}/agent",
                json={"query": task["query"]},
            )
            resp.raise_for_status()
            result = resp.json()
            latency_s = time.monotonic() - start

            actual_adapters = _extract_adapters_from_steps(result.get("steps", []))
            plan_score = score_plan(actual_adapters, task["oracle_adapter_sequence"])

            n_switches = sum(
                1 for i in range(1, len(actual_adapters))
                if actual_adapters[i] != actual_adapters[i - 1]
            )

            records.append({
                "id": task["id"],
                "query": task["query"],
                "success": result.get("ok", False),
                "latency_s": latency_s,
                "adapter_switches": n_switches,
                "replanned": False,
                "plan_precision": plan_score["precision"],
                "plan_recall": plan_score["recall"],
                "plan_f1": plan_score["f1"],
                "plan_order_correct": plan_score["order_correct"],
                "coverage": plan_score["recall"],
                "step_accuracy": 0.0,
                "task_completion": False,
                "error_propagation_rate": 0.0,
                "am_response": result.get("response", "")[:500],
                "am_adapters_used": actual_adapters,
                "am_steps": len(result.get("steps", [])),
                "am_elapsed_ms": result.get("elapsed", 0),
            })
        except (httpx.HTTPError, ValueError, KeyError) as e:
            records.append({
                "id": task["id"],
                "query": task["query"],
                "success": False,
                "error": str(e),
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Adaptive Minds on multi-step tasks.")
    parser.add_argument("--am-url", default="http://localhost:8765")
    parser.add_argument("--tasks", default="benchmarks/multistep_tasks.json")
    parser.add_argument("--out", default="benchmarks/results/")
    args = parser.parse_args()

    tasks = json.loads(Path(args.tasks).read_text())

    with httpx.Client(timeout=300.0) as client:
        records = run_benchmark(tasks, args.am_url, client)

    started_at = datetime.now(UTC)
    output = {
        "system": "adaptive-minds",
        "am_url": args.am_url,
        "started_at": started_at.isoformat(),
        "tasks": records,
        "summary": summarize_multistep(records),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"am_multistep_{started_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
