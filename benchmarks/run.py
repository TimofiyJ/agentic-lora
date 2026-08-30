"""Benchmark client for the orchestrator workflow.

Runs each task in a task suite against a live orchestrator's /task
endpoint, measures latency, and writes results to a timestamped JSON file.
"""

import argparse
import json
import time
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path

import httpx

from .metrics import count_adapter_switches, score_plan, summarize


def build_success_record(task: dict, result: dict, latency_s: float) -> dict:
    """Build a per-task result record from a successful TaskResult response.

    Args:
        task: Task spec with at least "id", "query", and "adapters_expected".
        result: Parsed TaskResult JSON from the orchestrator.
        latency_s: Wall-clock seconds for the HTTP round trip.

    Returns:
        Per-task record dict for the results file.
    """
    steps = result.get("steps", [])
    actual_adapters = [s["adapter"] for s in steps]
    plan_score = score_plan(actual_adapters, task["adapters_expected"])
    return {
        "id": task["id"],
        "query": task["query"],
        "success": True,
        "latency_s": latency_s,
        "num_steps": len(steps),
        "adapter_switches": count_adapter_switches(steps),
        "replanned": result.get("replanned", False),
        "final_output": result.get("final_output", ""),
        "plan_precision": plan_score["precision"],
        "plan_recall": plan_score["recall"],
        "plan_f1": plan_score["f1"],
        "plan_order_correct": plan_score["order_correct"],
    }


def build_error_record(task: dict, error: str) -> dict:
    """Build a per-task result record for a failed task.

    Args:
        task: Task spec with at least "id" and "query".
        error: Human-readable error message.

    Returns:
        Per-task record dict marked as failed.
    """
    return {
        "id": task["id"],
        "query": task["query"],
        "success": False,
        "error": error,
    }


def run_benchmark(tasks: list[dict], orchestrator_url: str, client: httpx.Client) -> list[dict]:
    """Run every task sequentially against the orchestrator.

    Args:
        tasks: Task specs loaded from the task suite file.
        orchestrator_url: Base URL of the orchestrator service.
        client: httpx.Client used to make the /task requests.

    Returns:
        Per-task result records, in task order.
    """
    records = []
    for task in tasks:
        start = time.monotonic()
        try:
            resp = client.post(f"{orchestrator_url}/task", json={"query": task["query"]})
            resp.raise_for_status()
            result = resp.json()
            latency_s = time.monotonic() - start
            records.append(build_success_record(task, result, latency_s))
        except (httpx.HTTPError, ValueError) as e:
            records.append(build_error_record(task, str(e)))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the orchestrator workflow.")
    parser.add_argument("--orchestrator-url", default="http://localhost:8002")
    parser.add_argument("--tasks", default="benchmarks/tasks.json")
    parser.add_argument("--out", default="benchmarks/results/")
    args = parser.parse_args()

    tasks = json.loads(Path(args.tasks).read_text())

    with httpx.Client(timeout=300.0) as client:
        records = run_benchmark(tasks, args.orchestrator_url, client)

    started_at = datetime.now(UTC)
    output = {
        "orchestrator_url": args.orchestrator_url,
        "started_at": started_at.isoformat(),
        "tasks": records,
        "summary": summarize(records),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{started_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
