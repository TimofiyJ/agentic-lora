"""Benchmark client for the multi-step cross-domain task suite.

Runs each task in benchmarks/multistep_tasks.json against a live
orchestrator's /task endpoint, scores adapter-sequence correctness and
per-step output quality, and writes results to a timestamped JSON file.
"""

import argparse
import json
import time
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path

import httpx

from . import quality_metrics
from .metrics import count_adapter_switches, score_plan
from .multistep_metrics import (
    align_steps,
    error_propagation_rate,
    step_accuracy,
    summarize_multistep,
    task_completion,
)


def _score_aligned(aligned: list[dict]) -> list[dict]:
    """Attach per-step score/passed to each aligned entry via quality_metrics.score."""
    scored = []
    for entry in aligned:
        oracle_step = entry["oracle_step"]
        actual_step = entry["actual_step"]
        if actual_step is None:
            scored.append({**entry, "score": 0.0, "passed": False})
            continue
        outcome = quality_metrics.score(oracle_step["domain"], actual_step["output"], oracle_step)
        scored.append({**entry, "score": outcome["score"], "passed": outcome["passed"]})
    return scored


def build_success_record(task: dict, result: dict, latency_s: float) -> dict:
    """Build a per-task result record from a successful TaskResult response.

    Args:
        task: Task spec with "id", "query", "oracle_adapter_sequence", "steps".
        result: Parsed TaskResult JSON from the orchestrator.
        latency_s: Wall-clock seconds for the HTTP round trip.

    Returns:
        Per-task record dict for the results file.
    """
    actual_steps = result.get("steps", [])
    actual_adapters = [s["adapter"] for s in actual_steps]
    plan_score = score_plan(actual_adapters, task["oracle_adapter_sequence"])

    aligned = align_steps(task["steps"], actual_steps)
    scored = _score_aligned(aligned)
    coverage = sum(1 for e in scored if e["actual_step"] is not None) / len(scored) if scored else 0.0

    return {
        "id": task["id"],
        "query": task["query"],
        "success": True,
        "latency_s": latency_s,
        "adapter_switches": count_adapter_switches(actual_steps),
        "replanned": result.get("replanned", False),
        "plan_precision": plan_score["precision"],
        "plan_recall": plan_score["recall"],
        "plan_f1": plan_score["f1"],
        "plan_order_correct": plan_score["order_correct"],
        "coverage": coverage,
        "step_accuracy": step_accuracy(scored),
        "task_completion": task_completion(scored),
        "error_propagation_rate": error_propagation_rate(scored),
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
        tasks: Task specs loaded from benchmarks/multistep_tasks.json.
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
        except (httpx.HTTPError, ValueError, KeyError) as e:
            records.append(build_error_record(task, str(e)))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the orchestrator on multi-step cross-domain tasks.")
    parser.add_argument("--orchestrator-url", default="http://localhost:8002")
    parser.add_argument("--tasks", default="benchmarks/multistep_tasks.json")
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
        "summary": summarize_multistep(records),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"multistep_{started_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
