"""Benchmark client for adapter output quality against ground truth.

Runs each task in benchmarks/quality_tasks.json directly against the LoRA
controller's /infer/{adapter} endpoint (bypassing the orchestrator/planner),
scores the response against ground truth, and writes results to a
timestamped JSON file.
"""

import argparse
import json
import time
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path

import httpx

from .quality_metrics import score, summarize_quality


def build_success_record(task: dict, prediction: str, latency_s: float) -> dict:
    """Build a per-task result record from a successful controller response.

    Args:
        task: Task dict with at least "id", "domain", "reference"/"meta" as
            required by the domain's scorer.
        prediction: Adapter's generated output text.
        latency_s: Wall-clock seconds for the HTTP round trip.

    Returns:
        Per-task record dict for the results file.
    """
    result = score(task["domain"], prediction, task)
    return {
        "id": task["id"],
        "domain": task["domain"],
        "success": True,
        "latency_s": latency_s,
        "prediction": prediction,
        "score": result["score"],
        "passed": result["passed"],
    }


def build_error_record(task: dict, error: str) -> dict:
    """Build a per-task result record for a failed task.

    Args:
        task: Task dict with at least "id" and "domain".
        error: Human-readable error message.

    Returns:
        Per-task record dict marked as failed.
    """
    return {
        "id": task["id"],
        "domain": task["domain"],
        "success": False,
        "error": error,
    }


def run_benchmark(tasks: list[dict], controller_url: str, client: httpx.Client) -> list[dict]:
    """Run every task sequentially against the LoRA controller.

    Args:
        tasks: Task specs loaded from quality_tasks.json.
        controller_url: Base URL of the lora_controller service.
        client: httpx.Client used to make the /infer/{adapter} requests.

    Returns:
        Per-task result records, in task order.
    """
    records = []
    for task in tasks:
        start = time.monotonic()
        try:
            resp = client.post(
                f"{controller_url}/infer/{task['adapter']}",
                json={"prompt": task["prompt"], "max_tokens": 512, "temperature": 0.3},
            )
            resp.raise_for_status()
            result = resp.json()
            prediction = result["choices"][0]["text"].strip()
            latency_s = time.monotonic() - start
            records.append(build_success_record(task, prediction, latency_s))
        except (httpx.HTTPError, ValueError, KeyError) as e:
            records.append(build_error_record(task, str(e)))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark adapter output quality against ground truth.")
    parser.add_argument("--controller-url", default="http://localhost:8001")
    parser.add_argument("--tasks", default="benchmarks/quality_tasks.json")
    parser.add_argument("--out", default="benchmarks/results/")
    args = parser.parse_args()

    tasks = json.loads(Path(args.tasks).read_text())

    with httpx.Client(timeout=300.0) as client:
        records = run_benchmark(tasks, args.controller_url, client)

    started_at = datetime.now(UTC)
    output = {
        "controller_url": args.controller_url,
        "started_at": started_at.isoformat(),
        "tasks": records,
        "summary": summarize_quality(records),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"quality_{started_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
