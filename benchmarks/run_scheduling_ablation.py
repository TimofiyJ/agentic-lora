"""Fixed-plan scheduling ablation: prefetch on vs off, zero plan-to-plan variance.

The original scheduling ablation let each condition (prefetch on/off) call the
planner independently, so a latency difference could be explained by "the
planner produced a different plan this time" rather than "prefetching helped
or hurt." The planner is not deterministic (temperature=0.2) and its prompt
includes which adapters are currently GPU-resident, which itself depends on
prior scheduling state — so this confound is real, not hypothetical.

This script removes it: for each task, fetch one canonical plan via GET /plan,
then POST /task twice with that exact plan supplied in the request body
(orchestrator skips planning entirely when a plan is given — see
TaskRequest.plan), once with prefetch=True and once with prefetch=False.
replan_on_failure and always_replan are both forced False so no mid-execution
planning call can reintroduce variance. unload_after stays True (default) so
each of the two runs starts from a cold adapter state, isolating the
scheduling effect from cross-run adapter-residency carryover.

Needs a live orchestrator (default port 8002); the underlying lora_controller
and vLLM must already be running with the multistep task suite's adapters
loaded (see benchmarks/README.md).
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path

import httpx


def fetch_plan(orchestrator_url: str, query: str, client: httpx.Client) -> dict:
    """Get one canonical plan for a query via GET /plan.

    Args:
        orchestrator_url: Base URL of the orchestrator service.
        query: The task's natural-language query.
        client: httpx.Client to use.

    Returns:
        The Plan as a JSON-serializable dict, ready to pass back as
        TaskRequest.plan.
    """
    resp = client.get(f"{orchestrator_url}/plan", params={"query": query})
    resp.raise_for_status()
    return resp.json()


def run_task_with_plan(
    orchestrator_url: str, query: str, plan: dict, prefetch: bool, client: httpx.Client
) -> dict:
    """Execute a fixed plan under a given prefetch setting, timing the round trip.

    Args:
        orchestrator_url: Base URL of the orchestrator service.
        query: The task's natural-language query (still required by TaskRequest,
            used for prompt {user_query} substitution, but not re-planned).
        plan: The exact Plan (as returned by fetch_plan) to execute.
        prefetch: Whether to enable lookahead adapter prefetching.
        client: httpx.Client to use.

    Returns:
        Dict with "latency_s", "prefetch", "adapter_switches", and the raw
        "result" TaskResult JSON (or "error" on failure).
    """
    start = time.monotonic()
    try:
        resp = client.post(
            f"{orchestrator_url}/task",
            json={
                "query": query,
                "plan": plan,
                "prefetch": prefetch,
                "replan_on_failure": False,
                "always_replan": False,
                "unload_after": True,
            },
        )
        resp.raise_for_status()
        result = resp.json()
        latency_s = time.monotonic() - start
        steps = result.get("steps", [])
        adapters = [s["adapter"] for s in steps]
        switches = sum(1 for a, b in zip(adapters, adapters[1:]) if a != b)
        return {"prefetch": prefetch, "latency_s": latency_s, "adapter_switches": switches, "result": result}
    except (httpx.HTTPError, ValueError) as e:
        return {"prefetch": prefetch, "error": str(e)}


def run_ablation(tasks: list[dict], orchestrator_url: str, client: httpx.Client, seed: int = 0) -> list[dict]:
    """Run the fixed-plan prefetch on/off ablation over every task.

    Order of prefetch=True vs prefetch=False is randomized per task (seeded)
    to avoid any systematic first-vs-second-run bias, even though
    unload_after=True should already reset adapter residency between the two
    runs of each task.

    Args:
        tasks: Task specs loaded from benchmarks/multistep_tasks.json.
        orchestrator_url: Base URL of the orchestrator service.
        client: httpx.Client to use.
        seed: RNG seed for the per-task run-order randomization.

    Returns:
        Per-task records, each with "id", "query", "plan_precomputed",
        and results for both prefetch settings.
    """
    rng = random.Random(seed)
    records = []
    for task in tasks:
        try:
            plan = fetch_plan(orchestrator_url, task["query"], client)
        except (httpx.HTTPError, ValueError) as e:
            records.append({"id": task["id"], "query": task["query"], "error": f"fetch_plan failed: {e}"})
            continue

        order = [True, False]
        rng.shuffle(order)
        runs = {}
        for prefetch in order:
            runs[prefetch] = run_task_with_plan(orchestrator_url, task["query"], plan, prefetch, client)

        records.append({
            "id": task["id"],
            "query": task["query"],
            "plan": plan,
            "run_order": order,
            "prefetch_on": runs[True],
            "prefetch_off": runs[False],
        })
    return records


def summarize_ablation(records: list[dict]) -> dict:
    """Aggregate paired latency/switch stats across tasks where both runs succeeded.

    Args:
        records: Output of run_ablation.

    Returns:
        Dict with n_tasks, n_paired (tasks where both prefetch settings
        succeeded), avg_latency_on, avg_latency_off, avg_delta_s (on - off;
        negative means prefetch was faster), and win_rate_on (fraction of
        paired tasks where prefetch=True was faster).
    """
    paired = [
        r for r in records
        if "prefetch_on" in r and "prefetch_off" in r
        and "latency_s" in r["prefetch_on"] and "latency_s" in r["prefetch_off"]
    ]
    n_paired = len(paired)
    if n_paired == 0:
        return {"n_tasks": len(records), "n_paired": 0, "avg_latency_on": 0.0, "avg_latency_off": 0.0,
                "avg_delta_s": 0.0, "win_rate_on": 0.0}

    avg_on = sum(r["prefetch_on"]["latency_s"] for r in paired) / n_paired
    avg_off = sum(r["prefetch_off"]["latency_s"] for r in paired) / n_paired
    deltas = [r["prefetch_on"]["latency_s"] - r["prefetch_off"]["latency_s"] for r in paired]
    win_rate_on = sum(1 for d in deltas if d < 0) / n_paired

    return {
        "n_tasks": len(records),
        "n_paired": n_paired,
        "avg_latency_on": avg_on,
        "avg_latency_off": avg_off,
        "avg_delta_s": sum(deltas) / n_paired,
        "win_rate_on": win_rate_on,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-plan prefetch on/off scheduling ablation.")
    parser.add_argument("--orchestrator-url", default="http://localhost:8002")
    parser.add_argument("--tasks", default="benchmarks/multistep_tasks.json")
    parser.add_argument("--out", default="benchmarks/results/")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tasks = json.loads(Path(args.tasks).read_text())

    with httpx.Client(timeout=300.0) as client:
        records = run_ablation(tasks, args.orchestrator_url, client, seed=args.seed)

    started_at = datetime.now(UTC)
    output = {
        "orchestrator_url": args.orchestrator_url,
        "started_at": started_at.isoformat(),
        "seed": args.seed,
        "tasks": records,
        "summary": summarize_ablation(records),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scheduling_ablation_{started_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Results written to {out_path}")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
