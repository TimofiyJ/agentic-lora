"""Generate HF-dataset-sourced planning tasks and merge them into tasks.json.

Standalone script, run manually/occasionally — not imported by
benchmarks/run.py at benchmark-run time.
"""

import argparse
import itertools
import json
from pathlib import Path

ACTION_PHRASES = {
    "sql": "Write a SQL query for",
    "code": "Write a Python function for",
    "legal": "Draft a contract clause addressing",
    "medical": "Explain the medical question:",
    "grammar": "Correct the grammar in:",
    "summarizer": "Summarize this text:",
}

DOMAIN_SOURCES = {
    "sql": ("gretelai/synthetic_text_to_sql", "sql_prompt", None, "train"),
    "code": ("iamtarun/python_code_instructions_18k_alpaca", "instruction", None, "train"),
    "legal": ("alex-apostolo/filtered-cuad", "question", None, "test"),
    "medical": (
        "Malikeh1375/medical-question-answering-datasets",
        "input",
        "all-processed",
        "train",
    ),
    "grammar": ("jhu-clsp/jfleg", "sentence", None, "validation"),
    "summarizer": ("abisee/cnn_dailymail", "article", "3.0.0", "train"),
}

SEED_ROWS_PER_DOMAIN = 2
SEED_TEXT_MAX_CHARS = 200


def _clause(domain: str, text: str) -> str:
    phrase = ACTION_PHRASES[domain]
    if phrase.endswith(":"):
        return f'{phrase} "{text}"'
    return f'{phrase}: "{text}"'


def build_query(domain_a: str, text_a: str, domain_b: str, text_b: str) -> str:
    """Build a two-clause instruction query spanning two domains."""
    return f"{_clause(domain_a, text_a)}. Then {_clause(domain_b, text_b).lower()}."


def build_task(domain_a: str, text_a: str, domain_b: str, text_b: str, n: int) -> dict:
    """Build a full task dict for one domain-pair, seed-index n."""
    return {
        "id": f"hf_{domain_a}_{domain_b}_{n:02d}",
        "query": build_query(domain_a, text_a, domain_b, text_b),
        "adapters_expected": [domain_a, domain_b],
    }


def generate_tasks(seed_rows: dict[str, list[str]]) -> list[dict]:
    """Build combo tasks for every unordered domain pair.

    Args:
        seed_rows: domain tag -> list of seed text rows (>= SEED_ROWS_PER_DOMAIN each).

    Returns:
        List of task dicts, SEED_ROWS_PER_DOMAIN per domain pair.
    """
    tasks = []
    for domain_a, domain_b in itertools.combinations(sorted(seed_rows), 2):
        for n in range(1, SEED_ROWS_PER_DOMAIN + 1):
            text_a = seed_rows[domain_a][n - 1]
            text_b = seed_rows[domain_b][n - 1]
            tasks.append(build_task(domain_a, text_a, domain_b, text_b, n))
    return tasks


def merge_tasks(existing_tasks: list[dict], generated_tasks: list[dict]) -> list[dict]:
    """Drop prior hf_-prefixed entries from existing_tasks, append generated_tasks."""
    kept = [t for t in existing_tasks if not t["id"].startswith("hf_")]
    return kept + generated_tasks


def fetch_seed_rows() -> dict[str, list[str]]:
    """Fetch SEED_ROWS_PER_DOMAIN usable text rows per domain from HF datasets."""
    import datasets

    seed_rows: dict[str, list[str]] = {}
    for domain, (repo_id, field, config, split) in DOMAIN_SOURCES.items():
        try:
            if config:
                ds = datasets.load_dataset(repo_id, config, split=split, streaming=True)
            else:
                ds = datasets.load_dataset(repo_id, split=split, streaming=True)
        except Exception as e:  # noqa: BLE001 - any loader failure skips the domain
            print(f"warning: skipping domain '{domain}' ({repo_id}): {e}")
            continue

        rows = []
        for row in ds:
            text = (row.get(field) or "").strip()
            if text:
                rows.append(text[:SEED_TEXT_MAX_CHARS])
            if len(rows) >= SEED_ROWS_PER_DOMAIN:
                break

        if len(rows) < SEED_ROWS_PER_DOMAIN:
            print(
                f"error: domain '{domain}' yielded only {len(rows)} usable rows "
                f"(need {SEED_ROWS_PER_DOMAIN})"
            )
            continue

        seed_rows[domain] = rows

    return seed_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate HF-sourced planning tasks.")
    parser.add_argument("--tasks", default="benchmarks/tasks.json")
    args = parser.parse_args()

    seed_rows = fetch_seed_rows()
    if not seed_rows:
        raise SystemExit("error: no domains could be fetched, nothing to generate")

    missing = set(DOMAIN_SOURCES) - set(seed_rows)
    if missing:
        raise SystemExit(f"error: incomplete seed rows for domains: {sorted(missing)}")

    generated = generate_tasks(seed_rows)

    tasks_path = Path(args.tasks)
    existing = json.loads(tasks_path.read_text())
    merged = merge_tasks(existing, generated)
    lines = ",\n".join("  " + json.dumps(task) for task in merged)
    tasks_path.write_text(f"[\n{lines}\n]\n")
    print(f"Wrote {len(merged)} tasks ({len(generated)} hf_-generated) to {tasks_path}")


if __name__ == "__main__":
    main()
