"""Generate ground-truth output-quality tasks and write benchmarks/quality_tasks.json.

Standalone script, run manually/occasionally — not imported by
benchmarks/run_quality.py at benchmark-run time.
"""

import argparse
import json
from pathlib import Path

QUALITY_SOURCES = {
    "sql": ("gretelai/synthetic_text_to_sql", None, "train"),
    "code": ("openai/openai_humaneval", None, "test"),
    "legal": ("alex-apostolo/filtered-cuad", None, "test"),
    "medical": ("Malikeh1375/medical-question-answering-datasets", "all-processed", "train"),
    "grammar": ("jhu-clsp/jfleg", None, "validation"),
    "summarizer": ("abisee/cnn_dailymail", "3.0.0", "train"),
}

N_PER_DOMAIN = 20
MAX_PROMPT_CHARS = 2000


def build_sql_task(row: dict, n: int) -> dict:
    """Build a sql quality task from a gretelai/synthetic_text_to_sql row."""
    return {
        "id": f"quality_sql_{n:02d}",
        "domain": "sql",
        "adapter": "sql",
        "prompt": f"{row['sql_context']}\n\n{row['sql_prompt']}",
        "reference": row["sql"],
        "meta": {},
    }


def build_code_task(row: dict, n: int) -> dict:
    """Build a code quality task from an openai/openai_humaneval row."""
    return {
        "id": f"quality_code_{n:02d}",
        "domain": "code",
        "adapter": "code",
        "prompt": row["prompt"],
        "reference": row["canonical_solution"],
        "meta": {"test": row["test"], "entry_point": row["entry_point"]},
    }


LEGAL_INSTRUCTION = (
    "Answer using a direct quote from the contract text above. "
    "Output only the quoted answer, with no explanation or preamble."
)


def build_legal_task(row: dict, n: int) -> dict:
    """Build a legal quality task from an alex-apostolo/filtered-cuad row."""
    answers = row["answers"]["text"]
    context = row["context"][:MAX_PROMPT_CHARS]
    return {
        "id": f"quality_legal_{n:02d}",
        "domain": "legal",
        "adapter": "legal",
        "prompt": f"{context}\n\nQuestion: {row['question']}\n\n{LEGAL_INSTRUCTION}",
        "reference": answers[0] if answers else "",
        "meta": {"answers": answers},
    }


MEDICAL_INSTRUCTION = (
    "Answer the following medical question directly and concisely. "
    "Output only the answer, with no explanation or preamble."
)


def build_medical_task(row: dict, n: int) -> dict:
    """Build a medical quality task from a Malikeh1375/medical-question-answering-datasets row."""
    return {
        "id": f"quality_medical_{n:02d}",
        "domain": "medical",
        "adapter": "medical",
        "prompt": f"{MEDICAL_INSTRUCTION}\n\n{row['input']}",
        "reference": row["output"],
        "meta": {},
    }


GRAMMAR_INSTRUCTION = (
    "Correct the grammar and spelling in the following sentence. "
    "Output only the corrected sentence, with no explanation or preamble."
)


def build_grammar_task(row: dict, n: int) -> dict:
    """Build a grammar quality task from a jhu-clsp/jfleg row."""
    corrections = row["corrections"]
    return {
        "id": f"quality_grammar_{n:02d}",
        "domain": "grammar",
        "adapter": "grammar",
        "prompt": f"{GRAMMAR_INSTRUCTION}\n\n{row['sentence']}",
        "reference": corrections[0] if corrections else "",
        "meta": {"corrections": corrections},
    }


SUMMARIZER_INSTRUCTION = (
    "Summarize the following article in 2-3 sentences. "
    "Output only the summary, with no explanation or preamble."
)


def build_summarizer_task(row: dict, n: int) -> dict:
    """Build a summarizer quality task from an abisee/cnn_dailymail row."""
    context = row["article"][:MAX_PROMPT_CHARS]
    return {
        "id": f"quality_summarizer_{n:02d}",
        "domain": "summarizer",
        "adapter": "summarizer",
        "prompt": f"{SUMMARIZER_INSTRUCTION}\n\n{context}",
        "reference": row["highlights"],
        "meta": {},
    }


BUILDERS = {
    "sql": build_sql_task,
    "code": build_code_task,
    "legal": build_legal_task,
    "medical": build_medical_task,
    "grammar": build_grammar_task,
    "summarizer": build_summarizer_task,
}


def generate_all_tasks(seed_rows: dict[str, list[dict]]) -> list[dict]:
    """Build quality tasks for every domain present in seed_rows.

    Args:
        seed_rows: domain tag -> list of raw dataset rows (dicts with the
            fields that domain's builder in BUILDERS expects).

    Returns:
        Flat list of task dicts, in domain-then-index order.
    """
    tasks = []
    for domain in sorted(seed_rows):
        builder = BUILDERS[domain]
        for i, row in enumerate(seed_rows[domain], start=1):
            tasks.append(builder(row, i))
    return tasks


def fetch_rows(domain: str) -> list[dict]:
    """Fetch N_PER_DOMAIN usable rows for a domain from its HF dataset.

    Args:
        domain: One of the keys in QUALITY_SOURCES.

    Returns:
        List of raw row dicts (up to N_PER_DOMAIN), or fewer if the dataset
        yields fewer usable rows. Empty list if the dataset fails to load.
    """
    import datasets

    repo_id, config, split = QUALITY_SOURCES[domain]
    required_fields = {
        "sql": ["sql_context", "sql_prompt", "sql"],
        "code": ["prompt", "canonical_solution", "test", "entry_point"],
        "legal": ["context", "question", "answers"],
        "medical": ["input", "output"],
        "grammar": ["sentence", "corrections"],
        "summarizer": ["article", "highlights"],
    }[domain]

    try:
        if config:
            ds = datasets.load_dataset(repo_id, config, split=split, streaming=True)
        else:
            ds = datasets.load_dataset(repo_id, split=split, streaming=True)
    except Exception as e:  # noqa: BLE001 - any loader failure skips the domain
        print(f"warning: skipping domain '{domain}' ({repo_id}): {e}")
        return []

    rows = []
    for row in ds:
        if all(row.get(f) not in (None, "", []) for f in required_fields):
            rows.append(row)
        if len(rows) >= N_PER_DOMAIN:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ground-truth output-quality tasks.")
    parser.add_argument("--out", default="benchmarks/quality_tasks.json")
    args = parser.parse_args()

    seed_rows: dict[str, list[dict]] = {}
    for domain in QUALITY_SOURCES:
        rows = fetch_rows(domain)
        if len(rows) < N_PER_DOMAIN:
            print(f"error: domain '{domain}' yielded only {len(rows)} usable rows (need {N_PER_DOMAIN})")
            continue
        seed_rows[domain] = rows

    if not seed_rows:
        raise SystemExit("error: no domains could be fetched, nothing to generate")

    tasks = generate_all_tasks(seed_rows)
    out_path = Path(args.out)
    lines = ",\n".join("  " + json.dumps(task) for task in tasks)
    out_path.write_text(f"[\n{lines}\n]\n")
    print(f"Wrote {len(tasks)} tasks to {out_path}")


if __name__ == "__main__":
    main()
