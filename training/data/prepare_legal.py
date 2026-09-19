"""Generate legal training data matching benchmarks/quality_tasks.json's legal format.

Source: alex-apostolo/filtered-cuad, train split (the benchmark itself draws
from test — see training/data/sources.py). Run: python -m training.data.prepare_legal
"""

import argparse

from training.data.common import run_prepare

DOMAIN = "legal"
MAX_PROMPT_CHARS = 2000
LEGAL_INSTRUCTION = (
    "Answer using a direct quote from the contract text above. "
    "Output only the quoted answer, with no explanation or preamble."
)


def format_row(row: dict) -> dict | None:
    """Build one {"messages": [...]} example from a filtered-cuad row.

    Drops rows with no answer span (CUAD includes unanswerable questions;
    the benchmark's own build_legal_task never emits those).
    """
    answers = row["answers"]["text"]
    if not answers:
        return None
    context = row["context"][:MAX_PROMPT_CHARS]
    user = f"{context}\n\nQuestion: {row['question']}\n\n{LEGAL_INSTRUCTION}"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answers[0]},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Prepare {DOMAIN} training data.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    run_prepare(DOMAIN, format_row, args.out_dir)


if __name__ == "__main__":
    main()
