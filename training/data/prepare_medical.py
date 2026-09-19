"""Generate medical training data matching benchmarks/quality_tasks.json's medical format.

Source: Malikeh1375/medical-question-answering-datasets, config all-processed,
train split — see training/data/sources.py. Run: python -m training.data.prepare_medical
"""

import argparse

from training.data.common import run_prepare

DOMAIN = "medical"
MEDICAL_INSTRUCTION = (
    "Answer the following medical question directly and concisely. "
    "Output only the answer, with no explanation or preamble."
)


def format_row(row: dict) -> dict | None:
    """Build one {"messages": [...]} example from a medical-qa row."""
    user = f"{MEDICAL_INSTRUCTION}\n\n{row['input']}"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["output"]},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Prepare {DOMAIN} training data.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    run_prepare(DOMAIN, format_row, args.out_dir)


if __name__ == "__main__":
    main()
