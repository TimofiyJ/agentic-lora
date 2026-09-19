"""Generate SQL training data matching benchmarks/quality_tasks.json's sql format.

Source: gretelai/synthetic_text_to_sql, train split — see training/data/sources.py.
Run: python -m training.data.prepare_sql
"""

import argparse

from training.data.common import run_prepare

DOMAIN = "sql"


def format_row(row: dict) -> dict | None:
    """Build one {"messages": [...]} example from a synthetic_text_to_sql row."""
    user = f"{row['sql_context']}\n\n{row['sql_prompt']}"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["sql"]},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Prepare {DOMAIN} training data.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    run_prepare(DOMAIN, format_row, args.out_dir)


if __name__ == "__main__":
    main()
