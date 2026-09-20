"""Local (no-vLLM) quality evaluation for an exported PEFT adapter.

Uses benchmarks.quality_metrics.score -- the exact same scorer the real
quality benchmark uses -- so a local checkpoint's score is directly
comparable to a benchmarks/results/quality_*.json entry for the same domain.
"""

import json
from pathlib import Path

from benchmarks.quality_metrics import score


def load_val_subset(path: str | Path, n: int) -> list[dict]:
    """Load the first n examples from a val.jsonl file."""
    lines = Path(path).read_text().splitlines()[:n]
    return [json.loads(line) for line in lines if line]


def user_content(example: dict) -> str:
    """Return the user message's content from a {"messages": [...]} example."""
    return next(m["content"] for m in example["messages"] if m["role"] == "user")


def reference_content(example: dict) -> str:
    """Return the assistant message's content from a {"messages": [...]} example."""
    return next(m["content"] for m in example["messages"] if m["role"] == "assistant")


def score_predictions(domain: str, val_examples: list[dict], predictions: list[str]) -> dict:
    """Score each prediction against its matching val example's reference.

    Args:
        domain: Passed straight to benchmarks.quality_metrics.score (e.g. "sql").
        val_examples: Same length and order as predictions.
        predictions: Model outputs, one per val_examples entry.

    Returns:
        {"scores": [per-example float, ...], "mean_score": float}
    """
    scores = [
        score(domain, prediction, {"reference": reference_content(example)})["score"]
        for example, prediction in zip(val_examples, predictions)
    ]
    return {"scores": scores, "mean_score": sum(scores) / len(scores)}
