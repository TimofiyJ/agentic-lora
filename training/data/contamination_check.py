"""Check a domain's prepared train/val data for overlap with the benchmark
tasks it must never leak into: benchmarks/quality_tasks.json and
benchmarks/multistep_tasks.json.

Run: python -m training.data.contamination_check sql
Exits non-zero (and never overwrites the domain's data) if anything is found.
"""

import argparse
import json
import re
import string
import sys
from pathlib import Path

NGRAM_N = 13
JACCARD_THRESHOLD = 0.8

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def example_text(example: dict) -> str:
    """Concatenate a training example's message contents, in order."""
    return "\n".join(m["content"] for m in example["messages"])


def bench_task_text(task: dict) -> str:
    """Concatenate a benchmark task's prompt/query and reference text, if any."""
    prompt = task.get("prompt", task.get("query", ""))
    return f"{prompt}\n{task.get('reference', '')}"


def ngrams(normalized_text: str, n: int = NGRAM_N) -> set[str]:
    tokens = normalized_text.split()
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_benchmark_texts(
    domain: str, quality_tasks: list[dict], multistep_tasks: list[dict]
) -> list[tuple[str, str]]:
    """Return [(bench_id, raw_text), ...] for every benchmark task touching domain."""
    texts = []
    for task in quality_tasks:
        if task.get("domain") == domain:
            texts.append((task["id"], bench_task_text(task)))
    for task in multistep_tasks:
        if domain in task.get("adapters_expected", []):
            texts.append((task["id"], bench_task_text(task)))
    return texts


def check_contamination(examples: list[dict], bench_texts: list[tuple[str, str]]) -> dict:
    """Compare every example against every benchmark text; report exact and near-dup hits."""
    bench_normalized = [(bench_id, normalize(text)) for bench_id, text in bench_texts]
    bench_ngrams = [(bench_id, norm, ngrams(norm)) for bench_id, norm in bench_normalized]

    exact_matches = []
    near_duplicates = []

    for i, example in enumerate(examples):
        ex_norm = normalize(example_text(example))
        ex_ngrams = ngrams(ex_norm)

        for bench_id, bench_norm in bench_normalized:
            if ex_norm == bench_norm:
                exact_matches.append({"train_index": i, "bench_id": bench_id})

        for bench_id, _bench_norm, bench_ng in bench_ngrams:
            score = jaccard(ex_ngrams, bench_ng)
            if score >= JACCARD_THRESHOLD:
                near_duplicates.append(
                    {"train_index": i, "bench_id": bench_id, "jaccard": round(score, 4)}
                )

    status = "clean" if not exact_matches and not near_duplicates else "contaminated"
    return {
        "n_examples_checked": len(examples),
        "exact_matches": exact_matches,
        "near_duplicates": near_duplicates,
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check training data for benchmark contamination.")
    parser.add_argument("domain", choices=["sql", "legal", "medical"])
    parser.add_argument("--data-dir", default=None, help="Defaults to training/data/output/<domain>")
    parser.add_argument("--quality-tasks", default="benchmarks/quality_tasks.json")
    parser.add_argument("--multistep-tasks", default="benchmarks/multistep_tasks.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir or f"training/data/output/{args.domain}")
    examples = []
    for filename in ("train.jsonl", "val.jsonl"):
        path = data_dir / filename
        if path.exists():
            examples += [json.loads(line) for line in path.read_text().splitlines() if line]

    quality_tasks = json.loads(Path(args.quality_tasks).read_text())
    multistep_path = Path(args.multistep_tasks)
    multistep_tasks = json.loads(multistep_path.read_text()) if multistep_path.exists() else []

    bench_texts = load_benchmark_texts(args.domain, quality_tasks, multistep_tasks)
    report = check_contamination(examples, bench_texts)
    report["domain"] = args.domain

    out_path = data_dir / f"contamination_report_{args.domain}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.domain}: {report['status']} ({len(examples)} examples checked)")

    if report["status"] != "clean":
        sys.exit(1)


if __name__ == "__main__":
    main()
