"""Shared streaming/formatting/splitting pipeline used by every prepare_<domain>.py."""

import datetime
import json
import random
from pathlib import Path
from typing import Callable, Iterable, Iterator

import datasets

from training.data.sources import SOURCES, DomainSource

VAL_SIZE = 200
SPLIT_SEED = 1234


def stream_valid_rows(source: DomainSource) -> Iterator[dict]:
    """Stream rows from source's HF split, in dataset order, keeping only rows
    where every field in source.required_fields is non-empty."""
    kwargs = {"split": source.split, "streaming": True, "revision": source.revision}
    if source.config:
        ds = datasets.load_dataset(source.repo_id, source.config, **kwargs)
    else:
        ds = datasets.load_dataset(source.repo_id, **kwargs)

    for row in ds:
        if all(row.get(field) not in (None, "", []) for field in source.required_fields):
            yield row


def skip_head(rows: Iterable[dict], n: int) -> Iterator[dict]:
    """Skip the first n items of rows, yield the rest."""
    it = iter(rows)
    for _ in range(n):
        if next(it, None) is None:
            break
    yield from it


def format_rows(rows: Iterable[dict], format_row: Callable[[dict], dict | None]) -> list[dict]:
    """Apply format_row to each row, keeping only the ones it doesn't reject (None)."""
    return [example for row in rows if (example := format_row(row)) is not None]


def train_val_split(
    examples: list[dict], val_size: int = VAL_SIZE, seed: int = SPLIT_SEED
) -> tuple[list[dict], list[dict]]:
    """Deterministically shuffle examples and split off val_size for validation."""
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    return shuffled[val_size:], shuffled[:val_size]


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


def write_manifest(
    path: Path,
    domain: str,
    source: DomainSource,
    n_rows_before_filter: int,
    n_rows_after_filter: int,
    n_train: int,
    n_val: int,
) -> None:
    manifest = {
        "domain": domain,
        "repo_id": source.repo_id,
        "config": source.config,
        "split": source.split,
        "revision": source.revision,
        "exclude_head_n": source.exclude_head_n,
        "n_rows_before_filter": n_rows_before_filter,
        "n_rows_after_filter": n_rows_after_filter,
        "n_train": n_train,
        "n_val": n_val,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def run_prepare(
    domain: str,
    format_row: Callable[[dict], dict | None],
    out_dir: str | None = None,
    val_size: int = VAL_SIZE,
) -> None:
    """Full prepare pipeline for one domain: stream -> skip head -> format -> split -> write."""
    source = SOURCES[domain]
    rows = list(skip_head(stream_valid_rows(source), source.exclude_head_n))
    n_before = len(rows)
    examples = format_rows(rows, format_row)
    n_after = len(examples)
    train, val = train_val_split(examples, val_size=val_size)

    out_path = Path(out_dir or f"training/data/output/{domain}")
    write_jsonl(out_path / "train.jsonl", train)
    write_jsonl(out_path / "val.jsonl", val)
    write_manifest(
        out_path / f"manifest_{domain}.json", domain, source, n_before, n_after, len(train), len(val)
    )
    print(f"{domain}: {n_before} valid rows -> {n_after} examples -> {len(train)} train / {len(val)} val")
