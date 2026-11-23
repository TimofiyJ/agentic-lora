"""BM25 passage retrieval shared by the RAG and PRAG baselines."""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from has have how in is it its of on or that the "
    "this to was were what when where which who whom whose why with".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens without stopwords.

    Args:
        text: Query or passage text.

    Returns:
        Token list.
    """
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSON-lines file.

    Args:
        path: File path.

    Returns:
        One dict per non-empty line.
    """
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class BM25Retriever:
    """Okapi BM25 over a fixed passage collection.

    Args:
        passages: Dicts with at least "id" and "text".
    """

    def __init__(self, passages: list[dict]):
        self._passages = passages
        self._bm25 = BM25Okapi([tokenize(p["text"]) for p in passages])

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Return the top-k passages for a query, best first.

        Args:
            query: Natural-language question.
            k: Number of passages to return.

        Returns:
            Passage dicts with an added "score" key.
        """
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{**self._passages[i], "score": float(scores[i])} for i in ranked]
