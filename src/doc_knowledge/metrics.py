"""Answer-quality metrics and paired statistics for knowledge-injection runs."""

import random
import re
import string
from collections import Counter

_ANSWER_PREFIX = re.compile(r"^\s*(so\s+)?the answer is[:\s]*", re.I)
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "sixty": "60",
    "twenty-nine": "29",
}


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization plus number words, thousands separators and units.

    Args:
        text: Raw answer or prediction.

    Returns:
        Lowercased text without punctuation, articles or redundant whitespace.
    """
    text = text.lower()
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    text = text.replace("metres", "meters").replace("litres", "liters").replace("centimetre", "centimeter")
    text = text.replace("%", " percent")
    text = "".join(ch if ch not in set(string.punctuation) - {"-", "."} else " " for ch in text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    tokens = [_NUMBER_WORDS.get(t, t) for t in text.split()]
    return " ".join(t for t in tokens if t not in {"a", "an", "the"})


def strip_answer_prefix(prediction: str) -> str:
    """Drop a leading "The answer is" so EM/F1 score only the answer span.

    Args:
        prediction: Raw model output.

    Returns:
        Prediction with the prefix and trailing period removed.
    """
    return _ANSWER_PREFIX.sub("", prediction.strip()).strip().rstrip(".")


def exact_match(prediction: str, answers: list[str]) -> float:
    """1.0 if the normalized prediction equals any normalized gold answer."""
    pred = normalize_answer(strip_answer_prefix(prediction))
    return float(any(pred == normalize_answer(a) for a in answers))


def token_f1(prediction: str, answers: list[str]) -> float:
    """Maximum token-level F1 between the prediction and any gold answer."""
    pred_tokens = normalize_answer(strip_answer_prefix(prediction)).split()
    best = 0.0
    for answer in answers:
        gold_tokens = normalize_answer(answer).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if not overlap:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def contains(prediction: str, answers: list[str]) -> float:
    """1.0 if any normalized gold answer appears inside the normalized prediction."""
    pred = f" {normalize_answer(prediction)} "
    return float(any(f" {normalize_answer(a)} " in pred for a in answers))


def score_answer(prediction: str, answer: str, aliases: list[str]) -> dict:
    """Compute EM, F1 and Contains for one prediction.

    Args:
        prediction: Raw model output.
        answer: Canonical gold answer.
        aliases: Accepted alternative answers.

    Returns:
        Dict with "em", "f1" and "contains".
    """
    answers = [answer, *aliases]
    return {
        "em": exact_match(prediction, answers),
        "f1": token_f1(prediction, answers),
        "contains": contains(prediction, answers),
    }


def bootstrap_ci(values: list[float], n_resamples: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% bootstrap confidence interval of the mean.

    Args:
        values: Per-question scores.
        n_resamples: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        (lower, upper) bounds; (0.0, 0.0) for empty input.
    """
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    return means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples) - 1]


def paired_bootstrap_pvalue(a: list[float], b: list[float], n_resamples: int = 10_000, seed: int = 0) -> float:
    """Two-sided paired bootstrap p-value for mean(a) != mean(b).

    Resamples question indices, recenters the paired differences under the
    null of zero mean difference, and counts resamples at least as extreme.

    Args:
        a: Per-question scores of system A.
        b: Per-question scores of system B, aligned with a.
        n_resamples: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        p-value in [0, 1].

    Raises:
        ValueError: If a and b differ in length.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    if not a:
        return 1.0
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    observed = sum(diffs) / n
    centered = [d - observed for d in diffs]
    rng = random.Random(seed)
    extreme = sum(
        abs(sum(rng.choices(centered, k=n)) / n) >= abs(observed) - 1e-12
        for _ in range(n_resamples)
    )
    return extreme / n_resamples
