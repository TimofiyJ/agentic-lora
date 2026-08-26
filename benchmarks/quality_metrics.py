"""Per-domain output-quality scoring against ground truth."""

import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from rouge_score import rouge_scorer


def normalize_sql(sql: str) -> str:
    """Normalize a SQL string for exact-match comparison.

    Args:
        sql: Raw SQL text.

    Returns:
        Lowercased, whitespace-collapsed, trailing-semicolon-stripped SQL.
    """
    stripped = sql.strip().rstrip(";").strip()
    collapsed = re.sub(r"\s+", " ", stripped)
    return collapsed.lower()


def score_sql(prediction: str, task: dict) -> dict:
    """Score a predicted SQL query by normalized exact match against the gold query.

    Args:
        prediction: Adapter's generated SQL text.
        task: Task dict with a "reference" key holding the gold SQL.

    Returns:
        Dict with "score" (1.0 or 0.0) and "passed" (bool).
    """
    match = normalize_sql(prediction) == normalize_sql(task["reference"])
    score = 1.0 if match else 0.0
    return {"score": score, "passed": match}


def score_code(prediction: str, task: dict) -> dict:
    """Score generated code by running it against the task's test cases (pass@1).

    Writes prompt + prediction + test harness to a temp file and executes it
    in a subprocess with a timeout. Any failure (assertion, exception, syntax
    error, timeout) scores 0.0 — this is expected model-output behavior, not
    a harness error.

    Args:
        prediction: Adapter's generated function body/completion.
        task: Task dict with "prompt" (function signature), and
            task["meta"]["test"] (test harness defining check()) and
            task["meta"]["entry_point"] (function name to pass to check()).

    Returns:
        Dict with "score" (1.0 or 0.0) and "passed" (bool).
    """
    entry_point = task["meta"]["entry_point"]
    program = (
        task["prompt"]
        + prediction
        + "\n"
        + task["meta"]["test"]
        + f"\ncheck({entry_point})\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "candidate.py"
        path.write_text(program)
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                cwd=tmpdir,
                capture_output=True,
                timeout=5,
            )
            passed = result.returncode == 0
        except subprocess.TimeoutExpired:
            passed = False
    score = 1.0 if passed else 0.0
    return {"score": score, "passed": passed}


def token_f1(prediction: str, reference: str) -> float:
    """Compute SQuAD-style token-overlap F1 between two strings.

    Args:
        prediction: Predicted text.
        reference: Gold text.

    Returns:
        F1 in [0.0, 1.0] over whitespace-tokenized, lowercased token multisets.
        0.0 if either string tokenizes to nothing, or there is no overlap.
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


_PREAMBLE_RE = re.compile(
    r"^\s*("
    r"here(?:'s| is)[^:]*:|"
    r"(?:the )?(?:corrected(?: sentence| version)?|summary|answer|correction)[^:]*:|"
    r"sure[,!]?\s*"
    r")\s*",
    re.IGNORECASE,
)


def strip_preamble(text: str) -> str:
    """Strip a common chat-style preamble from a model prediction.

    Instruct-tuned models often prepend filler like "Here's the corrected
    sentence:" or "Summary:" even when told to output only the answer. This
    removes one leading preamble match plus any surrounding quote marks so
    lexical metrics (F1/GLEU/ROUGE-L) aren't penalized for formatting noise
    the prompt asked the model not to add.

    Args:
        text: Raw model prediction.

    Returns:
        Prediction with a leading preamble phrase removed, stripped and
        unquoted.
    """
    stripped = _PREAMBLE_RE.sub("", text, count=1).strip()
    if len(stripped) >= 2 and stripped[0] in "\"'" and stripped[-1] == stripped[0]:
        stripped = stripped[1:-1].strip()
    return stripped


def score_legal(prediction: str, task: dict) -> dict:
    """Score a legal answer by max token-F1 against any gold answer span.

    Args:
        prediction: Adapter's generated answer text.
        task: Task dict with task["meta"]["answers"] (list of gold spans,
            possibly empty when the dataset marks the question unanswerable).

    Returns:
        Dict with "score" (float in [0.0, 1.0]) and "passed" (score >= 0.5).
    """
    prediction = strip_preamble(prediction)
    answers = task["meta"]["answers"]
    score = max((token_f1(prediction, a) for a in answers), default=0.0)
    return {"score": score, "passed": score >= 0.5}


def _ngram_counts(tokens: list[str], max_n: int = 4) -> Counter:
    counts = Counter()
    for n in range(1, max_n + 1):
        counts.update(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return counts


def gleu_score(prediction: str, reference: str, max_n: int = 4) -> float:
    """Compute simplified sentence-level GLEU between prediction and reference.

    GLEU (Napoles et al. 2015, the standard JFLEG metric) is min(precision,
    recall) over combined 1..max_n n-gram counts.

    Args:
        prediction: Predicted (e.g. grammar-corrected) sentence.
        reference: Gold reference sentence.
        max_n: Maximum n-gram order to include (default 4).

    Returns:
        GLEU score in [0.0, 1.0]. 0.0 if either string tokenizes to nothing.
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = _ngram_counts(pred_tokens, max_n)
    ref_counts = _ngram_counts(ref_tokens, max_n)
    overlap = sum((pred_counts & ref_counts).values())
    total_pred = sum(pred_counts.values())
    total_ref = sum(ref_counts.values())
    if total_pred == 0 or total_ref == 0:
        return 0.0
    precision = overlap / total_pred
    recall = overlap / total_ref
    return min(precision, recall)


def score_grammar(prediction: str, task: dict) -> dict:
    """Score a grammar correction by max GLEU against any reference correction.

    Args:
        prediction: Adapter's corrected sentence.
        task: Task dict with task["meta"]["corrections"] (list of reference
            corrections for the source sentence).

    Returns:
        Dict with "score" (float in [0.0, 1.0]) and "passed" (score >= 0.5).
    """
    prediction = strip_preamble(prediction)
    corrections = task["meta"]["corrections"]
    score = max((gleu_score(prediction, c) for c in corrections), default=0.0)
    return {"score": score, "passed": score >= 0.5}


_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F-measure between prediction and reference.

    Args:
        prediction: Predicted text.
        reference: Gold reference text.

    Returns:
        ROUGE-L F-measure in [0.0, 1.0].
    """
    return _rouge_scorer.score(reference, prediction)["rougeL"].fmeasure


def score_medical(prediction: str, task: dict) -> dict:
    """Score a medical answer by ROUGE-L against the reference answer.

    Args:
        prediction: Adapter's generated answer.
        task: Task dict with a "reference" key holding the gold answer.

    Returns:
        Dict with "score" (float in [0.0, 1.0]) and "passed" (score >= 0.5).
    """
    prediction = strip_preamble(prediction)
    fscore = rouge_l(prediction, task["reference"])
    return {"score": fscore, "passed": fscore >= 0.5}


def score_summarizer(prediction: str, task: dict) -> dict:
    """Score a summary by ROUGE-L against the reference summary.

    Args:
        prediction: Adapter's generated summary.
        task: Task dict with a "reference" key holding the gold summary.

    Returns:
        Dict with "score" (float in [0.0, 1.0]) and "passed" (score >= 0.5).
    """
    prediction = strip_preamble(prediction)
    fscore = rouge_l(prediction, task["reference"])
    return {"score": fscore, "passed": fscore >= 0.5}


_SCORERS = {
    "sql": score_sql,
    "code": score_code,
    "legal": score_legal,
    "medical": score_medical,
    "grammar": score_grammar,
    "summarizer": score_summarizer,
}


def score(domain: str, prediction: str, task: dict) -> dict:
    """Score a prediction against ground truth using the domain's metric.

    Args:
        domain: One of "sql", "code", "legal", "medical", "grammar", "summarizer".
        prediction: Adapter's generated output text.
        task: Task dict (shape depends on domain; see individual scorers).

    Returns:
        Dict with "score" (float in [0.0, 1.0]) and "passed" (bool).

    Raises:
        KeyError: If domain is not one of the six supported domains.
    """
    return _SCORERS[domain](prediction, task)


def summarize_quality(records: list[dict]) -> dict:
    """Aggregate per-task quality records into per-domain and overall summary stats.

    Only records with success=True contribute to averages. Overall average is
    the mean of per-domain averages (equal domain weighting), not a flat mean
    over all records.

    Args:
        records: Per-task result records. Successful records must include
            "domain", "score", and "passed". Records may be marked
            skipped=True (e.g. adapter not registered) — these are excluded
            from n_success/by_domain and reported separately in n_skipped.

    Returns:
        Dict with n_tasks, n_success, n_skipped, by_domain (domain -> {n,
        avg_score, pass_rate}), and overall_avg_score. overall_avg_score and
        by_domain are empty/0.0 when there are no successful records.
    """
    n_tasks = len(records)
    n_skipped = sum(1 for r in records if r.get("skipped"))
    successful = [r for r in records if r["success"]]
    n_success = len(successful)

    by_domain: dict[str, dict] = {}
    domains = sorted({r["domain"] for r in successful})
    for domain in domains:
        domain_records = [r for r in successful if r["domain"] == domain]
        n = len(domain_records)
        avg_score = sum(r["score"] for r in domain_records) / n
        pass_rate = sum(1 for r in domain_records if r["passed"]) / n
        by_domain[domain] = {"n": n, "avg_score": avg_score, "pass_rate": pass_rate}

    overall_avg_score = (
        sum(d["avg_score"] for d in by_domain.values()) / len(by_domain)
        if by_domain else 0.0
    )

    return {
        "n_tasks": n_tasks,
        "n_success": n_success,
        "n_skipped": n_skipped,
        "by_domain": by_domain,
        "overall_avg_score": overall_avg_score,
    }
