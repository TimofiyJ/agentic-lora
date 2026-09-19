from training.data.contamination_check import (
    check_contamination,
    load_benchmark_texts,
    ngrams,
    normalize,
)


def test_normalize_lowercases_strips_punctuation_and_collapses_whitespace():
    assert normalize("SELECT  *  FROM Users;") == "select from users"


def test_ngrams_returns_whole_text_as_one_gram_when_shorter_than_n():
    assert ngrams("a b c", n=13) == {"a b c"}


def test_ngrams_produces_sliding_windows_of_size_n():
    text = " ".join(str(i) for i in range(15))
    grams = ngrams(text, n=13)
    assert len(grams) == 3  # 15 tokens - 13 + 1


def test_load_benchmark_texts_filters_by_domain_across_quality_and_multistep():
    quality = [
        {"id": "quality_sql_01", "domain": "sql", "prompt": "p1", "reference": "r1"},
        {"id": "quality_legal_01", "domain": "legal", "prompt": "p2", "reference": "r2"},
    ]
    multistep = [
        {"id": "hf_sql_legal_01", "query": "q1", "adapters_expected": ["sql", "legal"]},
        {"id": "hf_medical_code_01", "query": "q2", "adapters_expected": ["medical", "code"]},
    ]
    texts = load_benchmark_texts("sql", quality, multistep)
    ids = {bench_id for bench_id, _ in texts}
    assert ids == {"quality_sql_01", "hf_sql_legal_01"}


def test_check_contamination_flags_an_exact_duplicate():
    examples = [
        {
            "messages": [
                {"role": "user", "content": "What is 1+1?"},
                {"role": "assistant", "content": "2"},
            ]
        }
    ]
    bench_texts = [("quality_sql_01", "What is 1+1?\n2")]
    report = check_contamination(examples, bench_texts)
    assert report["status"] == "contaminated"
    assert report["exact_matches"] == [{"train_index": 0, "bench_id": "quality_sql_01"}]
    # An exact duplicate is trivially also a near-duplicate (jaccard == 1.0
    # >= threshold) — both checks are independent and both fire here.
    assert len(report["near_duplicates"]) == 1
    assert report["near_duplicates"][0]["bench_id"] == "quality_sql_01"


def test_check_contamination_reports_clean_when_nothing_overlaps():
    examples = [
        {
            "messages": [
                {"role": "user", "content": "Totally unrelated question about databases."},
                {"role": "assistant", "content": "An unrelated answer."},
            ]
        }
    ]
    bench_texts = [
        ("quality_sql_01", "A completely different benchmark prompt entirely on another topic.\nreference text")
    ]
    report = check_contamination(examples, bench_texts)
    assert report["status"] == "clean"
    assert report["exact_matches"] == []
    assert report["near_duplicates"] == []
