from benchmarks.quality_metrics import (
    gleu_score,
    normalize_sql,
    rouge_l,
    score,
    score_code,
    score_grammar,
    score_legal,
    score_medical,
    score_sql,
    score_summarizer,
    summarize_quality,
    token_f1,
)


def test_normalize_sql_lowercases_and_collapses_whitespace():
    assert normalize_sql("SELECT  *   FROM users;") == "select * from users"


def test_normalize_sql_strips_trailing_semicolon_and_case():
    assert normalize_sql("select id from t") == normalize_sql("SELECT id FROM t;")


def test_score_sql_exact_match():
    task = {"reference": "SELECT id FROM users;"}
    result = score_sql("select id from users", task)
    assert result == {"score": 1.0, "passed": True}


def test_score_sql_mismatch():
    task = {"reference": "SELECT id FROM users;"}
    result = score_sql("SELECT name FROM users;", task)
    assert result == {"score": 0.0, "passed": False}


def _add_one_task():
    return {
        "prompt": "def add_one(x):\n",
        "reference": "    return x + 1\n",
        "meta": {
            "test": "\n\ndef check(candidate):\n    assert candidate(1) == 2\n    assert candidate(-1) == 0\n",
            "entry_point": "add_one",
        },
    }


def test_score_code_passing_solution():
    task = _add_one_task()
    result = score_code("    return x + 1\n", task)
    assert result == {"score": 1.0, "passed": True}


def test_score_code_failing_solution():
    task = _add_one_task()
    result = score_code("    return x - 1\n", task)
    assert result == {"score": 0.0, "passed": False}


def test_score_code_syntax_error():
    task = _add_one_task()
    result = score_code("    return x +\n", task)
    assert result == {"score": 0.0, "passed": False}


def test_score_code_infinite_loop_times_out():
    task = _add_one_task()
    result = score_code("    while True:\n        pass\n", task)
    assert result == {"score": 0.0, "passed": False}


def test_token_f1_exact_match():
    assert token_f1("supply contract", "supply contract") == 1.0


def test_token_f1_partial_overlap():
    # prediction has 2 tokens, reference has 2 tokens, 1 shared -> P=0.5 R=0.5 F1=0.5
    assert token_f1("supply agreement", "supply contract") == 0.5


def test_token_f1_no_overlap():
    assert token_f1("unrelated text here", "supply contract") == 0.0


def test_token_f1_empty_prediction():
    assert token_f1("", "supply contract") == 0.0


def test_score_legal_takes_max_across_gold_spans():
    task = {"meta": {"answers": ["completely different text", "supply contract"]}}
    result = score_legal("supply contract", task)
    assert result == {"score": 1.0, "passed": True}


def test_score_legal_no_answers_scores_zero():
    task = {"meta": {"answers": []}}
    result = score_legal("supply contract", task)
    assert result == {"score": 0.0, "passed": False}


def test_gleu_score_exact_match():
    assert gleu_score("the cat sat on the mat", "the cat sat on the mat") == 1.0


def test_gleu_score_no_overlap():
    assert gleu_score("completely different words here", "the cat sat on the mat") == 0.0


def test_gleu_score_empty_prediction():
    assert gleu_score("", "the cat sat on the mat") == 0.0


def test_score_grammar_takes_max_across_references():
    task = {"meta": {"corrections": ["a totally unrelated sentence", "the cat sat on the mat"]}}
    result = score_grammar("the cat sat on the mat", task)
    assert result == {"score": 1.0, "passed": True}


def test_rouge_l_exact_match():
    assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == 1.0


def test_rouge_l_no_overlap():
    assert rouge_l("zebra giraffe elephant", "the cat sat on the mat") == 0.0


def test_score_medical_uses_reference():
    task = {"reference": "take ibuprofen and rest"}
    result = score_medical("take ibuprofen and rest", task)
    assert result == {"score": 1.0, "passed": True}


def test_score_summarizer_uses_reference():
    task = {"reference": "the market rose sharply today"}
    result = score_summarizer("a bird flew over the ocean", task)
    assert result["score"] < 0.5
    assert result["passed"] is False


def test_score_dispatch_routes_to_correct_scorer():
    sql_task = {"reference": "SELECT 1"}
    assert score("sql", "SELECT 1", sql_task) == {"score": 1.0, "passed": True}

    medical_task = {"reference": "rest and hydrate"}
    assert score("medical", "rest and hydrate", medical_task) == {"score": 1.0, "passed": True}


def test_summarize_quality_per_domain_and_overall():
    records = [
        {"domain": "sql", "success": True, "score": 1.0, "passed": True},
        {"domain": "sql", "success": True, "score": 0.0, "passed": False},
        {"domain": "code", "success": True, "score": 1.0, "passed": True},
    ]
    result = summarize_quality(records)
    assert result["n_tasks"] == 3
    assert result["n_success"] == 3
    assert result["by_domain"]["sql"] == {"n": 2, "avg_score": 0.5, "pass_rate": 0.5}
    assert result["by_domain"]["code"] == {"n": 1, "avg_score": 1.0, "pass_rate": 1.0}
    assert result["overall_avg_score"] == 0.75


def test_summarize_quality_excludes_failures():
    records = [
        {"domain": "sql", "success": True, "score": 1.0, "passed": True},
        {"domain": "sql", "success": False, "error": "timeout"},
    ]
    result = summarize_quality(records)
    assert result["n_tasks"] == 2
    assert result["n_success"] == 1
    assert result["by_domain"]["sql"] == {"n": 1, "avg_score": 1.0, "pass_rate": 1.0}
    assert result["overall_avg_score"] == 1.0


def test_summarize_quality_no_success_returns_empty():
    records = [{"domain": "sql", "success": False, "error": "timeout"}]
    result = summarize_quality(records)
    assert result == {
        "n_tasks": 1,
        "n_success": 0,
        "by_domain": {},
        "overall_avg_score": 0.0,
    }
