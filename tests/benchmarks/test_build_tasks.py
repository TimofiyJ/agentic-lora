from benchmarks.build_tasks import (
    ACTION_PHRASES,
    DOMAIN_SOURCES,
    build_query,
    build_task,
    generate_tasks,
    merge_tasks,
)


def test_action_phrases_cover_all_six_domains():
    assert set(ACTION_PHRASES) == {
        "sql", "code", "legal", "medical", "grammar", "summarizer",
    }


def test_build_query_two_clauses():
    query = build_query("sql", "find top customers", "summarizer", "quarterly revenue")
    assert query == (
        'Write a SQL query for: "find top customers". '
        'Then summarize this text: "quarterly revenue".'
    )


def test_build_task_shape():
    task = build_task("sql", "find top customers", "summarizer", "quarterly revenue", n=1)
    assert task == {
        "id": "hf_sql_summarizer_01",
        "query": (
            'Write a SQL query for: "find top customers". '
            'Then summarize this text: "quarterly revenue".'
        ),
        "adapters_expected": ["sql", "summarizer"],
    }


def test_build_task_id_zero_padded():
    task = build_task("code", "reverse a list", "grammar", "fix this sentence", n=2)
    assert task["id"] == "hf_code_grammar_02"


def test_generate_tasks_all_pairs_and_seeds():
    seed_rows = {domain: [f"{domain} seed 1", f"{domain} seed 2"] for domain in DOMAIN_SOURCES}
    tasks = generate_tasks(seed_rows)

    assert len(tasks) == 30
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids))

    pairs = {tuple(t["adapters_expected"]) for t in tasks}
    assert len(pairs) == 15

    for task in tasks:
        assert task["id"].startswith("hf_")
        assert len(task["adapters_expected"]) == 2
        assert task["adapters_expected"][0] < task["adapters_expected"][1]


def test_merge_tasks_keeps_hand_written_drops_old_hf_appends_new():
    existing = [
        {"id": "sql_summarize_01", "query": "hand-written", "adapters_expected": ["sql"]},
        {"id": "hf_sql_code_01", "query": "stale", "adapters_expected": ["sql", "code"]},
    ]
    generated = [
        {"id": "hf_sql_code_01", "query": "fresh", "adapters_expected": ["sql", "code"]},
    ]

    merged = merge_tasks(existing, generated)

    assert merged == [
        {"id": "sql_summarize_01", "query": "hand-written", "adapters_expected": ["sql"]},
        {"id": "hf_sql_code_01", "query": "fresh", "adapters_expected": ["sql", "code"]},
    ]
