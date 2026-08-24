from benchmarks.build_quality_tasks import (
    BUILDERS,
    QUALITY_SOURCES,
    build_code_task,
    build_grammar_task,
    build_legal_task,
    build_medical_task,
    build_sql_task,
    build_summarizer_task,
    generate_all_tasks,
)


def test_quality_sources_cover_all_six_domains():
    assert set(QUALITY_SOURCES) == {
        "sql", "code", "legal", "medical", "grammar", "summarizer",
    }


def test_builders_cover_all_six_domains():
    assert set(BUILDERS) == {
        "sql", "code", "legal", "medical", "grammar", "summarizer",
    }


def test_build_sql_task():
    row = {"sql_context": "CREATE TABLE t (id INT);", "sql_prompt": "get all ids", "sql": "SELECT id FROM t;"}
    task = build_sql_task(row, 1)
    assert task == {
        "id": "quality_sql_01",
        "domain": "sql",
        "adapter": "sql",
        "prompt": "CREATE TABLE t (id INT);\n\nget all ids",
        "reference": "SELECT id FROM t;",
        "meta": {},
    }


def test_build_code_task():
    row = {
        "prompt": "def add(a, b):\n",
        "canonical_solution": "    return a + b\n",
        "test": "\n\ndef check(candidate):\n    assert candidate(1, 2) == 3\n",
        "entry_point": "add",
    }
    task = build_code_task(row, 2)
    assert task == {
        "id": "quality_code_02",
        "domain": "code",
        "adapter": "code",
        "prompt": "def add(a, b):\n",
        "reference": "    return a + b\n",
        "meta": {
            "test": "\n\ndef check(candidate):\n    assert candidate(1, 2) == 3\n",
            "entry_point": "add",
        },
    }


def test_build_legal_task():
    row = {
        "context": "This is a supply contract between the parties.",
        "question": "What is the document name?",
        "answers": {"text": ["supply contract"], "answer_start": [10]},
    }
    task = build_legal_task(row, 1)
    assert task == {
        "id": "quality_legal_01",
        "domain": "legal",
        "adapter": "legal",
        "prompt": "This is a supply contract between the parties.\n\nQuestion: What is the document name?",
        "reference": "supply contract",
        "meta": {"answers": ["supply contract"]},
    }


def test_build_legal_task_no_answers():
    row = {"context": "ctx", "question": "q?", "answers": {"text": [], "answer_start": []}}
    task = build_legal_task(row, 3)
    assert task["reference"] == ""
    assert task["meta"] == {"answers": []}


def test_build_medical_task():
    row = {"input": "what causes fever?", "output": "infection or inflammation"}
    task = build_medical_task(row, 1)
    assert task == {
        "id": "quality_medical_01",
        "domain": "medical",
        "adapter": "medical",
        "prompt": "what causes fever?",
        "reference": "infection or inflammation",
        "meta": {},
    }


def test_build_grammar_task():
    row = {"sentence": "he go to school", "corrections": ["he goes to school", "he went to school"]}
    task = build_grammar_task(row, 1)
    assert task == {
        "id": "quality_grammar_01",
        "domain": "grammar",
        "adapter": "grammar",
        "prompt": "he go to school",
        "reference": "he goes to school",
        "meta": {"corrections": ["he goes to school", "he went to school"]},
    }


def test_build_summarizer_task():
    row = {"article": "a long article body", "highlights": "short summary"}
    task = build_summarizer_task(row, 1)
    assert task == {
        "id": "quality_summarizer_01",
        "domain": "summarizer",
        "adapter": "summarizer",
        "prompt": "a long article body",
        "reference": "short summary",
        "meta": {},
    }


def test_generate_all_tasks_uses_correct_builder_and_count():
    seed_rows = {
        "sql": [
            {"sql_context": "ctx", "sql_prompt": f"q{i}", "sql": f"SELECT {i}"} for i in range(20)
        ],
    }
    tasks = generate_all_tasks(seed_rows)
    assert len(tasks) == 20
    assert all(t["domain"] == "sql" for t in tasks)
    assert tasks[0]["id"] == "quality_sql_01"
    assert tasks[19]["id"] == "quality_sql_20"
