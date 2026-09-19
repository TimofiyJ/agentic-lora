from training.data.prepare_legal import LEGAL_INSTRUCTION, format_row


def test_format_row_builds_quoted_answer_example():
    row = {
        "context": "This Agreement shall terminate on December 31, 2030.",
        "question": "What is the termination date?",
        "answers": {"text": ["December 31, 2030"], "answer_start": [30]},
    }
    example = format_row(row)
    assert example["messages"][0]["content"] == (
        "This Agreement shall terminate on December 31, 2030.\n\n"
        "Question: What is the termination date?\n\n"
        f"{LEGAL_INSTRUCTION}"
    )
    assert example["messages"][1]["content"] == "December 31, 2030"


def test_format_row_drops_unanswerable_rows():
    row = {
        "context": "Some contract text.",
        "question": "Is there a non-compete clause?",
        "answers": {"text": [], "answer_start": []},
    }
    assert format_row(row) is None


def test_format_row_truncates_context_to_2000_chars():
    row = {
        "context": "x" * 5000,
        "question": "q",
        "answers": {"text": ["a"], "answer_start": [0]},
    }
    example = format_row(row)
    user_content = example["messages"][0]["content"]
    context_part = user_content.split("\n\nQuestion:")[0]
    assert len(context_part) == 2000
