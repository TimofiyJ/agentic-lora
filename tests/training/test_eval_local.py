import json

from training.eval_local import (
    load_val_subset,
    reference_content,
    score_predictions,
    user_content,
)


def test_load_val_subset_reads_first_n_lines(tmp_path):
    path = tmp_path / "val.jsonl"
    examples = [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(5)]
    path.write_text("\n".join(json.dumps(e) for e in examples))

    result = load_val_subset(path, n=3)

    assert len(result) == 3
    assert result[0]["messages"][0]["content"] == "q0"


def test_user_content_extracts_the_user_message():
    example = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    assert user_content(example) == "What is 2+2?"


def test_reference_content_extracts_the_assistant_message():
    example = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    assert reference_content(example) == "4"


def test_score_predictions_matches_up_examples_and_predictions_in_order():
    val_examples = [
        {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "SELECT 1;"}]},
        {"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "SELECT 2;"}]},
    ]
    predictions = ["SELECT 1;", "SELECT 3;"]

    report = score_predictions("sql", val_examples, predictions)

    assert report["scores"] == [1.0, 0.0]
    assert report["mean_score"] == 0.5
