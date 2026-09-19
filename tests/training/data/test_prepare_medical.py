from training.data.prepare_medical import MEDICAL_INSTRUCTION, format_row


def test_format_row_builds_user_and_assistant_messages():
    row = {"input": "What causes a fever?", "output": "Infection is the most common cause."}
    example = format_row(row)
    assert example == {
        "messages": [
            {
                "role": "user",
                "content": f"{MEDICAL_INSTRUCTION}\n\nWhat causes a fever?",
            },
            {"role": "assistant", "content": "Infection is the most common cause."},
        ]
    }
