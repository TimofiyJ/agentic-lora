from training.data.prepare_sql import format_row


def test_format_row_builds_user_and_assistant_messages():
    row = {
        "sql_context": "CREATE TABLE users (id INT, name TEXT)",
        "sql_prompt": "List all user names",
        "sql": "SELECT name FROM users",
    }
    example = format_row(row)
    assert example == {
        "messages": [
            {
                "role": "user",
                "content": "CREATE TABLE users (id INT, name TEXT)\n\nList all user names",
            },
            {"role": "assistant", "content": "SELECT name FROM users"},
        ]
    }
