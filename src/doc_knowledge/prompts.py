"""Answer prompt shared by every condition (PRAG's non-CoT template, src/prompt_template.py)."""

USER_PROMPT = (
    "You should answer the question by referring to the knowledge provided below and integrating your own "
    "knowledge.\n{passages}\n\nQuestion: {question}"
)
ANSWER_TEMPLATE = "The answer is {answer}"
SHORT_ANSWER_HINT = "Answer with a short phrase, starting with \"The answer is\"."


def normalize_question(question: str) -> str:
    """Ensure the question ends with a single "?" as PRAG does."""
    question = question.strip()
    if question.endswith(" ?"):
        question = question[:-1].strip() + "?"
    return question if question.endswith("?") else question + "?"


def build_user_prompt(question: str, passages: list[str] | None = None, hint: bool = False) -> str:
    """Format the PRAG user prompt.

    Args:
        question: The question.
        passages: Passages to place in context, or None for parametric/closed-book use.
        hint: Append the short-answer instruction (used at evaluation time).

    Returns:
        Prompt text.
    """
    contexts = "".join(f"Passage {i + 1}: {p}\n" for i, p in enumerate(passages or []))
    prompt = USER_PROMPT.format(passages=contexts, question=normalize_question(question))
    return f"{prompt}\n{SHORT_ANSWER_HINT}" if hint else prompt
