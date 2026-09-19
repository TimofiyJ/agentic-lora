"""Single source of truth for where each P0 domain's training data comes from.

Mirrors QUALITY_SOURCES / DOMAIN_SOURCES in benchmarks/build_quality_tasks.py
and benchmarks/build_tasks.py. If those change which dataset/split/config a
domain's benchmark tasks are drawn from, update the matching entry here too.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainSource:
    repo_id: str
    config: str | None
    split: str
    revision: str
    exclude_head_n: int
    required_fields: tuple[str, ...]


SOURCES: dict[str, DomainSource] = {
    "sql": DomainSource(
        repo_id="gretelai/synthetic_text_to_sql",
        config=None,
        split="train",
        revision="740ab236e64503fba51be1101df7a1be83bf455d",
        exclude_head_n=200,
        required_fields=("sql_context", "sql_prompt", "sql"),
    ),
    "legal": DomainSource(
        repo_id="alex-apostolo/filtered-cuad",
        config=None,
        split="train",
        revision="f5e8e0a268c18fa828f2ba41ea459bfeb8ceb12e",
        exclude_head_n=0,
        required_fields=("context", "question"),
    ),
    "medical": DomainSource(
        repo_id="Malikeh1375/medical-question-answering-datasets",
        config="all-processed",
        split="train",
        revision="29833779cb5921f474d9f469aa85c115277bf489",
        exclude_head_n=200,
        required_fields=("input", "output"),
    ),
}
