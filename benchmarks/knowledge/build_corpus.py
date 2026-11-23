"""Build the fictional knowledge corpus and QA set from hand-written sources.

Reads benchmarks/knowledge/docs/*.md and qa_source.yaml, validates that every
answer is grounded in its gold documents, and writes corpus.jsonl,
passages.jsonl and qa.json next to this file.
"""

import json
import re
from pathlib import Path

import yaml

from doc_knowledge.metrics import normalize_answer

HERE = Path(__file__).parent


def parse_docs(md_text: str, domain: str) -> list[dict]:
    """Parse a domain markdown file into document records.

    Args:
        md_text: File contents; documents start with "=== <id>" followed by
            "title:" and "entities:" lines, then blank-line separated paragraphs.
        domain: Domain label (the file stem).

    Returns:
        Document dicts with id, domain, title, entities, description, text, paragraphs.
    """
    docs = []
    for block in re.split(r"^=== ", md_text, flags=re.M)[1:]:
        header, _, body = block.partition("\n\n")
        lines = header.strip().splitlines()
        doc_id = lines[0].strip()
        meta = dict(line.split(": ", 1) for line in lines[1:])
        paragraphs = [p.strip().replace("\n", " ") for p in body.strip().split("\n\n") if p.strip()]
        entities = [e.strip() for e in meta["entities"].split(",")]
        docs.append({
            "id": doc_id,
            "domain": domain,
            "title": meta["title"],
            "entities": entities,
            "description": f"Knowledge of the document '{meta['title']}' (covers: {', '.join(entities)})",
            "text": "\n\n".join(paragraphs),
            "paragraphs": paragraphs,
        })
    return docs


def build_qa(source: dict, docs_by_id: dict[str, dict]) -> list[dict]:
    """Expand qa_source.yaml into flat QA records and check grounding.

    Args:
        source: Parsed qa_source.yaml with "single" and "multi" sections.
        docs_by_id: Document records keyed by id.

    Returns:
        QA records with id, type, question, answer, aliases, gold_doc_ids,
        oracle_adapter_sequence.

    Raises:
        ValueError: If an answer (or alias) is not found in its gold documents.
    """
    records = []
    for doc_id, items in source["single"].items():
        for i, item in enumerate(items, start=1):
            records.append({"id": f"s_{doc_id}_{i}", "type": "single", "item": item, "docs": [doc_id]})
    for i, item in enumerate(source["multi"], start=1):
        records.append({"id": f"m_{i:03d}", "type": "multi", "item": item, "docs": item["docs"]})

    qa, errors = [], []
    for r in records:
        item, docs = r["item"], r["docs"]
        answers = [item["a"], *item.get("alias", [])]
        gold_text = normalize_answer(" ".join(docs_by_id[d]["text"] for d in docs))
        if not any(normalize_answer(a) in gold_text for a in answers):
            errors.append(f"{r['id']}: answer {item['a']!r} not found in {docs}")
        qa.append({
            "id": r["id"],
            "type": r["type"],
            "question": item["q"],
            "answer": item["a"],
            "aliases": item.get("alias", []),
            "gold_doc_ids": docs,
            "oracle_adapter_sequence": docs,
        })
    if errors:
        raise ValueError("ungrounded answers:\n" + "\n".join(errors))
    return qa


def main() -> None:
    docs = []
    for md in sorted((HERE / "docs").glob("*.md")):
        docs.extend(parse_docs(md.read_text(), md.stem))
    docs_by_id = {d["id"]: d for d in docs}
    if len(docs_by_id) != len(docs):
        raise ValueError("duplicate document ids")

    passages = [
        {"id": f"{d['id']}#p{i}", "doc_id": d["id"], "text": p}
        for d in docs
        for i, p in enumerate(d["paragraphs"])
    ]
    qa = build_qa(yaml.safe_load((HERE / "qa_source.yaml").read_text()), docs_by_id)

    with open(HERE / "corpus.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps({k: v for k, v in d.items() if k != "paragraphs"}) + "\n")
    with open(HERE / "passages.jsonl", "w") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")
    (HERE / "qa.json").write_text(json.dumps(qa, indent=1))

    n_single = sum(q["type"] == "single" for q in qa)
    print(f"{len(docs)} docs, {len(passages)} passages, {n_single} single-hop + {len(qa) - n_single} multi-hop QA")


if __name__ == "__main__":
    main()
