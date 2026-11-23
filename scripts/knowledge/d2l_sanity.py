"""Doc-to-LoRA export parity and single-document sanity check (vLLM must be stopped).

1. Parity: next-token logits of the D2L model after internalize(doc) vs. the
   base model with the exported PEFT adapter loaded, on each document's questions.
2. Sanity: greedy answers from the internalized D2L model to each document's own
   single-hop questions, scored like the benchmark.

Usage:
    PYTHONPATH=src python scripts/knowledge/d2l_sanity.py --checkpoint ... --adapter-root /root/adapters/d2l
"""

import argparse
import json
from pathlib import Path

import torch

from doc_knowledge.d2l_export import D2LExporter
from doc_knowledge.metrics import score_answer
from doc_knowledge.prompts import build_user_prompt
from doc_knowledge.retriever import load_jsonl

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter-root", required=True)
    parser.add_argument("--parity-docs", type=int, default=5)
    parser.add_argument("--out", default="/root/logs/d2l_sanity.json")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    corpus = load_jsonl(ROOT / "benchmarks" / "knowledge" / "corpus.jsonl")
    qa = json.loads((ROOT / "benchmarks" / "knowledge" / "qa.json").read_text())
    exporter = D2LExporter(args.checkpoint)
    d2l = exporter.model
    tokenizer = AutoTokenizer.from_pretrained(exporter.base_model_name)
    reference = AutoModelForCausalLM.from_pretrained(
        exporter.base_model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    def chat_ids(question: str) -> torch.Tensor:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": build_user_prompt(question, None, hint=True)}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(d2l.device)

    parity, answers = [], []
    for i, doc in enumerate(corpus):
        questions = [q for q in qa if q["type"] == "single" and q["gold_doc_ids"] == [doc["id"]]]
        d2l.reset()
        d2l.internalize(doc["text"])
        for q in questions:
            ids = chat_ids(q["question"])
            out = d2l.generate(input_ids=ids, max_new_tokens=128, do_sample=False)
            pred = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            answers.append({"id": q["id"], "prediction": pred, **score_answer(pred, q["answer"], q["aliases"])})
        if i < args.parity_docs:
            peft_model = PeftModel.from_pretrained(reference, str(Path(args.adapter_root) / doc["id"]))
            with torch.no_grad():
                for q in questions[:2]:
                    ids = chat_ids(q["question"])
                    gen_kwargs = dict(max_new_tokens=1, do_sample=False, output_logits=True, return_dict_in_generate=True)
                    d2l_logits = d2l.generate(input_ids=ids, **gen_kwargs).logits[0][0].float()
                    ref_logits = peft_model.generate(input_ids=ids, **gen_kwargs).logits[0][0].float()
                    parity.append({
                        "id": q["id"],
                        "max_abs_diff": (d2l_logits - ref_logits).abs().max().item(),
                        "cosine": torch.nn.functional.cosine_similarity(d2l_logits, ref_logits, dim=0).item(),
                        "same_argmax": bool(d2l_logits.argmax() == ref_logits.argmax()),
                    })
            reference = peft_model.unload()
        print(doc["id"], sum(a["contains"] for a in answers[-len(questions):]), "/", len(questions), flush=True)

    summary = {
        "single_doc_contains": sum(a["contains"] for a in answers) / len(answers),
        "single_doc_f1": sum(a["f1"] for a in answers) / len(answers),
        "parity_same_argmax": sum(p["same_argmax"] for p in parity) / max(1, len(parity)),
        "parity_min_cosine": min((p["cosine"] for p in parity), default=None),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "parity": parity, "answers": answers}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
