"""Knowledge-injection benchmark: closed-book, BM25 RAG, PRAG, Doc-to-LoRA and Ours + Doc-to-LoRA.

All conditions share the fictional corpus in benchmarks/knowledge, the same
base model served by one vLLM instance, greedy decoding, repetition penalty
1.15 (the controller's default for orchestrated steps) and PRAG's answer
prompt. Writes {url, started_at, method, tasks, summary} to a timestamped JSON.

Usage (on the GPU host, PYTHONPATH=src):
    python -m benchmarks.run_knowledge --method rag
    python -m benchmarks.run_knowledge --method prag --prag-root /root/adapters/prag --prag-seed 0
    python -m benchmarks.run_knowledge --method d2l --d2l-root /root/adapters/d2l
    python -m benchmarks.run_knowledge --method ours_d2l --orchestrator-url http://localhost:8002
"""

import argparse
import json
import shutil
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

from doc_knowledge.lora_merge import merge_adapters
from doc_knowledge.metrics import bootstrap_ci, score_answer
from doc_knowledge.prag import adapter_dir
from doc_knowledge.prompts import SHORT_ANSWER_HINT, build_user_prompt
from doc_knowledge.retriever import BM25Retriever, load_jsonl

from .metrics import count_adapter_switches, score_plan

UTC = timezone.utc
HERE = Path(__file__).parent
METHODS = ("closed", "rag", "prag", "d2l", "ours_d2l")
D2L_BIAS_RANK = 8
MAX_TOKENS = 128
REPETITION_PENALTY = 1.15


class Runner:
    """Executes one QA item under a given method.

    Args:
        method: One of METHODS.
        vllm_url: vLLM base URL.
        model: Served base model name.
        passages: Passage records for retrieval.
        doc_ids: Corpus document ids (adapter names under ours_d2l).
        top_k: Retrieved passages for rag/prag.
        prag_root: Root of trained PRAG adapters.
        prag_seed: PRAG training seed to evaluate.
        d2l_root: Root of exported per-document D2L adapters.
        merge_dir: Scratch directory for merged PRAG adapters.
        orchestrator_url: Orchestrator base URL for ours_d2l.
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        method: str,
        vllm_url: str,
        model: str,
        passages: list[dict],
        doc_ids: set[str],
        top_k: int = 3,
        prag_root: str | None = None,
        prag_seed: int = 0,
        d2l_root: str | None = None,
        merge_dir: str = "/tmp/prag_merged",
        orchestrator_url: str = "http://localhost:8002",
        timeout: float = 600.0,
    ):
        self.method = method
        self.vllm_url = vllm_url.rstrip("/")
        self.model = model
        self.retriever = BM25Retriever(passages)
        self.doc_ids = doc_ids
        self.top_k = top_k
        self.prag_root = prag_root
        self.prag_seed = prag_seed
        self.d2l_root = d2l_root
        self.merge_dir = Path(merge_dir)
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def _chat(self, model: str, prompt: str) -> dict:
        resp = self.client.post(
            f"{self.vllm_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": MAX_TOKENS,
                "repetition_penalty": REPETITION_PENALTY,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def run(self, item: dict) -> dict:
        """Answer one QA item.

        Args:
            item: QA record from qa.json.

        Returns:
            Dict with prediction, latency_s and method-specific diagnostics.
        """
        return getattr(self, f"_run_{self.method}")(item)

    def _run_closed(self, item: dict) -> dict:
        t0 = time.perf_counter()
        out = self._chat(self.model, build_user_prompt(item["question"], None, hint=True))
        return {
            "prediction": out["choices"][0]["message"]["content"],
            "latency_s": time.perf_counter() - t0,
            "prompt_tokens": out["usage"]["prompt_tokens"],
        }

    def _retrieve(self, item: dict) -> tuple[list[dict], float]:
        t0 = time.perf_counter()
        hits = self.retriever.search(item["question"], self.top_k)
        return hits, time.perf_counter() - t0

    def _retrieval_fields(self, item: dict, hits: list[dict]) -> dict:
        retrieved_docs = [h["doc_id"] for h in hits]
        gold = set(item["gold_doc_ids"])
        return {
            "retrieved": [h["id"] for h in hits],
            "doc_recall": len(gold & set(retrieved_docs)) / len(gold),
        }

    def _run_rag(self, item: dict) -> dict:
        t0 = time.perf_counter()
        hits, _ = self._retrieve(item)
        out = self._chat(self.model, build_user_prompt(item["question"], [h["text"] for h in hits], hint=True))
        return {
            "prediction": out["choices"][0]["message"]["content"],
            "latency_s": time.perf_counter() - t0,
            "prompt_tokens": out["usage"]["prompt_tokens"],
            **self._retrieval_fields(item, hits),
        }

    def _run_prag(self, item: dict) -> dict:
        hits, retrieve_s = self._retrieve(item)
        dirs = [adapter_dir(self.prag_root, self.prag_seed, h["id"]) for h in hits]
        return self._run_merged(item, hits, retrieve_s, dirs, "prag", 0)

    def _run_d2l(self, item: dict) -> dict:
        hits, retrieve_s = self._retrieve(item)
        doc_ids = list(dict.fromkeys(h["doc_id"] for h in hits))
        dirs = [Path(self.d2l_root) / d for d in doc_ids]
        out = self._run_merged(item, hits, retrieve_s, dirs, "d2l", D2L_BIAS_RANK)
        return {**out, "n_docs_merged": len(doc_ids)}

    def _run_merged(
        self, item: dict, hits: list[dict], retrieve_s: float, dirs: list[Path], prefix: str, shared_tail_rank: int
    ) -> dict:
        t0 = time.perf_counter() - retrieve_s
        name = f"{prefix}_{item['id']}_{uuid.uuid4().hex[:6]}"
        merged = self.merge_dir / name
        t_merge = time.perf_counter()
        merge_adapters(dirs, merged, shared_tail_rank=shared_tail_rank)
        merge_s = time.perf_counter() - t_merge
        t_load = time.perf_counter()
        resp = self.client.post(
            f"{self.vllm_url}/v1/load_lora_adapter", json={"lora_name": name, "lora_path": str(merged)}
        )
        resp.raise_for_status()
        load_s = time.perf_counter() - t_load
        try:
            out = self._chat(name, build_user_prompt(item["question"], None, hint=True))
            latency = time.perf_counter() - t0
        finally:
            self.client.post(f"{self.vllm_url}/v1/unload_lora_adapter", json={"lora_name": name})
            shutil.rmtree(merged, ignore_errors=True)
        return {
            "prediction": out["choices"][0]["message"]["content"],
            "latency_s": latency,
            "prompt_tokens": out["usage"]["prompt_tokens"],
            "retrieve_s": retrieve_s,
            "merge_s": merge_s,
            "load_s": load_s,
            **self._retrieval_fields(item, hits),
        }

    def _run_ours_d2l(self, item: dict) -> dict:
        t0 = time.perf_counter()
        resp = self.client.post(
            f"{self.orchestrator_url}/task",
            json={
                "query": f"{item['question']} {SHORT_ANSWER_HINT}",
                "prefetch": True,
                "unload_after": True,
                "replan_on_failure": True,
                "step_temperature": 0.0,
                "step_max_tokens": MAX_TOKENS,
            },
        )
        resp.raise_for_status()
        result = resp.json()
        latency = time.perf_counter() - t0
        steps = result.get("steps", [])
        doc_steps = [s["adapter"] for s in steps if s["adapter"] in self.doc_ids]
        plan = score_plan(doc_steps, item["oracle_adapter_sequence"])
        return {
            "prediction": result.get("final_output", ""),
            "latency_s": latency,
            "adapters": [s["adapter"] for s in steps],
            "n_steps": len(steps),
            "adapter_switches": count_adapter_switches(steps),
            "replanned": result.get("replanned", False),
            "n_loads": sum(e["event"] == "prefetch" for e in result.get("scheduling_events", [])),
            "plan_precision": plan["precision"],
            "plan_recall": plan["recall"],
            "plan_f1": plan["f1"],
            "plan_order_score": plan["order_score"],
        }


def summarize(records: list[dict]) -> dict:
    """Aggregate per-item records overall and by question type.

    Args:
        records: Per-item result records.

    Returns:
        Means with 95% bootstrap CIs for answer metrics, latency percentiles,
        and means of any method-specific numeric diagnostics.
    """
    def block(rows: list[dict]) -> dict:
        ok = [r for r in rows if r["success"]]
        out = {"n": len(rows), "n_errors": len(rows) - len(ok)}
        for metric in ("em", "f1", "contains"):
            values = [r[metric] for r in rows]
            out[metric] = sum(values) / len(values) if values else 0.0
            out[f"{metric}_ci95"] = bootstrap_ci(values)
        latencies = sorted(r["latency_s"] for r in ok)
        if latencies:
            out["latency_p50_s"] = statistics.median(latencies)
            out["latency_p95_s"] = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
        for key in ("doc_recall", "prompt_tokens", "merge_s", "load_s", "plan_f1", "plan_recall",
                    "plan_precision", "plan_order_score", "n_steps", "adapter_switches", "n_loads", "n_docs_merged"):
            values = [r[key] for r in ok if key in r]
            if values:
                out[key] = sum(values) / len(values)
        return out

    by_type = {t: block([r for r in records if r["type"] == t]) for t in sorted({r["type"] for r in records})}
    return {"overall": block(records), **by_type}


def run_benchmark(runner: Runner, qa: list[dict]) -> list[dict]:
    """Run every QA item, scoring answers and keeping failures as zero-score records.

    Args:
        runner: Configured Runner.
        qa: QA records.

    Returns:
        Per-item records.
    """
    records = []
    for i, item in enumerate(qa, start=1):
        base = {"id": item["id"], "type": item["type"], "question": item["question"], "answer": item["answer"]}
        try:
            out = runner.run(item)
            rec = {**base, "success": True, **out, **score_answer(out["prediction"], item["answer"], item["aliases"])}
        except Exception as e:
            rec = {**base, "success": False, "error": repr(e), "prediction": "", "latency_s": 0.0,
                   "em": 0.0, "f1": 0.0, "contains": 0.0}
        records.append(rec)
        print(f"[{i}/{len(qa)}] {item['id']} contains={rec['contains']:.0f} f1={rec['f1']:.2f} "
              f"{rec['prediction'][:80]!r}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--qa", default=str(HERE / "knowledge" / "qa.json"))
    parser.add_argument("--passages", default=str(HERE / "knowledge" / "passages.jsonl"))
    parser.add_argument("--corpus", default=str(HERE / "knowledge" / "corpus.jsonl"))
    parser.add_argument("--exclude", default=None, help="JSON list of QA ids to skip (contamination filter)")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--orchestrator-url", default="http://localhost:8002")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--prag-root", default="/root/adapters/prag")
    parser.add_argument("--prag-seed", type=int, default=0)
    parser.add_argument("--d2l-root", default="/root/adapters/d2l")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="", help="Extra label in the output filename, e.g. ep10")
    parser.add_argument("--out", default=str(HERE / "results"))
    args = parser.parse_args()

    qa = json.loads(Path(args.qa).read_text())
    if args.exclude:
        excluded = set(json.loads(Path(args.exclude).read_text()))
        qa = [q for q in qa if q["id"] not in excluded]
    qa = qa[: args.limit] if args.limit else qa
    runner = Runner(
        args.method,
        args.vllm_url,
        args.model,
        load_jsonl(args.passages),
        {d["id"] for d in load_jsonl(args.corpus)},
        top_k=args.top_k,
        prag_root=args.prag_root,
        prag_seed=args.prag_seed,
        d2l_root=args.d2l_root,
        orchestrator_url=args.orchestrator_url,
    )
    started = datetime.now(UTC)
    records = run_benchmark(runner, qa)
    suffix = f"_seed{args.prag_seed}" if args.method == "prag" else ""
    tag = f"_{args.tag}" if args.tag else ""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"knowledge_{args.method}{tag}{suffix}_{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    url = args.orchestrator_url if args.method == "ours_d2l" else args.vllm_url
    payload = {"url": url, "started_at": started.isoformat(), "method": args.method, "tag": args.tag,
               "tasks": records, "summary": summarize(records)}
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"]["overall"], indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
