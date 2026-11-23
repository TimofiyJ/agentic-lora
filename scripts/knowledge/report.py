"""Aggregate knowledge-benchmark results into Markdown and LaTeX tables.

Usage:
    PYTHONPATH=src python scripts/knowledge/report.py --results benchmarks/results --adapters /root/adapters
"""

import argparse
import json
import statistics
from pathlib import Path

from doc_knowledge.metrics import bootstrap_ci, paired_bootstrap_pvalue

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"closed": "Closed-book", "rag": "Standard RAG (BM25 top-3)", "prag": "PRAG (BM25 top-3, merged LoRA)",
          "d2l": "D2L (BM25 top-3 docs, merged LoRA)", "ours_d2l": "Ours + D2L LoRAs", "prag_ep10": "PRAG, 10 epochs (sensitivity, seed 0)"}


def latest(results: Path, pattern: str) -> dict | None:
    """Load the newest result file matching a glob pattern, or None."""
    files = sorted(results.glob(pattern))
    return json.loads(files[-1].read_text()) if files else None


def by_id(run: dict, keep: set[str]) -> dict[str, dict]:
    """Index a run's records by QA id, restricted to kept ids."""
    return {t["id"]: t for t in run["tasks"] if t["id"] in keep}


def row(records: list[dict]) -> dict:
    """Mean metrics, 95% CI on Contains, and latency for a list of records."""
    out = {}
    for split in ("all", "single", "multi"):
        rows = [r for r in records if split == "all" or r["type"] == split]
        out[split] = {m: sum(r[m] for r in rows) / len(rows) for m in ("contains", "em", "f1")} if rows else {}
        if split == "all":
            out[split]["contains_ci"] = bootstrap_ci([r["contains"] for r in rows])
    ok = [r for r in records if r["success"]]
    out["latency_p50"] = statistics.median(r["latency_s"] for r in ok) if ok else None
    out["errors"] = len(records) - len(ok)
    for key in ("prompt_tokens", "doc_recall", "plan_f1", "plan_recall", "n_steps", "n_loads", "adapter_switches"):
        vals = [r[key] for r in ok if key in r]
        if vals:
            out[key] = sum(vals) / len(vals)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(ROOT / "benchmarks" / "results"))
    parser.add_argument("--adapters", default="/root/adapters")
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "knowledge_report.md"))
    args = parser.parse_args()
    results = Path(args.results)

    qa = json.loads((ROOT / "benchmarks" / "knowledge" / "qa.json").read_text())
    excluded = set(json.loads((ROOT / "benchmarks" / "knowledge" / "contaminated_ids.json").read_text()))
    keep = {q["id"] for q in qa} - excluded
    ids = sorted(keep)

    runs = {m: latest(results, f"knowledge_{m}_2*.json") for m in ("closed", "rag", "d2l", "ours_d2l")}
    prag_runs = [r for s in (0, 1, 2) if (r := latest(results, f"knowledge_prag_seed{s}_*.json"))]
    indexed = {m: by_id(r, keep) for m, r in runs.items() if r}
    rows = {m: row([v[i] for i in ids]) for m, v in indexed.items()}

    prag_seed_rows = [row([by_id(r, keep)[i] for i in ids]) for r in prag_runs]
    if prag_runs:
        prag_avg = [{**by_id(prag_runs[0], keep)[i],
                     **{m: statistics.mean(by_id(r, keep)[i][m] for r in prag_runs) for m in ("contains", "em", "f1")}}
                    for i in ids]
        indexed["prag"] = {r["id"]: r for r in prag_avg}
        rows["prag"] = row(prag_avg)
        rows["prag"]["seed_std"] = statistics.pstdev(s["all"]["contains"] for s in prag_seed_rows)
        rows["prag"]["latency_p50"] = statistics.mean(s["latency_p50"] for s in prag_seed_rows)

    ep10 = latest(results, "knowledge_prag_ep10_seed0_*.json")
    if ep10:
        indexed["prag_ep10"] = by_id(ep10, keep)
        rows["prag_ep10"] = row([indexed["prag_ep10"][i] for i in ids])

    pvals = {}
    if "ours_d2l" in indexed:
        ours = [indexed["ours_d2l"][i]["contains"] for i in ids]
        for m in ("closed", "rag", "prag", "d2l", "prag_ep10"):
            if m in indexed:
                pvals[m] = paired_bootstrap_pvalue(ours, [indexed[m][i]["contains"] for i in ids])

    adapters = Path(args.adapters)
    offline = {}
    if (adapters / "d2l" / "export_log.json").exists():
        log = json.loads((adapters / "d2l" / "export_log.json").read_text())
        offline["d2l"] = {"per_doc_s": statistics.mean(v["gen_seconds"] for v in log.values()),
                          "mb_per_doc": statistics.mean(v["size_mb"] for v in log.values()),
                          "rank": statistics.mean(v["rank"] for v in log.values()), "units": len(log)}
    if (adapters / "prag" / "train_log.json").exists():
        train = json.loads((adapters / "prag" / "train_log.json").read_text())
        aug = [json.loads(l) for l in open(adapters / "prag_augment.jsonl")]
        per_passage_train = statistics.mean(t for seed in train.values() for t in seed.values())
        per_passage_aug = statistics.mean(a["seconds"] for a in aug)
        sizes = [p.stat().st_size / 2**20 for p in (adapters / "prag" / "seed0").glob("*/adapter_model.safetensors")]
        offline["prag"] = {"per_passage_s": per_passage_train + per_passage_aug, "aug_s": per_passage_aug,
                           "train_s": per_passage_train, "mb_per_passage": statistics.mean(sizes) if sizes else None,
                           "units": len(aug), "per_doc_s": (per_passage_train + per_passage_aug) * len(aug) / 30}

    n_single = sum(1 for i in ids if i.startswith("s_"))
    lines = [f"# Knowledge-injection benchmark (Qwen3-4B-Instruct-2507, {len(ids)} questions: "
             f"{n_single} single-hop, {len(ids) - n_single} multi-hop; {len(excluded)} excluded by closed-book filter)",
             "", "| Method | Contains (95% CI) | EM | F1 | Single Contains | Multi Contains | p50 latency (s) | "
             "Prompt tok | Doc recall | Plan F1 | p vs Ours |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    tex = []
    for m in ("closed", "rag", "prag", "d2l", "ours_d2l", "prag_ep10"):
        if m not in rows:
            continue
        r = rows[m]
        lo, hi = r["all"]["contains_ci"]
        std = f" ±{r['seed_std']:.3f} (seed sd)" if "seed_std" in r else ""
        fmt = lambda k, p=2: f"{r[k]:.{p}f}" if k in r and r[k] is not None else "–"
        p = f"{pvals[m]:.4f}" if m in pvals else "–"
        lines.append(
            f"| {LABELS[m]} | {r['all']['contains']:.3f} [{lo:.3f}, {hi:.3f}]{std} | {r['all']['em']:.3f} | "
            f"{r['all']['f1']:.3f} | {r['single'].get('contains', 0):.3f} | {r['multi'].get('contains', 0):.3f} | "
            f"{fmt('latency_p50')} | {fmt('prompt_tokens', 0)} | {fmt('doc_recall')} | {fmt('plan_f1')} | {p} |")
        tex.append(f"{LABELS[m]} & {r['all']['contains']:.3f} & {r['all']['em']:.3f} & {r['all']['f1']:.3f} & "
                   f"{r['single'].get('contains', 0):.3f} & {r['multi'].get('contains', 0):.3f} & {fmt('latency_p50')} \\\\")
    if "ours_d2l" in rows:
        o = rows["ours_d2l"]
        lines += ["", f"Ours + D2L planner: plan F1 {o.get('plan_f1', 0):.3f}, plan recall {o.get('plan_recall', 0):.3f}, "
                      f"{o.get('n_steps', 0):.2f} steps/task, {o.get('n_loads', 0):.2f} prefetch loads/task, "
                      f"{o.get('adapter_switches', 0):.2f} switches/task, {o['errors']} errors"]
    lines += ["", "## Offline parameterization", "", "```", json.dumps(offline, indent=2), "```",
              "", "## LaTeX rows", "", "```latex", *tex, "```"]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
