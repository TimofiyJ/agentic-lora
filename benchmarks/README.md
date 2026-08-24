# Benchmarks

Three suites, testing different things:

| Suite | Tests | Entry point |
|---|---|---|
| Plan-selection | Did the orchestrator route to the right adapters, in the right order? | `run.py` |
| Output-quality | Given the correct adapter, is the generated output actually good? | `run_quality.py` |
| Multi-step cross-domain | Did both routing AND per-step output hold up across a genuinely multi-step task? | `run_multistep.py` |

Plan-selection goes through the full orchestrator (`/task` endpoint) and scores routing, not output content. Output-quality bypasses the planner entirely and hits the LoRA controller's `/infer/{adapter}` directly, so a bad score can only mean adapter quality, not a routing mistake. Multi-step cross-domain also goes through the full orchestrator, but scores both routing and per-step output on tasks that require chaining 2-4 adapters, closing the gap between the other two suites.

## Output-quality benchmark

### Data

`quality_tasks.json` — 120 tasks (20 per domain: `sql`, `code`, `legal`, `medical`, `grammar`, `summarizer`), each with a prompt and ground truth pulled from a public dataset:

| Domain | Source dataset | Metric |
|---|---|---|
| sql | `gretelai/synthetic_text_to_sql` | normalized exact-match |
| code | `openai/openai_humaneval` | pass@1 (sandboxed subprocess, 5s timeout) |
| legal | `alex-apostolo/filtered-cuad` | SQuAD-style token F1 (max over gold spans) |
| medical | `Malikeh1375/medical-question-answering-datasets` | ROUGE-L F-measure |
| grammar | `jhu-clsp/jfleg` | sentence-level GLEU (n=1-4, max over refs) |
| summarizer | `abisee/cnn_dailymail` | ROUGE-L F-measure |

Regenerate the task file (occasional, not part of a normal benchmark run):

```bash
python3 -m benchmarks.build_quality_tasks --out benchmarks/quality_tasks.json
```

Pulls fresh rows from Hugging Face Hub (streaming) for each domain; skips a domain with a warning if its dataset fails to load or yields too few usable rows.

### Running

Needs a live LoRA controller (default port 8001, see `docs/serving.md`):

```bash
python3 -m benchmarks.run_quality --controller-url http://localhost:8001 \
  --tasks benchmarks/quality_tasks.json --out benchmarks/results/
```

Writes a timestamped `benchmarks/results/quality_<timestamp>.json` with per-task records (`prediction`, `score`, `passed`) and a `summary` (`by_domain` avg score / pass rate, `overall_avg_score` — equal-weighted mean across domains, not a flat mean over tasks).

### Code

- `quality_metrics.py` — pure per-domain scorers + `score(domain, prediction, task)` dispatch + `summarize_quality(records)`.
- `build_quality_tasks.py` — standalone generator, not imported at benchmark-run time.
- `run_quality.py` — CLI harness: calls the controller, scores each response, writes results.

## Plan-selection benchmark

`tasks.json` (built via `build_tasks.py`) holds queries with `adapters_expected`. `run.py` posts each to the orchestrator's `/task` endpoint and scores the actual adapter sequence against the expected one (`metrics.py`: precision/recall/f1/order_correct).

```bash
python3 -m benchmarks.run --orchestrator-url http://localhost:8002 \
  --tasks benchmarks/tasks.json --out benchmarks/results/
```

## Multi-step cross-domain benchmark

Runs the full orchestrator plan+execute path (like the plan-selection benchmark) but scores both adapter-sequence correctness AND per-step output quality against fixed hand-authored ground truth (reusing `quality_metrics.py`'s per-domain scorers).

### Data

`multistep_tasks.json` — 20 hand-authored instances (5×2-step, 10×3-step, 5×4-step), each with an `oracle_adapter_sequence` and per-step `domain`/`reference`/`meta` shaped like `quality_tasks.json` entries. No two instances share the same adapter sequence; every instance requires genuine cross-domain chaining (`metadata.solvable_by_single_adapter: false`). Hand-authored, not pulled from a dataset — there's no build script.

### Running

Needs a live orchestrator (default port 8002):

```bash
python3 -m benchmarks.run_multistep --orchestrator-url http://localhost:8002 \
  --tasks benchmarks/multistep_tasks.json --out benchmarks/results/
```

Writes a timestamped `benchmarks/results/multistep_<timestamp>.json` with per-task records and a `summary`: `task_completion_rate` (every oracle step matched and passed), `avg_step_accuracy` (mean per-step score), `avg_coverage` (fraction of oracle steps whose adapter was actually invoked), `avg_error_propagation_rate` (among tasks with ≥1 failed step, how much downstream steps also failed), plus the plan-precision/recall/f1/order-correct and latency/switches/replan-rate fields shared with the plan-selection benchmark.

### Code

- `multistep_metrics.py` — step alignment (`align_steps`, matches actual orchestrator steps to oracle steps by adapter) + aggregate metrics (`task_completion`, `step_accuracy`, `error_propagation_rate`, `summarize_multistep`).
- `run_multistep.py` — CLI harness: calls the orchestrator, aligns and scores each step via `quality_metrics.score`, writes results.
