# Research Results, Problems & Theories

Running log of empirical findings, blockers, and hypotheses as the system is built and tested.

---

## B0 — Qwen3-8B Passthrough Baseline (2026-09-20)

E1 (`docs/superpowers/plans/2026-09-20-e1-qwen3-8b-baseline.md`), branch `tinker-custom-loras`. vLLM 0.29.0, `Qwen/Qwen3-8B`, tensor-parallel-size 2 on 2x RTX 4060 Ti (16 GB each) via vast.ai. All 6 domains passthrough (no LoRA), `chat_template_kwargs: {enable_thinking: false}` on every adapter.

**Thinking suppression: confirmed working.** Live-checked before running any benchmark (per the plan's mandatory gate) and again in the results below — zero `<think>` blocks, `reasoning: null` on every response. The `chat_template_kwargs` mechanism added to `AdapterMeta`/`VLLMClient`/`app.py` this session works on vLLM 0.29.0; none of the design spec's fallback options (`--default-chat-template-kwargs`, custom chat template, post-hoc stripping) were needed.

**Real bug found and fixed along the way:** the orchestrator's `planner.py` makes its own direct `/v1/chat/completions` call to vLLM for plan generation, *separate* from the controller's `chat_completions` — and it wasn't disabling thinking. Qwen3-8B's thinking is verbose enough (492 tokens to output `{"x": 1}` in a minimal test) that on real planning prompts it exceeded `max_tokens: 1024` mid-`<think>`, leaving `content` empty and every multistep task failing with `JSONDecodeError: Expecting value: line 1 column 1` → 500 (all 50/50 tasks failed on the first run). Fixed by adding the same `chat_template_kwargs: {"enable_thinking": false}` to `planner.py`'s `_call_llm` payload (commit `a90c133`) — confirmed by the same minimal test dropping from 492 to 7 completion tokens for the trivial JSON case.

### Quality benchmark (120 tasks, 20/domain)

| Domain | Metric | B0 | B-old (ours) | B-old (AM) |
|---|---|---|---|---|
| sql | Normalised EM | 0.000 | 0.000 | 0.150 |
| legal | Token-F1 | 0.209 | 0.092 | 0.025 |
| medical | ROUGE-L | 0.137 | 0.103 | 0.130 |
| code | pass@1 | 0.050 | 0.050 | — |
| grammar | GLEU | 0.522 | 0.519 | — |
| summarizer | ROUGE-L | 0.187 | 0.190 | — |

Overall avg score 0.184. Single-domain pass rate (avg of per-domain pass_rate): code 0.05, grammar 0.60, legal 0.15, medical 0.00, sql 0.00, summarizer 0.00 → **13.3%** overall.

Avg latency per domain: sql 14.8s, code 15.3s, summarizer 2.5s, medical 1.5s, grammar 1.0s, legal 0.9s (overall avg 6.0s).

**sql/code latency and score are both explained by one thing, not two:** those are the only two domains whose benchmark prompt has no "answer directly, no explanation" instruction (legal/medical/summarizer all have one baked into their prompt template). Sample sql prediction: `"...ORDER BY s.name; \n\nWait, but what if there are multiple entries for the same salesperson..."` — the model free-associates after a technically-plausible answer. This is not thinking-token leakage (confirmed no `<think>`, `reasoning: null`) — it's Qwen3-8B's default verbose-explainer style on an under-specified prompt, which directly costs both the exact-match-style score and the latency (more tokens generated). Not fixed here — this is exactly the gap the sql/legal/medical LoRA adapters (data already prepared, see the P0 data pipeline entry) are meant to close, and changing the benchmark prompts now would break comparability with `B-old`.

### Multistep benchmark (50 tasks — `benchmarks/tasks.json` has grown since the 20-task B-old reference run)

| Metric | B0 | B-old (Qwen2.5-7B) |
|---|---|---|
| n_success | 50/50 | — |
| plan_precision | 1.00 | — |
| plan_recall | 0.99 | — |
| plan_f1 | 0.993 | 0.879 |
| plan_order_correct_rate | 0.98 | 0.500 |
| avg_switches | 0.94 | 1.6 |
| avg_latency_s | 22.6 | 28.9 |

Raw Qwen3-8B (no orchestrator-side adapters at all) already clears the PRD's plan F1 target (≥0.85) and beats B-old on every planning metric measured, including latency. **Caveat, not glossed over:** `benchmarks/run.py`'s current summary does not compute `avg_coverage`, `avg_step_accuracy`, `task_completion_rate`, or `avg_error_propagation_rate` — the metrics R2's Qwen2.5-7B multistep table reported. Those either came from a since-changed script version or manual analysis of the raw per-step outputs; they are not available for B0 without either upgrading `benchmarks/multistep_metrics.py`'s summary or a manual pass over the 50 raw task records. Not fabricated here — flagged as an open gap for whoever needs those numbers next (likely E4).

**Headline:** the planning story on Qwen3-8B is already strong without any adapter — the bottleneck E3's adapters need to fix is single-domain output quality (13.3% pass rate), not plan structure.

---

## E3 — SQL LoRA, First Pass (2026-09-20)

Design/plan: `docs/superpowers/specs/2026-09-20-e3-sql-lora-sft-design.md`, `docs/superpowers/plans/2026-09-20-e3-sql-lora-sft.md` (both local, gitignored). Branch `tinker-custom-loras`. Budget context: user funded **$10 total** on Tinker (PRD's original ~$25-35 full-sweep estimate is not affordable — see budget math in the design spec).

**Recipe:** `Qwen/Qwen3-8B`, rank 32, `train_unembed=False`, `qwen3_disable_thinking` renderer. Subsampled **5,000** of the 99,600 sql train examples (`training/data/output/sql/train.jsonl`, fixed seed 0), 2 epochs, batch size 32 (157 batches/epoch, 314 total steps), `lr = get_lr("Qwen/Qwen3-8B") = 4.73e-4` (no sweep — single value, single seed, by design, given the budget).

**Loss:** 1.87 (batch 0) → mostly 0.05–0.15 by end of epoch 1, e.g. batch 156 (last) = 0.0507. Clean, monotonic-ish convergence, no signs of divergence.

**Export:** `train_unembed=False` worked as intended this time — `validate_adapter_config` passed cleanly on the first attempt (no `lm_head` in `target_modules`, unlike E0's first, uncorrected run).

**Local validation result — the actual headline number:**

| | B0 (no adapter) | sql LoRA, first pass |
|---|---|---|
| sql normalized EM (30 val examples, same scorer as the real benchmark) | 0.000 | **0.300** |

Scored via `training/eval_local.py`'s `score_predictions`, which calls the exact same `benchmarks.quality_metrics.score("sql", ...)` the real quality benchmark uses — this number is directly comparable to a future `benchmarks/results/quality_*.json` sql entry, not a bespoke metric.

**Cost (confirmed via `tinker billing usage`):** session `444503c2` used 1,535,084 training tokens → **$0.68** at $0.44/M — close to the $0.61 design-time estimate.

**Reading:** a single cheap (~5% of the funded budget, by estimate), unswept, single-seed run already produces a real, non-trivial quality lift over B0 on the domain with the biggest prior gap. This is a strong signal to continue — either legal/medical with the same recipe, or a quick LR check on sql first — with the remaining ~$9.

### Legal LoRA, first pass (same session)

Same recipe, `--domain legal`. Legal's train set is only 3,830 examples total (smaller than the 5,000 default subsample), so this run trained on all of it, 2 epochs, same rank/lr.

**Loss:** 4.97 (batch 0, epoch 0) → mostly 0.3–0.9 by end of epoch 1, noisier than sql's curve but trending down (legal's contract-clause task is harder/more varied than sql's schema+query format).

**Real bug hit and fixed along the way:** `training/eval_local.py`'s `score_predictions` assumed every domain scores against `task["reference"]` (a bare string) — true for sql/medical/summarizer, but `benchmarks.quality_metrics.score_legal` reads `task["meta"]["answers"]` (a *list* of acceptable gold spans) instead, and raised `KeyError: 'meta'` on the first attempt. Fixed by adding `build_score_task(domain, example)` to `eval_local.py`, which special-cases legal to wrap the single reference in a list (commit `54bd986`) — verified with a new unit test before rerunning.

**Separate, unrelated hiccup:** the local eval step (loading `Qwen/Qwen3-8B` + adapter via `transformers`/`peft`, `device_map="auto"`) crashed with `KeyError` inside `accelerate`'s disk-offload path (`accelerate/utils/offload.py`) the first time — root-caused to system memory pressure from unrelated local applications (a Minecraft/Java process alone was using ~9.5 GB RSS, alongside PyCharm/Chrome/Discord and two other Claude sessions on the same machine), which pushed `device_map="auto"`'s planner into a disk-offload strategy that hit a bug. Not a bug in this project's code or in the trained adapter — the adapter itself exported and validated cleanly on the first try. Worked around by forcing `device_map="cpu"` for the local eval (slower, but avoids the disk-offload hook path entirely) once memory pressure was confirmed via `free -h`/`ps aux --sort=-%mem`.

**Local validation result:**

| | B0 (no adapter) | legal LoRA, first pass |
|---|---|---|
| legal token-F1 (30 val examples) | 0.209 | **0.2785** |

A real but more modest lift than sql's (0.000 → 0.300) — legal's B0 starting point was already non-zero, and CUAD-style span extraction from long contract text is a harder task than sql's schema+query format.

**Cost (confirmed via `tinker billing usage`):** session `7b12a854` used 3,884,234 training tokens → **$1.71** — more than sql despite fewer training examples (3,830 vs 5,000), because legal's per-example sequences are much longer (contract context up to 2000 chars + question, vs sql's compact schema+query).

**Running total this session:** sql $0.68 + legal $1.71 + E0 hello-world (3 runs × 9,200 tokens) ≈ $0.01 ≈ **$2.40 of the funded $10**, before the medical run.

### Medical LoRA, first pass (same session)

Same recipe, `--domain medical`, default 5,000-example subsample (of 245,859 available), 2 epochs, same rank/lr. This run's local eval hit no memory issues (Minecraft/Java process from the legal run's hiccup was closed by then) — `device_map="auto"` worked cleanly, local eval took ~2 seconds instead of legal's CPU-forced multi-minute detour.

**Loss:** 3.27 (batch 0, epoch 0) → mostly 0.8–1.5 by end of epoch 1, e.g. batch 156 (last) = 0.7798. Higher starting loss than sql/legal (medical's free-text patient-question format is less structured than sql's schema+query or legal's span-extraction), but converges cleanly.

**Local validation result:**

| | B0 (no adapter) | medical LoRA, first pass |
|---|---|---|
| medical ROUGE-L (30 val examples) | 0.137 | **0.2680** |

Nearly a 2x lift over B0, using `benchmarks.quality_metrics.score("medical", ...)` (the `task["reference"]` shape, same as sql — medical didn't need the `build_score_task` legal special-case).

**All three P0 domains now show a real, adapter-attributable quality lift over B0 on a single cheap unswept pass:**

| Domain | B0 | LoRA (first pass) |
|---|---|---|
| sql (normalized EM) | 0.000 | 0.300 |
| legal (token-F1) | 0.209 | 0.2785 |
| medical (ROUGE-L) | 0.137 | 0.268 |

**Budget used today:** sql $0.68 + legal $1.71 + medical (billing not yet reconciled, design estimate ~$1.3) + E0 ~$0.01 ≈ **~$3.7 of the funded $10** — well within budget, no LR sweep or multi-seed runs done yet (deliberately, per the budget-scoped design).

## Model comparison table (2026-08-30)

Every base-model / system combination benchmarked this session, on identical qpiai LoRA weights. Full narrative in R2/R3 below.

### Qwen2.5-7B-Instruct — quality benchmark (sql / legal / medical, n=20/domain)

| System | Run | sql | legal | medical | overall | avg latency |
|---|---|---|---|---|---|---|
| Ours | 1 — no sys-prompt | 0.000 | 0.111 | 0.112 | 0.074 | 9.30s |
| Ours | 2 — matched sys-prompt | 0.050 | 0.102 | 0.112 | 0.088 | 5.88s |
| Ours | 3 — reproduced | 0.050 | 0.070 | 0.108 | 0.076 | — |
| Theirs (`/route`) | 1 | 0.150 | 0.025 | 0.130 | **0.101** | 5.65s |
| Theirs (`/route`) | 2 — reproduced | 0.150 | 0.033 | 0.118 | 0.100 | — |

Routing accuracy (theirs): 1.0, trivial with a 3-expert catalog. Legal near-zero for both — LEDGAR-classifier adapter tested against CUAD span-QA, a real benchmark/adapter mismatch shared by both systems, not a bug.

### Qwen3.5-9B — quality benchmark (legal / medical / code, n=20/domain)

| System | legal | medical | code | overall | avg latency | status |
|---|---|---|---|---|---|---|
| Ours | 0.008 | 0.041 | 0.000 | 0.017 | ~2 min/req | completed, 2h17m |
| Theirs | — | — | — | — | — | aborted by user |

No `sql` adapter exists for this base in qpiai's pool. `max_tokens` raised 512→2048 — model reasons visibly before answering.

### Multi-step orchestration — Qwen2.5-7B, our orchestrator only (20 tasks)

| Metric | Value | Reading |
|---|---|---|
| plan_f1 | 0.879 | picks close to the right adapter *set* |
| plan_order_correct_rate | 0.500 | right adapters, wrong order — half the time |
| avg_coverage | 0.833 | most oracle steps get invoked |
| avg_switches | 1.6 | real cross-adapter chaining happened, on hardware |
| avg_step_accuracy | 0.074 | per-step output rarely clears the quality bar |
| task_completion_rate | 0.000 | zero tasks fully completed end-to-end |
| avg_error_propagation_rate | 0.975 | one weak step drags down the whole chain |
| avg_latency_s | 28.9 | per full multi-step task |

Adaptive Minds has no equivalent multi-adapter endpoint to compare against — its own paper never demonstrates ≥2 adapters chained on one task.

### GPU-Time Efficiency ablation — our controller, forced eviction

| Condition | Domains | MAX_LORAS | Wall time | GPU-Time avg | GPU peak |
|---|---|---|---|---|---|
| Generous | 3 | 4 | 75.5s | 19.419 GiB | 19.427 GiB |
| Constrained | 3 | 2 | 88.7s | 19.427 GiB | 19.427 GiB |
| Generous | 4 | 4 | 80.5s | 19.524 GiB | 19.524 GiB |
| Constrained | 4 | 2 | 99.8s | 19.524 GiB | 19.524 GiB |

Memory flat within noise (<10MB) across every condition — rank-16/32 LoRA weights are too small to register against the ~19.4–19.5GB base+KV floor. Latency scales with eviction *frequency*: +17.5% at 3 domains, +24% at 4. The metric that moves here is time, not memory.

**Headline:** Qwen2.5-7B numbers are stable across two independent runs (within noise at n=20/domain, "ours" vs "theirs" is roughly a tie on raw quality). Qwen3.5-9B's LoRA-on-GDN path works technically, but its default chain-of-thought verbosity burns the token budget on meta-reasoning before answering — tanking both latency (~70×) and every lexical-overlap score. Not a fair comparison; reported for completeness.

---

## Problems

### P1 — LoRA Adapter Compatibility with vLLM
**Status:** Active blocker (partially mitigated)

vLLM only supports LoRA on attention and MLP target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Many public adapters fail to load for one of three reasons:

- **Wrong target modules** — adapter was trained with LoRA on `lm_head` or `model.embed_tokens`. vLLM rejects these outright.
- **Merged weights** — adapter weights have been merged back into the base model and published as a full fine-tuned model. No `adapter_config.json` exists; the repo looks like an adapter but is a ~6 GB full model.
- **DoRA** — adapter uses weight-decomposed LoRA (DoRA). vLLM does not yet support DoRA.

Out of 5 candidate adapters tested on Qwen2.5-3B-Instruct, only 2 loaded successfully (`summarizer`, `translation`). The other 3 failed: `sql` (lm_head), `code` (DoRA), `math` (embed_tokens).

**Mitigation:** Before adding any adapter to `adapters.yaml`, verify with:
```bash
curl -sL https://huggingface.co/<repo>/resolve/main/adapter_config.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('peft_type'), 'r='+str(d.get('r')), d.get('target_modules'))"
```
An adapter is safe to use if `peft_type` is `LORA` (not `DORA`) and `target_modules` contains only attention/MLP layer names.

---

### P2 — Public Adapter Pool is Sparse for Non-Llama Models
**Status:** Ongoing

The majority of publicly available LoRA adapters on HuggingFace target Llama 2/3 or Mistral. Qwen2.5-3B-Instruct has far fewer, and those that exist tend to be narrow experiment checkpoints (competition fine-tunes, homework assignments) rather than well-maintained capability adapters. This limits the diversity of the adapter pool, which directly affects the planner's routing decisions and the quality of RQ3 (clustering).

**Mitigation options:**
- Expand to Llama 3.2 3B with a proper HuggingFace token — much larger adapter ecosystem.
- Train a small set of purpose-built adapters covering the target task domains.

---

### P3 — Adapter Rank Must Be Known at vLLM Startup
**Status:** Design constraint

`--max-lora-rank` is set once at vLLM startup and cannot be changed without a restart. If a new adapter with a higher rank is added to the registry, vLLM will refuse to load it. This is a deployment constraint for the controller: the registry should validate that no adapter exceeds the configured rank limit, and the serving guide must document the chosen rank ceiling.

Currently running with `--max-lora-rank 256`.

---

## Results

### R1 — Full Pipeline Functional on RTX 2080 Ti (22 GB VRAM)
**Date:** 2026-08-13

vLLM serving Qwen2.5-3B-Instruct with `--enable-lora --max-loras 4 --max-lora-rank 256` runs stably. The LoRA controller (FastAPI on port 8001) successfully:
- Loads adapters into vLLM on demand
- Tracks loaded adapters with LRU eviction
- Routes inference requests to the correct adapter
- Hot-reloads the adapter registry without restart

Confirmed working adapters: `summarizer` (rank 16), `translation` (rank 16).

---

### R2 — Head-to-Head vs Adaptive Minds on Qwen2.5-7B, Same Weights
**Date:** 2026-08-29

Deployed both our stack (`lora_controller` + `orchestrator`) and adaptive-minds' own server against a single shared vLLM instance (Qwen2.5-7B-Instruct, RTX 3090 24GB, `--enforce-eager --swap-space 1` — this box only had 10GB system RAM, see P4) serving the *same* qpiai LoRA weights (`sql`, `legal`, `medical`, `chemistry` from `pavan01729/adaptive-minds-loras`). Goal: isolate orchestration-layer differences from weight differences.

**Quality benchmark** (60 tasks, 20 each sql/legal/medical, from `benchmarks/quality_tasks.json`, scored with existing `quality_metrics.py`):

| Domain | Ours (no sys-prompt) | Ours (matched sys-prompt) | Theirs (`/route`) |
|---|---|---|---|
| sql (exact match) | 0.0 | 0.05 | 0.15 |
| legal (CUAD F1 vs LEDGAR-trained adapter) | 0.11 | 0.10 | 0.025 |
| medical (ROUGE-L) | 0.11 | 0.11 | 0.13 |
| **overall** | 0.074 | **0.088** | **0.101** |
| avg latency/req | 9.30s | 5.88s | 5.65s |

Adaptive-minds' `/route` routing accuracy: 1.0 (trivial — 3-expert catalog, domain-obvious prompts). Legal scores near-zero for **both** systems: not a bug, a benchmark/adapter mismatch — `qwen25_legal_v1_grpo` was LEDGAR-trained (single-integer classification) but tested here against CUAD span-extraction QA. First-pass run with no adapter-level system prompt made our latency and sql score look artificially worse than theirs (9.30s vs 5.65s, 0.0 vs 0.15) — this was pure prompt-engineering confound (their catalog injects a domain system_prompt per adapter; ours didn't). After adding matching `system_prompt` fields to `adapters.yaml` and threading them through `client.chat_completions`/`app.infer` (see `src/lora_controller/{models,client,app}.py`), the latency gap closed almost entirely (9.30s → 5.88s vs their 5.65s) and sql improved (0.0 → 0.05). Residual overall gap (0.088 vs 0.101) is small-sample sql exact-match variance, not an infra difference.

**Multi-step benchmark** (20 hand-authored 2-4-step tasks, `benchmarks/multistep_tasks.json`, our `orchestrator` only — adaptive-minds has no equivalent multi-adapter endpoint to compare against; `code`/`grammar`/`summarizer` registered as passthrough-to-base since no matching adapter exists in the qpiai pool):

```
task_completion_rate:       0.0
avg_step_accuracy:          0.074
avg_plan_precision:         0.975
avg_plan_recall:            0.833
avg_plan_f1:                0.879
plan_order_correct_rate:    0.5
avg_coverage:               0.833
avg_error_propagation_rate: 0.975
avg_switches:               1.6
avg_latency_s:              28.9
replan_rate:                0.0
```

This is the concrete result the Adaptive Minds paper never provides for itself (per its own v2 text — no experiment shows ≥2 adapters invoked in sequence on one task). Here: `avg_switches=1.6` is direct proof real cross-adapter chaining happened on hardware, and `plan_f1=0.879` shows the planner (Qwen2.5-7B self-as-planner via a JSON planning scaffold, not yet a small dedicated model) picks close to the *right set* of adapters for a multi-step task — comparable to single-step routing accuracy, but for genuinely compositional queries. `plan_order_correct_rate=0.5` shows selection and sequencing are separable failure modes: right adapters, wrong order, half the time. `task_completion_rate=0.0` and `avg_error_propagation_rate=0.975` are the sobering finding — per-step output almost never clears the strict ground-truth threshold (exact-match/F1≥0.5/ROUGE≥0.5), and when one step is weak it drags down the whole chain. Likely cause: these adapters were fine-tuned narrowly (Spider SQL, LEDGAR classification) and were never asked to produce free-form chained-task output during training — zero-shot composition is harder than single-turn use.

**GPU-Time Efficiency ablation → resolves T1.** See T1 below.

**Baseline configuration disclosure (for the "was AM's config modified?" question):** adaptive-minds was run unmodified for this comparison — same vLLM instance (`VLLM_BASE=http://localhost:8000/v1`), our own hand-written `catalogs/bench3.yaml` (3 adapters: SQL/Legal/Medical, `hf_subdir` pointing at the exact same qpiai weights our controller used), their stock router prompt template/temperature/max_tokens (`router_temperature=0.0`, `router_max_tokens=16`, unedited `adaptive_minds/router.py`), and their stock `/route` endpoint (`sys_prompt_mode="trained"`, i.e. their own catalog-defined system prompts — not disabled or swapped). The only adaptive-minds source change made anywhere in this project was later and unrelated to this comparison: a hardcoded `timeout=180` in `adaptive_minds/common.py` was raised to `timeout=600` during the separate Qwen3.5-9B run (R4) because that reasoning-heavy model's generations routinely exceeded 180s — see R4 for details. R2's numbers above predate that patch and were not affected by it.

**Full 20-task multi-step benchmark spec** is `benchmarks/multistep_tasks.json` in this repo (hand-authored, not sampled from an existing dataset — see R2 body for why). Composition: 5 two-step tasks, 10 three-step tasks, 5 four-step tasks; every task has a distinct `oracle_adapter_sequence` (no duplicates across all 20). Domain coverage across all steps: summarizer 14, grammar 14, code 10, medical 9, legal 8, sql 5 (a task can touch a domain more than once across different steps, so these don't sum to a clean per-domain task count).

**Reproduced 2026-08-30** after the Qwen3.5-9B detour (R3) and restoring Qwen2.5-7B: rerunning the identical 60-task quality benchmark gave overall_avg_score 0.076 (ours) / 0.100 (theirs) — matching the original 0.088 / 0.101 within the noise expected at n=20/domain. Confirms the comparison is stable and not an artifact of one run.

---

### P4 — Host System RAM, Not Just VRAM, Is a Hard Vast.ai Deployment Constraint
**Status:** Resolved (workaround), worth flagging in `docs/serving.md`

A rented RTX 3090 24GB instance had only **10GB total system RAM**. vLLM's defaults assume much more: CUDA graph capture (35 shapes) thrashed page cache badly enough that per-shape capture time went 4.8s → 54s → climbing before we aborted it, and separately, vLLM's default `--swap-space 4` (GB, for CPU-side KV cache blocks) OOM-killed the engine process mid-request (confirmed via `dmesg`: `oom-kill: ... Killed process ... shmem-rss:7753344kB`). Fix: `--enforce-eager` (skip CUDA graph capture entirely) + `--swap-space 1 --max-num-seqs 32` (shrink host-side KV cache reservation). Stable afterward across ~2 hours of continuous benchmarking. **Lesson: check `free -h` on a fresh GPU rental, not just `nvidia-smi` — GPU spec alone doesn't guarantee the box can run vLLM's defaults.**

---

### P5 — Orchestrator's Planner Doesn't Reliably Emit `{step_N}` Placeholders
**Status:** Fixed 2026-08-29

Root cause of R2's multi-step `task_completion_rate=0.0`, found by re-querying `/task` directly for a failing case (`ms_002`, SQL→summarize): the planner (Qwen2.5-7B, zero-shot, instructed via system prompt to reference prior-step output as `{step_N}`) generated a step-2 prompt of `"Summarize the result of the SQL query in one sentence."` — no `{step_1}` anywhere in it. `Executor._resolve_prompt` faithfully does the substitution when the placeholder is present, but there's nothing present to substitute; the summarizer adapter ran completely blind and asked "which query?" instead of summarizing. This is an LLM instruction-following failure, not a template-engine bug — telling a self-routing planner to use a specific substitution syntax doesn't guarantee compliance.

**Fix** (`src/orchestrator/executor.py`): `_resolve_prompt` now takes the step's `depends_on` list. If a depended-on step's `{step_N}` placeholder is missing from the resolved prompt, its output is prepended as explicit `"Result of step N: ..."` context instead of trusting the plan text. A step that declares a dependency can no longer silently run blind, regardless of how the planner phrased the prompt. Not yet re-benchmarked against the full `multistep_tasks.json` suite post-fix — do that before citing an updated `task_completion_rate`.

---

### R3 — Qwen3.5-9B Migration Attempt: LoRA-on-GDN Works, but Reasoning Overhead Makes It Impractical Here
**Status:** Abandoned for this hardware/benchmark combination, 2026-08-30

Attempted to redo R2 on `Qwen/Qwen3.5-9B` (the paper's other base model) using qpiai's `qwen3.5-9b/` adapter pool (`legal`, `medical`, `chemistry`, `code` — **no `sql` adapter exists for this base**, so that domain was dropped).

**Real engineering hurdles cleared, in order:**
1. **LoRA target modules are GDN-specific**, not standard attention/MLP: `in_proj_a`, `in_proj_b`, `in_proj_qkv`, `in_proj_z`, `out_proj` (Gated DeltaNet hybrid architecture) alongside the usual `q/k/v/o/gate/up/down_proj`. Confirmed via upstream GitHub issues that vLLM's LoRA support for these module names is real but very recent (bugs like `in_proj_a` being silently dropped, an `expand_packed_lora` crash on partial LoRA) — needed the *latest* vLLM (0.28.0), not the 0.6.6.post1 pinned for Qwen2.5.
2. vLLM 0.28.0's CLI restructured entirely (`--help=<ConfigGroup>` sections) and **dropped `--swap-space`** — no direct replacement was needed once `--enforce-eager` was already in use.
3. `huggingface-cli` is gone in the newer `huggingface_hub` release paired with vLLM 0.28.0 — only the `hf` CLI exists now (`hf download`, `hf auth whoami`).
4. Hit the exact known FlashInfer-sampler `nvcc`-JIT crash from P4/`docs/serving.md` again, on a *different* vLLM version — same fix (`VLLM_USE_FLASHINFER_SAMPLER=0`) applied cleanly, confirming that workaround is version-independent.
5. After all of the above, **LoRA-on-GDN genuinely works**: `/v1/models` listed all 4 adapters, and a real inference call through the `legal` adapter returned coherent (if verbose) text — no crash, no garbage output.

**Why it was abandoned anyway:** Qwen3.5-9B is heavily CoT/reasoning-tuned by default. A single legal-adapter request at `max_tokens=512` spent the *entire* budget on visible meta-reasoning about whether to follow its system-prompt persona or answer the (mismatched-domain) question, and never produced an answer (`finish_reason: "length"`). Raising `max_tokens` to 2048 made requests complete, but at a real cost: **~45s/request** observed initially, settling to **~2 min/request average** in the full 60-task run (2h17m wall-clock for what took ~2 minutes total on Qwen2.5-7B — a **~70× slowdown**, not explained by model size alone (9B vs 7B) — it's the reasoning-token tax). Scores were also far worse than Qwen2.5-7B (overall_avg_score **0.0165** vs Qwen2.5-7B's 0.076–0.088): long CoT preambles dilute lexical-overlap metrics (F1/ROUGE) against short reference answers even when the final answer inside the trace might be reasonable — these ground-truth scorers were never designed for reasoning-model output and need a "strip the thinking, score the answer" step to be meaningful on this class of model.

Reverted to Qwen2.5-7B for R2's numbers to stand as the reference comparison. **Lesson for future base-model swaps:** check a model's default verbosity/reasoning behavior with one manual request before committing to a multi-hour benchmark loop — token budget and wall-clock cost do not transfer across "similarly-sized" models when one is CoT-tuned and the other isn't.

**Also added while debugging this:** both benchmark harnesses (`run_our_system.py`, `run_their_system.py` in `benchmarks/`) now checkpoint every task result to a JSONL file as it completes and resume from it on restart, specifically because the Qwen3.5 run's 2+ hour wall-clock with zero incremental output made "what if it crashes at task 58/60" a real, avoidable risk.

---

### R4 — Cross-Model Comparison: Qwen2.5-7B vs Llama-3.1-8B vs Qwen3.5-9B
**Date:** 2026-08-30

Same quality-benchmark methodology (`benchmarks/quality_tasks.json` domains, `quality_metrics.py` scorers) run against three base models, each paired with the closest available qpiai adapter set. Not all three cover the same domains — the qpiai pool doesn't have matching adapters for every base model, so gaps are filled with base-model passthrough (documented per row) rather than skipped, to keep every table row an actual measurement.

| Base model | Adapters (real) | Ours overall | Theirs overall | Notes |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | sql, legal, medical | 0.078 | 0.089 | Full 3-domain real-adapter coverage; result files `ours_20260830T085040Z.json` / `theirs_20260830T090235Z.json`. Reproduced 3× across 3 separate instances this session (0.074–0.088 / 0.089–0.101 range) — see R2. |
| Llama-3.1-8B-Instruct | medical only | 0.098 (all 3 domains, sql/legal via passthrough) | 0.186 (medical-only, their system has no passthrough concept) | Not directly comparable to the Qwen2.5-7B row — different domain coverage. Files `ours_20260830T093237Z.json` / `theirs_20260830T093713Z.json`. |
| Qwen3.5-9B | legal, medical, chemistry, code (no sql) | 0.016 | *(incomplete — instance terminated mid-run, ~5-10/60 tasks done, checkpoint lost with the instance)* | GDN hybrid architecture, heavily CoT-tuned. File `ours_20260830T143802Z.json`. Theirs would need a full rerun on a new instance to get a real number — not worth it given R3/R4's already-established finding that Qwen3.5's CoT dilutes these scorers regardless of which system serves it. |

**Real finding, not noise: Llama-3.1-8B's medical LoRA scores ~0.18, nearly double Qwen2.5-7B's ~0.11-0.13 on the same medical benchmark task set.** Both are qpiai-trained adapters on the same underlying architecture family (standard attention/MLP LoRA, not GDN), same evaluation methodology, same scoring function (ROUGE-L). This is either a genuinely better adapter (different/larger training data — the medical README describes reasoning-style SFT) or Llama-3.1-8B's base instruction-following is simply stronger for this ROUGE-L-scored free-form QA style. Worth digging into for the thesis if base-model choice is claimed to matter for adapter quality.

**Qwen3.5-9B is a cautionary tale for benchmark design, not a system failure.** overall_avg_score 0.016 is ~5× worse than Qwen2.5-7B on the same lexical-overlap scorers, but the underlying generations aren't necessarily 5× worse — they're diluted by unremovable chain-of-thought preambles that our scorers (exact-match, F1, ROUGE-L) were never designed to see past. `code` domain scored a flat 0.0 across all 20 tasks (pass@1 execution scorer can't find a runnable function inside a page of reasoning text). **Any future benchmark run against a reasoning-tuned model needs a "strip `<think>`-style preamble, score only the final answer" step in the harness** — this was flagged as a risk in R3 and is now confirmed with a second, independent 60-task run (0.0161 this time vs 0.0165 in the abandoned R3 attempt — the finding reproduces).

**New infra bug found and fixed while running Qwen3.5-9B "theirs":** adaptive-minds' own `adaptive_minds/common.py` hardcodes `timeout=180` (seconds) on its vLLM call with no env-var override. Qwen3.5-9B's CoT generations regularly exceed 180s, so their server was returning `HTTPConnectionPool ... Read timed out` for the majority of requests — not a quality problem, a silent availability problem that would have corrupted the "theirs" comparison with `None`-score errors rather than real (low) scores. Patched to `timeout=600` locally for this run. This is worth reporting upstream if this repo is ever used for anything beyond one-off benchmarking — any reasoning-tuned base model will hit the same wall.

**New TP-specific bug found and fixed:** `flashinfer`'s `comm/fd_exchange.py` uses `array.array[int]` generic-subscript syntax that only works on Python 3.11+; vLLM's tensor-parallel path imports this module unconditionally (regardless of `--disable-custom-all-reduce`) whenever `world_size > 1`. On this project's Python 3.10 venvs, any `--tensor-parallel-size 2` launch of vLLM 0.28.0 crashes with `TypeError: 'type' object is not subscriptable` before the engine even starts. Fixed by adding `from __future__ import annotations` to the top of that one file in site-packages (defers annotation evaluation, sidesteps the runtime subscript). Single-GPU deployments never hit this since the flashinfer TP all-reduce communicator is only imported for `world_size > 1`.

---

## Experiments

### E1 — Doc-to-LoRA: Training Purpose-Built Adapters from Document Corpora
**Status:** Superseded by `PRD_custom_LoRA_adapters.md` (see E2 below) — same core idea (train our own domain LoRAs, restrict target modules to attn/MLP for vLLM compatibility), now underway for real on `tinker-custom-loras`.

Train 5-6 LoRA adapters from scratch on domain-specific document collections, giving the system a controlled adapter pool with unambiguous domain boundaries. Motivated by P1 (vLLM compatibility) and P2 (sparse public adapters) — both problems disappear when we train our own.

**Proposed domains:** Medical (PubMed), Legal (FreeLaw), Code (GitHub), Finance (SEC filings), Science (arXiv), Multilingual (OPUS translation pairs).

**Training stack:** PEFT + trl `SFTTrainer` on Qwen2.5-3B-Instruct. LoRA target modules restricted to `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` to guarantee vLLM compatibility. Rank 16-64 depending on domain complexity.

**What to measure:**
- Domain accuracy on held-out test sets (does the medical adapter outperform base model on medical Q&A?)
- Routing clarity: can the planner reliably pick the correct adapter from description alone?
- Weight space structure: do Doc-to-LoRA adapters cluster by domain in RQ3 experiments?

---

### E2 — Tinker Hello-World SFT on `Qwen/Qwen3-8B` (PRD stage E0)
**Status:** Done, 2026-09-20, branch `tinker-custom-loras`

First real train → export → verify cycle through the Tinker API on the project's actual target model, before spending real domain-data budget on it (E3). Toy task: 20 identical single-turn conversations teaching an unguessable Q→A pair (`"What is the secret code?"` → `"XYZZY-42"`), rank-32 LoRA, 20 steps of `forward_backward`/`optim_step`, Adam lr 2e-4.

**Loss curve** (full run, `training/hello_world_sft.py`):

| Step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 10 | 15 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|
| Loss | 8.0119 | 5.6583 | 3.3250 | 1.4601 | 0.4718 | 0.0466 | 0.0058 | 0.0006 | 0.0007 | 0.0001 |

**Two real bugs found and fixed (matter for E3's actual domain training scripts too):**

1. `service_client.create_lora_training_client_async(base_model=..., rank=...)` defaults `train_unembed=True`. For `Qwen/Qwen3-8B` this makes Tinker LoRA-adapt `lm_head`, which vLLM's LoRA loader rejects (`lm_head.weight is unsupported LoRA weight` — same failure class as P1/the disabled sql adapter in `adapters.yaml`, this time from Tinker's own default on the real target model). Fix: always pass `train_unembed=False`. Caught immediately by `training/validate_adapter_config.py` on the very first live run — the validator did its job.
2. `tokenizer.apply_chat_template(..., return_tensors="pt")` returns a `BatchEncoding` in the installed `transformers==5.5.4` (pinned indirectly — `tinker_cookbook` requires `<=5.5.4`), not a bare tensor. Must extract `.input_ids` before calling `model.generate()`.

**Final verification (local, `transformers`+`peft`, no vLLM — see below):**
- Base `Qwen/Qwen3-8B` output for the prompt: `'The secret code is **"Hello, World!"**.'` (does not know the real answer, as expected)
- Same base model + exported LoRA adapter: `'XYZZY-42'` (exact match)

**Scope decision:** the live vLLM boot ("→ vLLM" in the PRD's E0 description) was deferred to E1 rather than done here — local GPU is only 8GB (not enough for any Qwen3 variant in vLLM), and E1 already needs a vast.ai rental for the real Qwen3-8B baseline (B0), so the first real vLLM+Qwen3-8B boot happens there instead of being duplicated on a throwaway smoke test. Full design rationale in `docs/superpowers/specs/2026-09-20-e0-tinker-hello-world-design.md` (local, gitignored).

**Renderer naming trap avoided:** `tinker_cookbook.renderers.get_renderer("qwen3_disable_thinking", ...)` is for the `Qwen3` family (our target). `"qwen3_8_disable_thinking"` is a different, unrelated model line (`Qwen/Qwen3.8-*`, version "3.8") that only looks similar in text — do not conflate in E3.

---

## Theories

### T1 — Scheduling Story May Shift to Memory Efficiency
**Status:** Measured 2026-08-29 — **resolved, in the opposite direction from what was flagged.**

The thesis plan flagged this as a key decision point: if adapter load/unload latency is negligible (<10ms), the RQ1 narrative should shift from latency optimisation to GPU memory efficiency. Measured directly: our `lora_controller`'s real LRU load/unload path (not vLLM's internal static-registration paging — adapters were registered dynamically, no `--lora-modules` at boot), forcing genuine eviction by round-robining requests across more domains than `MAX_LORAS` slots, Qwen2.5-7B rank-32 adapters, RTX 3090:

| Condition | requests | domains | MAX_LORAS | wall time | GPU-Time avg | GPU peak |
|---|---|---|---|---|---|---|
| generous | 15 | 3 (sql/legal/medical) | 4 (no eviction) | 75.5s | 19.419 GiB | 19.427 GiB |
| constrained | 15 | 3 | 2 (evict every req) | 88.7s | 19.427 GiB | 19.427 GiB |
| generous | 20 | 4 (+chemistry) | 4 (no eviction) | 80.5s | 19.524 GiB | 19.524 GiB |
| constrained | 20 | 4 | 2 (evict every req) | 99.8s | 19.524 GiB | 19.524 GiB |

**GPU memory is flat, no measurable difference between conditions (<10MB, noise floor)** — the GPU-Time Efficiency metric (as defined: time-weighted average memory) has **zero discriminative power at this scale**. Rank-32 LoRA weights on a 7B model are tens of MB each, invisible against the ~19.4-19.5GB base-weights+KV-cache floor. **Latency is where the real, measurable cost lives**: forced eviction cost +17.5% wall time at 3 domains/2 slots, +24% at 4 domains/2 slots — and the effect scales with eviction *frequency*, not adapter count.

**Implication for the thesis:** don't lead RQ1's evaluation with the GPU-Time Efficiency integral as originally specified — it will report a flat line regardless of scheduling quality, at LoRA-rank scale on a single mid-size base model. Either (a) report **eviction-forced latency overhead** as the primary systems cost metric instead, or (b) keep GPU-Time Efficiency but validate it at a scale where it should move — many more concurrently-tracked adapters (dozens, per S-LoRA's hundreds-to-thousands regime) and/or much higher LoRA rank, where per-adapter weight size becomes non-negligible relative to the KV-cache/model-weight floor.

### T2 — Plan Regeneration Frequency Will Be the Core Tradeoff of RQ1
The continuous planner can regenerate the task plan after every step, but doing so adds an LLM call on the critical path. The interesting research question is not whether to replan, but *when* — triggered by low-confidence output, unexpected format, or just always. The naive orchestrator (fixed plan, no replanning) will be the baseline that makes this tradeoff measurable.

### T3 — Parallel Step Execution Requires Careful Adapter Slot Management
If the plan contains independent steps that can run in parallel, they may require different adapters to be resident simultaneously. With `--max-loras 4`, a plan with 4 parallel steps using 4 different adapters saturates the GPU slot budget. The scheduler must either execute parallel steps within the slot budget or serialise them — and that choice affects latency in a way the naive orchestrator won't capture.
