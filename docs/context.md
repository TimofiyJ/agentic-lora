# Research Context: Agentic LoRA Resource Planning

## Core Research Idea

Design an **agentic architecture** that plans the execution of complex multi-step tasks by dynamically connecting and disconnecting LoRA adapters — treating adapter load/unload as first-class planning actions, not just infrastructure operations.

The key insight: adapter selection is a *sequential planning problem*, not a stateless routing decision. A dedicated planning agent must reason about which domain-specialized adapters to activate at each step of a multi-step task, when to prefetch upcoming adapters, and when to evict adapters that are no longer needed — all under real GPU memory constraints.

**Scope decision:** This research does not pursue LoRA internals (weight merging, MoE routing, representation learning) or RL-based environments as the primary focus. The contribution is the **planning layer** that orchestrates a fixed pool of black-box adapters.

---

## Main Research Question

> **How can an agent create and execute a plan for a multi-step task by dynamically loading and unloading domain-specific LoRA adapters, operating under real GPU memory constraints?**

Sub-questions:
- Does a dedicated small planner outperform the large model routing itself (Adaptive Minds pattern)?
- Can system-aware scheduling (prefetching, eviction) improve throughput and latency beyond naive sequential execution?
- How does the planner handle failures — replanning when a step produces unexpected output?
- What is the right metric for GPU resource efficiency alongside task quality?

---

## The Rival Paper: Adaptive Minds (LoRA-as-Tools)

**Adaptive Minds** (2025) is the most directly related prior work. It frames LoRA adapters as tools that the base LLM calls using function-calling syntax. The large model itself selects which adapter to use at each step.

**What it does well:**
- Proves the concept of treating adapters as callable tools
- Demonstrates multi-step adapter selection in practice
- Uses the large model's existing instruction-following to drive adapter choice

**Critical gaps this research fills:**

| Dimension | Adaptive Minds | This Work |
|---|---|---|
| Planner | Large LLM routes itself | Dedicated small planner (1–3B) |
| System awareness | None — no GPU state tracking | Tracks loaded adapters, prefetches, evicts LRU |
| Memory constraints | Ignored — assumes adapters always available | Central constraint; max_loras=4 is a hard limit |
| Planning horizon | Per-step, reactive | Full task plan upfront + reactive replanning |
| Resource metric | Not reported | GPU-Time Efficiency (memory × time integral) |
| Scheduling | None | Prefetching, LRU eviction, concurrent loading |

The key controlled experiment: **same base model, same adapter pool, same tasks** — only the planning mechanism differs. This isolates the value of a dedicated planner over self-routing.

---

## Related Papers

### Serving Infrastructure (not the focus, but the substrate)

| Paper | Contribution | Relevance |
|---|---|---|
| **S-LoRA** (Berkeley/Stanford, MLSys 2024) | Stores thousands of adapters in host memory, fetches to GPU on demand; Unified Paging for adapter weights + KV cache; up to 30× throughput over HF PEFT | Our controller implements similar load-on-demand logic |
| **dLoRA** (PKU, OSDI 2024) | Dynamically merges/unmerges adapters per replica using credit-based batching; up to 57.9× throughput over vLLM | Key insight: neither always-merged nor always-unmerged is optimal |
| **Punica** (UW/Duke, MLSys 2024) | SGMV CUDA kernel for batching multiple LoRA adapters in one kernel launch; ~2ms/token overhead | Backbone of vLLM multi-LoRA support |
| **vLLM multi-LoRA** | Production stack: `--enable-lora`, `--max-loras`, LRU CPU cache | Our serving substrate |

### Routing / Mixture over Adapters (not the focus, but informative baselines)

| Paper | Contribution | Relevance |
|---|---|---|
| **Arrow** (Microsoft/Mila, ICML 2024) | Zero-shot token+layer routing via SVD of adapter weight directions; no retraining needed | Zero-shot routing baseline; shows single-shot routing without planning |
| **PHATGOOSE** (ICML 2024) | Post-hoc per-expert gate training for token-level routing | Another zero-shot routing baseline |
| **LoRAHub** (COLM 2024) | Task-level adapter composition via CMA-ES optimization | Shows task-level (not step-level) blending |
| **X-LoRA** (2024) | Trained scaling head for token+layer-wise adapter mixing | Token-level routing, not agentic planning |
| **MeteoRA** (2024) | MoE-style routing within a model | Not agentic multi-step decomposition |
| **LoRA-Switch** (2024) | Token-level routing within a single task | Not multi-step planning |

### Planning Literature (motivating the difficulty)

| Paper | Finding | Implication |
|---|---|---|
| **PlanBench** (Kambhampati et al., NeurIPS 2023) | LLMs score ~5% on Mystery Blocksworld (obfuscated); GPT-4o only 35.5% on standard Blocksworld | LLMs are weak standalone planners |
| **LLMs Still Can't Plan; Can LRMs?** (2024) | o1-preview reaches 97.8% on standard Blocksworld but only 52.8% on obfuscated; degrades beyond ~20 steps | Even large reasoning models struggle with hard planning |
| **NATURAL PLAN** (Google, 2024) | All models below 5% on 10-city trip planning | Natural language planning is harder than benchmarks suggest |
| **LLM-Modulo** (Kambhampati, ICML 2024) | LLM as generator inside verifier-guided loops, not standalone planner | Motivates external verifiers / dedicated planners |
| **LATS** (Zhou et al., ICML 2024) | MCTS unifying reasoning + acting + planning; 94.4% pass@1 on HumanEval with GPT-4 | Search-based planning outperforms single-pass |
| **ReAct, Reflexion, ToT, CoT** | Prompting scaffolds for reasoning and action interleaving | Standard baselines for agentic planning |

### Weight Generation / Self-Modification (related but out of scope)

| Paper | Contribution |
|---|---|
| **Text-to-LoRA** (Sakana AI, ICML 2025) | Hypernetwork generates LoRA weights from a natural-language task description in one forward pass |
| **SEAL** (MIT, NeurIPS 2025) | Model generates its own finetuning data via RL loop; persistent weight updates |
| **Transformer²** (Sakana AI, 2025) | SVD-based z-vectors; two-pass inference: identify task, compose z-vectors |
| **Drag-and-Drop LLMs** (2025) | Maps unlabeled prompts → LoRA weights via hyper-convolutional decoder |

These are out of scope for the current research direction — the focus is planning over a fixed adapter pool, not generating or modifying adapters.

### Agentic Frameworks (comparators for Phase 7)

| Framework | Relevance |
|---|---|
| **LangGraph** | Explicit state-graph control flow; comparable orchestration pattern |
| **AutoGen** | Conversational multi-agent; baseline for multi-agent routing |
| **HuggingFace Tau** | Typed async tool framework; planned wrapper target for thesis demo |
| **RouteLLM** (ICLR 2025) | Routes between strong/weak models to save cost; complementary to adapter routing |

### Parametric Knowledge Approaches (critical baselines)

| Paper | Contribution | Role in This Work |
|---|---|---|
| **Parametric RAG (PRAG)** (2024–25) | Encodes domain knowledge into base weights statically (full fine-tune or merged adapters) | Critical baseline: does dynamic routing beat statically-merged knowledge? |
| **Standard RAG** | Retrieves domain documents at query time | Baseline: do adapter weights beat retrieval? |

---

## System Architecture

```
Client
  └─→ Orchestrator (port 8002)
        ├─ Planner    — generates a step-by-step adapter plan for the task
        ├─ Executor   — runs plan waves, chains step outputs, handles replanning
        └─ Scheduler  — prefetch/unload adapter lifecycle concurrently with execution
              └─→ LoRA Controller (port 8001)
                    ├─ Registry  — adapters.yaml with descriptions
                    ├─ LRU eviction when max_loras exceeded
                    └─→ vLLM (port 8000)
                          └─ Qwen2.5-3B-Instruct + LoRA adapter pool
```

**Key design decisions:**
- `max_loras=4` is the hard GPU constraint the planner must reason about
- Prefetching: next-wave adapters load concurrently while current wave executes
- LRU eviction: when at capacity, least-recently-used adapter is unloaded
- Replanning: if a step fails or produces low-confidence output, planner revises remaining steps
- `adapter_was_preloaded` flag measured per step to quantify scheduling benefit

---

## Current Implementation State

### What is built (Phase 1 + Phase 3)

**LoRA Controller** (`src/lora_controller/`, port 8001):
- `GET /adapters` — list all registered adapters
- `GET /adapters/loaded` — dict of GPU-resident adapters with load time, use count
- `POST /adapters/{name}/load` — load adapter; LRU eviction if at capacity
- `POST /adapters/{name}/unload` — unload from GPU
- `POST /infer/{adapter_name}` — inference, auto-loads if not resident
- `POST /registry/reload` — hot-reload adapters.yaml

**Orchestrator** (`src/orchestrator/`, port 8002):
- `POST /task` — full execution: plan → wave execution → scheduling events → result
- `GET /plan` — preview plan without executing
- `ContinuousScheduler` — concurrent prefetch and unload via async tasks
- `Planner` — system-aware: passes currently-loaded adapter names to LLM prompt
- `Executor` — wave-based execution with prefetch concurrency and replanning support

**vLLM** (port 8000):
- Qwen2.5-3B-Instruct with `--enable-lora --max-loras 4 --max-lora-rank 256`
- Tested on RTX 2080 Ti (22 GB VRAM)

**Known working adapters:** `summarizer` (rank 16), `translation` (rank 16)

### Active Problems

**P1 — vLLM adapter compatibility:** vLLM only supports LoRA on attention/MLP modules. Many public adapters fail (lm_head target, DoRA, merged weights). Out of 5 tested, only 2 loaded.

**P2 — Sparse public adapter pool for Qwen2.5:** Most public LoRAs target Llama 2/3 or Mistral. Solution: train purpose-built adapters (Doc-to-LoRA, E1).

**P3 — max-lora-rank is fixed at startup:** Cannot load higher-rank adapters without restart. Currently running at rank 256.

---

## Research Questions & Evaluation

### RQ1: System-Aware Scheduling (Primary)

Can a planner aware of GPU state (what's loaded, load latency, memory capacity) improve throughput and latency over naive sequential execution?

**Measurable via:**
- Adapter prefetch hit rate (`adapter_was_preloaded` per step)
- End-to-end task latency vs. naive orchestrator (no prefetching)
- GPU-Time Efficiency metric (see below)

**Key theory (T2):** The interesting question is not whether to replan but *when* — triggered by low-confidence output, unexpected format, or always. The naive baseline (fixed plan, no replanning) makes this measurable.

### RQ2: Small Dedicated Planner vs. Self-Routing (Secondary)

Does a dedicated 1–3B planner outperform the large model routing itself (Adaptive Minds)?

**Comparison:** Same base model, same adapters, same tasks — only planning mechanism differs.
- Planner variants: prompted → fine-tuned → (optional) RL-trained
- Metrics: task quality, adapter selection accuracy, latency, cost

### RQ3: Adapter Clustering and Representations (Tertiary)

What is the best way to describe adapters so the planner can route correctly? Do adapter weight spaces reveal meaningful structure?

**Out of scope for current focus** — deprioritized in favor of RQ1 and RQ2.

### RQ4: Benchmark (Quaternary)

Multi-step cross-domain task suite with ground-truth adapter sequences.

---

## Evaluation Framework

### The Fairness Principle

The adapter pool contains domain knowledge baked into weights. Any baseline must access **equivalent knowledge** via a different mechanism — not less knowledge.

`This system = Base LLM + adapter pool (domain-specialized LoRA weights) + dedicated planner`

### Baselines

| Baseline | What it proves |
|---|---|
| **Self-routing (Adaptive Minds)** | Value of dedicated planner over self-routing |
| **Parametric RAG (PRAG)** | Dynamic routing vs. statically merged domain knowledge |
| **Standard RAG** | Adapter weights vs. retrieval for specialized tasks |
| **Best-single-adapter** | Multi-step routing vs. static expert selection |
| **All-adapters merged** | Dynamic switching vs. blending all knowledge |
| **Oracle routing** | Upper bound — ceiling on what any planner can achieve |

### Primary Metrics (task quality leads)

| Metric | Description |
|---|---|
| **Task completion rate** | Did the full multi-step task succeed end-to-end? (headline) |
| **Step accuracy** | Correct output at each step |
| **Adapter selection accuracy** | Did the planner pick the right adapter per step? |
| **Error propagation rate** | How often does a wrong step corrupt downstream steps? |

### Secondary Metric: GPU-Time Efficiency (novel contribution)

`GPU-Time Cost = (1/T) × ∫ GPU_mem(t) dt`   (average GB over total task duration)

Example: System A holds all adapters loaded → 4 GB × 5 min = 4.0 GB·avg. System B uses aggressive unloading → 5 GB for 1 min + 2 GB for 4 min = 2.6 GB·avg. System B is more efficient despite a higher peak.

This metric directly rewards the continuous planner's adapter lifecycle management. Report alongside peak GPU memory, total GPU·minutes, and task quality.

### Domain-Specific Benchmarks per Adapter

| Adapter | Benchmark |
|---|---|
| Medical | MedQA, PubMedQA |
| Legal | LegalBench, CUAD |
| Code | HumanEval, MBPP |
| Finance | FinQA |
| Science | MMLU science subsets, SciQ |
| Translation | BLEU on WMT pairs |

### Benchmark Task Design

Tasks must require ≥2–3 distinct domain steps — tasks solvable by one adapter make the routing contribution invisible.

Example valid tasks:
- *"Read this medical paper → extract statistical findings → write Python to plot them → translate to Spanish"*
- *"Classify this legal clause → summarize in plain English → estimate financial exposure → generate code to flag it"*

Dataset target: 20 instances (5 two-step, 10 three-step, 5 four-step). No two instances share the same adapter sequence.

---

## Planned Adapter Pool (Doc-to-LoRA Training, Experiment E1)

Train 5–6 purpose-built LoRA adapters from domain document corpora to solve vLLM compatibility and sparse public adapter problems:

| Domain | Training Data | Benchmark |
|---|---|---|
| Medical | PubMed abstracts, clinical notes | MedQA, PubMedQA |
| Legal | Court rulings, contracts (FreeLaw) | LegalBench, CUAD |
| Code | GitHub code + docstrings | HumanEval, MBPP |
| Finance | SEC filings, earnings reports | FinQA |
| Science | arXiv abstracts | MMLU science, SciQ |
| Multilingual | OPUS parallel translation pairs | BLEU on WMT |

**Training stack:** PEFT + trl `SFTTrainer` on Qwen2.5-3B-Instruct, LoRA target modules restricted to attention/MLP only for vLLM compatibility.

---

## Key Thesis Strengths

**S1 — Fills the gap between infrastructure and intelligence.** Serving systems (S-LoRA, dLoRA) and routing algorithms (Arrow, PHATGOOSE) are studied separately. This work bridges both — a planner that is simultaneously a task reasoner and a resource scheduler.

**S2 — Continuous planning loop is a genuine architectural contribution.** Adaptive Minds and other systems treat adapter selection as stateless per-step decisions. Maintaining state across the full task lifecycle — tracking what's loaded, what's coming next, when to prefetch, when to evict — is closer to how OS schedulers work.

**S3 — GPU-Time Efficiency metric is simple, novel, and reusable.** Other researchers can immediately adopt it. Captures the tradeoff between peak memory and duration that no existing metric addresses.

**S4 — The most critical comparison (vs. Adaptive Minds) is tightly controlled.** Same model, same adapters, same tasks. Only the planning mechanism varies. This isolates the contribution cleanly.

---

## Key Thesis Weaknesses

**W1 — Adapter load/unload latency may be negligible.** If loading a LoRA takes <10ms vs. 2s inference, scheduling optimizations may show marginal latency improvements. Pivot story to GPU memory efficiency if so. (Theory T1)

**W2 — Benchmark may be too synthetic.** Hand-crafted tasks may not reflect real workflows. Mitigate by grounding tasks in real pipelines (academic paper writing, customer support triage).

**W3 — Small planner results depend on model choice.** Test at 1B, 3B, 7B scales to show trends not single data points.

**W4 — Parallel step execution creates slot contention.** With max_loras=4, a plan with 4 parallel steps using 4 different adapters saturates GPU slots. Scheduler must manage within budget or serialize. (Theory T3)

---

## Implementation Phases

| Phase | Status | Description |
|---|---|---|
| 1 — Infrastructure | Done | vLLM + LoRA Controller + adapter registry |
| 2 — Automated Deployment | Planned | `scripts/deploy.py` single-command vast.ai deploy |
| 3 — Continuous Planner | Done | ContinuousScheduler, Executor waves, replanning |
| 4 — Planner Comparison | Planned | Small planner vs. self-routing (RQ2) |
| 5 — Clustering | Deprioritized | Adapter weight space analysis (RQ3) |
| 6 — Benchmark | Planned | Full eval suite (RQ4) |
| 7 — Tau Agent Wrapper | Optional | HuggingFace Tau integration for demo |
| 8 — LoRA Registry Platform | Optional | Open adapter discovery platform |

---

## Open Research Gaps (from Literature Review)

The intersection of adapter routing and sequential planning is **largely unexplored**:

1. No mature system treats "load LoRA k" as an action in a planning/RL policy.
2. LATS-style MCTS with adapter-switch actions is unexplored.
3. Routing (§1.3 systems) and serving (§1.2 systems) are studied separately — no joint optimization of accuracy and SLO under memory constraints.
4. No PlanBench-analogue for "skill/adapter selection as planning" — a benchmark measuring whether a router+library can plan a sequence of skill invocations to solve a compositional task.

This work directly addresses gap 4 (the benchmark) and gap 1/3 (the planning layer under serving constraints).
