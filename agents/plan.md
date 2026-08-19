## Thesis: Dynamic LoRA Adapter Management for Multi-Step LLM Tasks

A system that dynamically selects, loads, and unloads LoRA adapters on LLMs to solve complex multi-step tasks — combining a dedicated small planner model, system-level adapter scheduling, and a rigorous benchmark.

## Research Questions (in order of priority)

### RQ1: Adapter Scheduling as a Systems Problem (Priority 1)
Can a planner that is aware of system constraints (GPU memory, adapter loading latency, pipeline structure) improve throughput and latency?
- **Continuous planning loop**: planning is not a one-shot operation — the planner re-evaluates after each step. When a task completes and a LoRA is no longer needed, it is immediately unloaded from the LLM and freed from GPU memory. The planner then decides the next adapter to load based on remaining/new tasks and current system state.
- Adapter prefetching: predict upcoming needs from the task plan, overlap loading with computation
- Eviction/caching strategies for adapter weights
- Optimize load/unload ordering to minimize GPU memory churn
- Reactive replanning: if a step fails or produces unexpected output, the planner can revise the remaining plan and adjust which adapters are needed

### RQ2: Small Planner vs. Self-Routing (Priority 2)
Does a dedicated small language model (1-3B params) as an external planner outperform the pattern where the large model routes itself (as in Adaptive Minds)?
- Compare on latency, cost, and task quality
- Explore planner architectures: prompted vs. fine-tuned vs. RL-trained

### RQ3: LoRA Representations, Clustering, and Structural Analysis (Priority 3)
What is the most effective way to represent/describe LoRA adapters, and what can we learn from analyzing adapter weight spaces?

**Representation strategies:**
- Textual descriptions (natural language capability summaries)
- Learned embeddings of adapter weights (e.g., embed the adapter matrices themselves)
- Performance profiles (accuracy on probe tasks)
- Hybrid representations combining multiple signals
- How does representation quality affect routing accuracy?

**Clustering and structural exploration:**
- Cluster LoRA adapters by their weight matrices (e.g., SVD of adapter weights, cosine similarity)
- Do clusters correspond to task families? (e.g., do all translation adapters cluster together?)
- Can we discover latent adapter "skills" that cut across task categories?
- Analyze adapter similarity vs. task similarity — are adapters that solve related tasks geometrically close?
- Investigate whether adapter interpolation between cluster neighbors produces useful intermediate capabilities
- Dimensionality reduction (t-SNE, UMAP) of adapter weight space — visualize the "LoRA landscape"
- Can clustering predict adapter composability? (which adapters can be merged vs. which conflict?)

### RQ4: Benchmark for Task-Adaptive Weight/Architecture Switching (Priority 4)
Build a benchmark to measure how well a system adapts its own weights or architecture based on a given task.
- Multi-step tasks requiring diverse domain skills (translation → summarization → code, etc.)
- Baselines: single adapter, no adapter (base model), oracle routing, Adaptive Minds self-routing
- Ablations: number of available adapters, task complexity, adapter similarity

**Metrics:**
- Task quality, switching overhead, adapter selection accuracy, end-to-end latency

- **GPU-Time Efficiency (new metric):** Measures GPU memory utilization integrated over time, capturing the tradeoff between peak memory and duration. Defined as:

  `GPU-Time Cost = (1/T) * ∫ GPU_mem(t) dt`  (average GB over total duration)

  Example: System A uses 4 GB for 5 minutes → GPU-Time Cost = 4.0 GB·avg. System B uses 5 GB for 1 min then 2 GB for 4 min → GPU-Time Cost = (5×1 + 2×4)/5 = 2.6 GB·avg. System B is more efficient despite a higher peak, because aggressive adapter unloading frees memory faster.

  This metric directly rewards the continuous planner's ability to unload adapters as soon as tasks complete — a system that holds all adapters loaded scores worse than one that loads/unloads on demand, even if both produce the same output quality.

  Report alongside: peak GPU memory, total GPU·minutes (area under the curve, not averaged), and task quality — so readers can see the full quality-vs-resource tradeoff.


## Key Prior Work to Position Against

| Paper | Year | Key Difference from Our Work |
|-------|------|------------------------------|
| Adaptive Minds (LoRA-as-Tools) | 2025 | Uses base LLM as router, no separate planner, no system-aware scheduling |
| AgentFlux | 2025 | Decouples tool selection/arg gen, but no multi-adapter planning |
| S-LoRA / dLoRA | 2023-24 | Infrastructure only, no intelligent planning layer |
| LoRA-Switch | 2024 | Token-level routing within single task, not multi-step planning |
| MeteoRA | 2024 | MoE-style routing, not agentic multi-step decomposition |
| LoRA-Augmented Generation | 2025 | Retrieval-based selection, no lifecycle/scheduling awareness |


## Candidate research directions
1. RL environment for planning/routing — another LLM checks task completion. Risk: LLM-as-judge is a reward model; multi-step tasks are where judges get gamed. Ground reward in executable verification where the task admits it.

## Implementation Plan

### Phase 1: Infrastructure
- Set up vLLM with a base model (e.g., Llama 3 or Mistral) with `--enable-lora`
- Collect/train 5-8 LoRA adapters spanning diverse tasks
- Build adapter pool service with load/unload API

### Phase 1.5: Doc-to-LoRA Adapter Training (Experiment)
Train 5-6 purpose-built LoRA adapters from document corpora so the planner has a controlled, well-understood adapter pool with clear domain boundaries. This directly addresses the public adapter scarcity problem (P2) and produces adapters that are guaranteed vLLM-compatible (no lm_head/DoRA issues).

**Concept:** Given a collection of domain documents, fine-tune a LoRA adapter on that corpus using supervised instruction-following pairs derived from the documents. Each adapter becomes a "document expert" — it knows its domain deeply and knows the limits of that domain, making routing decisions clearer.

**Proposed domains (5-6 adapters):**
- **Medical** — PubMed abstracts, clinical notes; adapter answers medical questions
- **Legal** — court rulings, contracts; adapter interprets and drafts legal text
- **Code** — GitHub code + docstrings; adapter generates and explains code
- **Finance** — earnings reports, financial news; adapter analyzes financial data
- **Science** — arXiv abstracts; adapter summarizes and explains research papers
- **Multilingual** — parallel translation pairs; adapter translates between languages

**Training pipeline:**
1. Collect domain documents (public datasets: PubMed, FreeLaw, GitHub Code, SEC filings, arXiv, OPUS)
2. Convert documents to instruction-following pairs (e.g., "Q: What does this abstract say about X? A: ...")
3. Fine-tune LoRA on Qwen2.5-3B-Instruct with PEFT + trl (`SFTTrainer`)
4. Validate: adapter must only target attention/MLP modules (vLLM constraint)
5. Register in `adapters.yaml` with rich textual descriptions for the planner

**Why this matters for the thesis:**
- Gives full control over adapter content — you know exactly what each adapter knows
- Domain boundaries are sharp, making routing ground truth unambiguous
- Enables RQ3 clustering: adapters trained on distinct corpora should cluster by domain
- Makes the benchmark (RQ4) more rigorous — task sequences have clear correct adapter sequences
- The Doc-to-LoRA pipeline itself is a reusable contribution for the LoRA registry platform (Phase 6)

### Phase 2: Automated Deployment (`scripts/deploy.py`)
Single-command deployment to vast.ai: search cheapest GPU, rent, copy code, install deps, start all 3 services, poll until healthy, run end-to-end test, optionally destroy.

```
python scripts/deploy.py [--keep] [--instance-id ID] [--gpu-ram N] [--disk N]
```

### Phase 3: Continuous Planner + System-Aware Scheduling (RQ1)
- Build continuous planning loop with reactive replanning
- Implement system-aware scheduling (prefetching, caching, memory management)
- Adapter lifecycle management: load on demand, unload immediately after task completion

### Phase 4: Planner Architecture Comparison (RQ2)
- Build small planner model (prompted → fine-tuned → RL-trained progression)
- Compare against self-routing baseline (large model routes itself)
- Evaluate latency, cost, and quality tradeoffs

### Phase 5: Adapter Representation & Clustering Study (RQ3)
- Implement multiple representation strategies
- Run clustering experiments on adapter weight spaces
- Visualize LoRA landscape, analyze structure
- Integrate best representation into the planner

### Phase 6: Benchmark + Evaluation (RQ4)
- Design multi-step task benchmark with ground-truth adapter sequences
- Implement baselines (no adapter, single adapter, oracle, self-routing)
- Run full evaluation: quality, latency, GPU-Time Efficiency, scaling behavior
- Ablation studies on adapter pool size, task complexity, representation method

### Phase 7: Tau Agent Wrapper (HuggingFace Integration)
Wrap the thesis system as a set of Tau tools so the full LoRA orchestration stack runs inside [HuggingFace Tau](https://github.com/huggingface/tau) — a minimalist, provider-neutral terminal coding agent.

**Concept:** Tau tools are typed async functions with schemas. We expose adapter management and inference as Tau tools, then let the Tau agent loop handle planning and execution. This replaces (or sits alongside) our custom orchestrator with Tau's native agent loop, giving us a clean comparison point between our bespoke planner and a general-purpose agent framework doing the same job.

**Implementation steps:**
1. Implement Tau tools:
   - `load_adapter(name: str)` — loads a LoRA adapter via the controller API
   - `unload_adapter(name: str)` — unloads a resident adapter
   - `infer(adapter: str, prompt: str)` — runs inference through a loaded adapter
   - `list_adapters()` — returns available adapters with descriptions for the agent to reason about
2. Wire tools into a `tau_agent` loop backed by the vLLM base model via `tau_ai`
3. Run the same multi-step benchmark tasks through Tau and compare against our custom orchestrator (RQ2 extension)
4. Build an optional `AgentHarness`-based TUI frontend for interactive demos
5. Package as a reusable Tau extension and publish to HuggingFace Hub

**Why this matters:**
- Provides a real-world agent framework baseline for RQ2 — does our custom planner outperform Tau's general agent loop on LoRA-switching tasks?
- Validates that the controller API is general enough to be consumed by external agent frameworks without modification
- The Tau frontend makes the system accessible as an interactive terminal demo for the thesis defense
- Positions the work as ecosystem-compatible, not just a standalone research artifact

### Phase 8: LoRA Registry Platform (was Phase 7 / Phase 6)
Build an open platform (analogous to HuggingFace but specialized for LoRA adapters) where:
- **Publishing**: Users upload LoRA adapters with structured metadata — base model, task description, training data summary, hyperparameters, adapter rank/architecture
- **Benchmarks**: Each adapter has standardized benchmark results (auto-evaluated on upload); community can submit additional benchmark runs
- **Search & Discovery**: Semantic search over adapter capabilities (not just keyword matching); agents can query the registry API to find adapters matching a task description
- **Agent API**: Programmatic access for agentic systems to search, evaluate, download, and compose adapters at runtime — the platform becomes the "adapter pool" from the thesis architecture
- **Compatibility metadata**: Track which adapters are compatible (same base model, rank), which compose well together, clustering/similarity info from RQ3
- **Community features**: Ratings, usage stats, adapter lineage (which adapter was fine-tuned from which), reproducibility info

## Strengths and Weaknesses Analysis

### Strengths

**S1: Fills a real gap between infrastructure and intelligence.**
Existing work splits cleanly into "serving infrastructure" (S-LoRA, dLoRA, Punica) and "routing algorithms" (MeteoRA, LoRA-Switch). This thesis bridges both — a planner that is simultaneously a task reasoner and a resource scheduler. No existing paper does this.

**S2: The continuous planning loop is a genuine architectural contribution.**
Adaptive Minds and other agentic systems treat adapter selection as stateless per-step decisions. A planner that maintains state across the full task lifecycle — tracking what's loaded, what's coming next, when to prefetch, when to evict — is closer to how OS schedulers work and is a more principled design.

**S3: LoRA weight space analysis (RQ3) has high discovery potential.**
This is the kind of research that can produce surprising findings. If adapter clusters reveal meaningful structure, that's publishable independent of the rest of the thesis. If they don't, that's also a useful negative result.

**S4: GPU-Time Efficiency metric is simple, novel, and reusable.**
Other researchers can immediately adopt it. Simple metrics that capture something no existing metric captures are valuable contributions on their own.

**S5: The platform (Phase 6) creates lasting impact beyond the thesis.**
If even a prototype exists, it validates the full research pipeline end-to-end: representations from RQ3 power the search, the benchmark from RQ4 provides standardized evaluation, and the agent API demonstrates the planner from RQ1/RQ2 working against a real adapter ecosystem.

**S6: Clear fallback structure.**
RQ5 as a stretch goal means the thesis is complete without it. Each RQ produces a standalone contribution even if later phases are cut.

### Weaknesses

**W1: Scale of adapter pool may limit conclusions.**
With 5-8 adapters, clustering analysis (RQ3) may not reveal meaningful structure — you need enough adapters for clusters to emerge. Mitigation: supplement with adapters from HuggingFace (there are hundreds of public LoRAs for popular base models). Aim for 30-50+ adapters for the clustering study even if only 5-8 are used in the scheduling experiments.

**W2: Benchmark (RQ4) risks being too synthetic.**
If multi-step tasks are hand-crafted, reviewers may question whether the benchmark reflects real-world usage. Mitigation: ground tasks in real workflows — e.g., academic paper writing pipeline (literature search → outline → draft → cite → proofread), customer support (classify intent → retrieve info → translate → respond). Include at least one task set derived from real user traces if possible.

**W3: Small planner vs. self-routing comparison (RQ2) depends heavily on model choice.**
Results with Phi-3-1B as planner might not generalize to other small models. Mitigation: test at least 2-3 small models at different scales (1B, 3B, 7B) to show trends rather than single data points.

**W4: vLLM adapter loading latency may dominate all other effects.**
If loading/unloading a LoRA takes 500ms+ and inference takes 2s, the scheduling optimizations from RQ1 might show marginal improvements. Mitigation: measure adapter load/unload latency upfront (Phase 1). If it's negligible compared to inference, pivot the scheduling story toward memory efficiency rather than latency.

**W5: Platform (Phase 6) scope creep risk.**
Building "HuggingFace for LoRAs" is an enormous engineering task. A thesis committee won't evaluate you on production polish. Mitigation: define an MVP scope now — upload/download adapters, metadata + search API, basic benchmark display. No auth system, no scaling, no community features in V1.

**W6: Reactive replanning (RQ1) is hard to evaluate rigorously.**
How do you measure "the planner replanned well"? You need tasks that fail in predictable ways to test replanning. Mitigation: design specific failure injection scenarios — e.g., adapter produces low-confidence output, adapter fails to load, task output doesn't match expected format — and measure recovery quality.

**W7: Representation comparison (RQ3) could be inconclusive.**
If all representation strategies perform similarly for routing, the contribution is weak. Mitigation: ensure the adapter pool is diverse enough that representations matter. Test on edge cases — adapters with similar descriptions but different capabilities, or similar weights but different tasks.

## Roadmap

### Month 1-2: Foundation
- [ ] Literature review: deep-read all papers from the prior work table + their references
- [ ] Set up GPU infrastructure (cloud or university cluster)
- [ ] Set up vLLM with base model, verify LoRA hot-swapping works
- [ ] **Critical experiment**: measure adapter load/unload latency and memory impact — this determines how much RQ1 scheduling matters
- [ ] Collect 5-8 diverse LoRA adapters (mix of self-trained and HuggingFace)
- [ ] Build minimal adapter pool service (registry + load/unload API)
- [ ] Deliverable: working vLLM setup that can swap adapters on demand

### Month 3-4: Core System (RQ1 + RQ2)
- [ ] Implement naive orchestrator (sequential plan execution, no scheduling intelligence)
- [ ] Implement continuous planner with system state tracking
- [ ] Add prefetching and eviction strategies
- [ ] Implement self-routing baseline (large model picks its own adapters)
- [ ] Implement small planner variants (prompted, fine-tuned)
- [ ] Run initial comparison: continuous planner vs. naive orchestrator vs. self-routing
- [ ] Deliverable: working end-to-end system with measurable scheduling improvements

### Month 5-6: Representations & Clustering (RQ3)
- [ ] Expand adapter pool to 30-50+ adapters (HuggingFace sourcing)
- [ ] Implement all representation strategies (text, weight embeddings, performance profiles, hybrid)
- [ ] Run clustering experiments (SVD, cosine similarity, UMAP/t-SNE visualization)
- [ ] Analyze: do clusters match task families? Can you predict composability?
- [ ] Test adapter interpolation between cluster neighbors
- [ ] Integrate best representation into the planner, measure routing accuracy improvement
- [ ] Deliverable: LoRA landscape visualizations, representation comparison results

### Month 7-8: Benchmark (RQ4) + Evaluation
- [ ] Design benchmark task suite (10-20 multi-step tasks across diverse domains)
- [ ] Implement all baselines (no adapter, single adapter, oracle, Adaptive Minds self-routing)
- [ ] Implement GPU-Time Efficiency metric with profiling infrastructure
- [ ] Run full evaluation across all systems and metrics
- [ ] Ablation studies: pool size, task complexity, representation method, planner size
- [ ] Failure injection experiments for reactive replanning evaluation
- [ ] Deliverable: complete benchmark results, comparison tables, plots

### Month 9-10: Platform (Phase 6) + Writing
- [ ] Build LoRA registry platform MVP (upload, metadata, search API, benchmark display)
- [ ] Integrate agent API — demonstrate planner querying the registry to discover adapters
- [ ] Begin thesis writing (introduction, related work, methodology chapters)
- [ ] Deliverable: working platform prototype, first thesis draft

### Month 11-12: Thesis Completion + Optional RQ5
- [ ] Complete thesis writing (results, discussion, conclusion)
- [ ] If time permits: explore special token switching (RQ5) as a proof-of-concept
- [ ] Thesis review cycles with advisor
- [ ] Prepare defense presentation
- [ ] Deliverable: final thesis, defense

### Key Decision Points
- **End of Month 2**: If adapter load/unload latency is <10ms, pivot RQ1 focus from latency optimization to memory efficiency
- **End of Month 4**: If small planner shows no advantage over self-routing, focus the thesis narrative on scheduling + representations instead of planner architecture
- **End of Month 6**: If clustering reveals no meaningful structure with available adapters, focus RQ3 on representation comparison only and cut the structural analysis
- **End of Month 8**: Go/no-go on RQ5 based on remaining time and energy
