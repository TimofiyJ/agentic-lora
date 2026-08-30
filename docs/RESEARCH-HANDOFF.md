# Research Handoff: Agentic Resource Planning for LoRA Adapters

> Consolidated context document. Everything needed to resume this research in a fresh
> session, with no prior conversation available.
> Last updated: 28 August 2026.

---

## 1. Research statement

**Working title:** Dynamic LoRA Adapter Management for Multi-Step LLM Tasks
*(alternative framing: Agentic Resource Planning for Adapter Libraries)*

**Main research question:**

> How can we build an agent that produces a plan to execute a multi-step task and
> connects/disconnects the specific LoRA adapters trained for each step, under real
> resource constraints?

Two things are load-bearing and must both appear in every framing of this work:

1. **Planning** — the adapter sequence is *planned*, not classified per query.
2. **Limited resources** — loading, unloading, merging, and evicting adapters has real
   cost, and the plan must account for it.

Existing work does one or the other. Nothing found does both.

---

## 2. Scope decisions (decided 28 Aug 2026)

### In scope
- Agentic architecture that plans over a fixed library of pre-trained LoRA adapters.
- System-level adapter scheduling: load/unload, merge/unmerge, eviction, prefetch.
- A benchmark for adapter-selection-as-planning (multi-step, compositional).
- Ablation: **small dedicated planner model vs. self-routing** (base model routes itself).
  Cheap, high-information, and directly interrogates whether a separate planner is needed.

### Out of scope (explicitly dropped)
- **LoRA-internals research** — adapter weight-space analysis, representation learning over
  adapter parameters, novel PEFT variants. Not the contribution.
- **RL environment with LLM-judge for task completion** — dropped as a primary direction.

### Deferred / uncertain
- **Diffusion-based token generation to sharpen the switch decision.** Originally motivated
  as "a diffusion LM denoises a whole draft, so the adapter-switch boundary may be visible
  earlier than in left-to-right decoding." Interesting but couples two independent research
  bets and strands the work off the mature autoregressive multi-LoRA serving kernels
  (SGMV/MBGMV). Revisit only after a working baseline exists.

**Consequence of dropping the RL environment:** the planner still needs a training and
evaluation signal. With RL out, that signal has to come from (a) the benchmark's gold
adapter-sequences via supervised imitation, or (b) prompted planning with no training at
all. Decide this early — it determines whether the benchmark needs gold plans or only gold
outcomes.

**Consequence of dropping adapter-representation research:** the planner still has to
*address* adapters somehow. A minimal representation choice is unavoidable. Default to
natural-language metadata (name, description, keywords) as Adaptive Minds does; treat it as
an engineering choice, not a research question.

---

## 3. Theoretical foundation: the options framework

The chosen conceptual grounding is the **options framework** in hierarchical RL:

> Sutton, R.S., Precup, D. & Singh, S. (1999). "Between MDPs and semi-MDPs: A framework for
> temporal abstraction in reinforcement learning." *Artificial Intelligence* 112:181–211.

**Core idea.** Classical RL picks one small action per time step, which is hopeless for long
tasks. An **option** is a temporally-extended action — a chunk that lasts many steps and is
treated as a single decision. Every option has exactly three parts:

| Part | Question it answers | Adapter analogue |
|---|---|---|
| **Initiation set** | When may this start? | Which adapter applies in this state |
| **Internal policy** | What happens while running? | The loaded LoRA generating tokens |
| **Termination condition** | When do we stop? | The switch signal — **the open problem** |

Above the options sits a **policy over options** — that is the planner.
"Semi-MDP" simply means actions may last a variable number of steps.

**Why this framing pays off:**
- It names the hardest problem. "When should the planner switch adapters?" is the
  termination-condition problem, which has decades of existing literature.
- It supplies a known failure mode: the **interruption problem** — commit to a running
  option and waste time, or interrupt freely and thrash. Thrashing is expensive here
  because every switch has a measurable systems cost. This is a bug that can be predicted
  and measured rather than discovered by accident.
- It connects the planning half and the adapter half of the project through one vocabulary.
- "Option discovery" is the existing name for *where the skills come from*.

**Framing discipline — important.** Present this as an **analogy, not an identity**. Write
"we draw on the options framework, where each adapter plays the role of a
temporally-extended action." Do **not** write "we formulate adapter selection as a
semi-MDP" unless the formal apparatus (state space, transition model, discounting over
variable-duration actions, optimality argument) is actually delivered — reviewers will
expect it and its absence reads as decoration.

**Honest disanalogies to name yourself, before a reviewer does:**
- **No natural initiation set.** Any LoRA can technically be loaded in any state. Options
  assume some actions are unavailable in some states. Whether to *impose* learned
  preconditions per adapter is itself a possible contribution.
- **Options are usually discovered; these adapters are pre-trained and given.** This is
  "planning with a fixed option set," a real but narrower setting.

**Terminology note:** the correct phrasing is "the LoRA analogue of the options framework"
or "an extension of the options framework to adapter and resource planning." (Not
"prodigy" — that word means a child genius.)

---

## 4. The rival paper: Adaptive Minds

**Full record**
- *Adaptive Minds: Empowering Agents with LoRA-as-Tools* (v1) / *…with LoRAs as Tools* (v2)
- Pavan C Shekar, Aswanth Krishnan — QpiAI, Bengaluru, India
- arXiv:2510.15416 — v1: 17 Oct 2025; v2: 3 Jun 2026 [cs.AI]
- Venue: CompLearn Workshop, ICML 2026 — **non-archival**, not archivally peer-reviewed
- Code: github.com/qpiai/adaptive-minds (Apache-2.0, ~18 stars, 0 forks)
- Adapters: huggingface.co/pavan01729 (personal account, not the org)

**Version warning.** v1 and v2 are almost different papers. v1: Llama-3.1-8B-Instruct,
5 adapters (Chemistry, Finance, AI, Medical, General), LangGraph/FastAPI/Streamlit,
LoRA r=16 α=32 dropout 0.1 on q/k/v/o projections. v2: Qwen2.5-7B-Instruct and Qwen3.5-9B,
up to 30 adapters, a formalism, entropy-gated mode switching, vLLM serving. **Always
specify which version when citing.**

### What it actually does

**LoRA-as-tools abstraction.** Each adapter `aᵢ` with weights `Δθᵢ` becomes a tool
`τᵢ = (nameᵢ, descriptionᵢ, {sub_query:str}, generateᵢ)`. The tool registry is auto-generated
from adapter metadata, so adding an adapter needs no agent-code change.

**Routing mechanism.** The **base LLM itself is the router**, zero-shot, reading
natural-language metadata and emitting an adapter identifier. No separate classifier, no
learned gate, no embedding index. Invalid outputs fall back to a general adapter or the base
model. Mode selection uses **mean predictive entropy over the first T=16 tokens**:
H<0.8 → router mode (single adapter); H>1.5 → agent mode; between → router with option to
escalate. Thresholds fixed by hand on internal validation.

**Granularity: per-query, one-shot.** Not per-token, not per-step.

**Serving.** Two runtimes — a PEFT runtime studying GPU-resident / CPU-staged / on-demand
placement, and vLLM hosting all 30 adapters in one engine. Reported: adapter switching
**<1 ms**; router mode **3.1× faster** than baseline (3.49 s vs 10.81 s mean); adapter
placement changes end-to-end latency by **<5%**, so generation dominates. Hardware: a single
NVIDIA L40S 48 GB, single seed, small n.

**LoRA hyperparameters (v2).** Reasoning track: r=32, α=64, dropout 0.05, all-linear
targets, bf16, 500 steps, seq 2048. Specialist track (SQL/Cypher/Bash/SPARQL/Mermaid/regex):
r=32/α=64, 400–600 SFT steps, 200 GRPO steps with execution reward for regex.
~30 domain adapters: r=16, α=32, targets {q,k,v,o,gate,up,down}, 100 SFT steps.

### The critical finding

> **Adaptive Minds claims multi-step agentic reasoning but never demonstrates it.**

The v2 formalism defines a K-step Think → Generate-with-adapter → Synthesize loop that
reduces to routing at K=1. But **no experiment in the paper shows two or more distinct
adapters invoked in sequence to solve one task.** Verified against the v2 HTML, the
formalism (§4), Table 3, Appendix F (GPQA Diamond), Appendix G (agent-loop pseudocode), and
Appendix H (coverage benchmark).

The authors themselves concede agent mode *underperforms* routing on strict-format tasks
(−20.7 pp on GPQA Diamond) and that "multi-tool use is not inherently beneficial when direct
expert selection is already sufficient."

**This is the single most important fact for this project. The multi-step lane is open.**

### Its evaluation

Routing accuracy (Table 1): Llama-3.1-8B, 5 adapters, 29 queries — keyword 48.3% vs router
100.0%. Qwen3.5-9B, 30 adapters, 60 queries — keyword 31.7% vs router 98.3%.

Specialist strict-scorer accuracy (Table 2, Qwen2.5-7B), base → specialist:
Spider-dev SQL 12.3 → 41.7; Text2Cypher 0.0 → 39.3; nl2bash 8.4 → 17.3; LC-QuAD SPARQL
0.0 → 59.0; MermaidSeqBench 4.5 → 21.2; PII 2.7 → 79.0; Qiskit-HumanEval-hard 10.6 → 15.2;
LEDGAR 0.0 → 84.0; ChEMBL SMILES 6.3 → 36.7. MATH-500 69.2 → 67.2 (regression);
GSM8K-250 86.0 → 88.0.

Routing vs multi-step (Table 3, 100 domain-matched MMLU, Llama-3.1-8B):
vanilla 56.0 / router 62.0 / **agent 58.0** — agent mode is not uniformly better.

**Routing fails without domain cues (Table 7).** PII 93%, Bash 97%, Mermaid 96%, Regex 94% —
but natural-language SPARQL **23%** and domain-neutral function-calling **1%**. Routing works
only when the query surfaces trigger vocabulary. The authors call it "a mechanism, not a
uniform property."

**Baselines compared:** keyword matching, hybrid keyword+semantic, base model, pinned
specialist, vanilla/router/agent. **No comparison against Arrow, PHATGOOSE, LoRAHub, X-LoRA,
LoRAuter, or any MoE/learned-router method.** A significant evaluation gap — do not repeat it.

### Its stated limitations (useful — these are your openings)

- Untested beyond 30 adapters; "whether routing holds at hundreds or thousands with heavier
  overlap is open."
- Only SFT/GRPO recipes; "CPT, DPO, and preference- or distillation-based recipes are unexplored."
- Fixed entropy thresholds, no sensitivity sweep.
- Agent depth underperforms routing on strict-format tasks.
- Small eval (n=20–30 per benchmark, single seed, single primary base), no human evaluation.
- Weak-base gains are "format acquisition rather than knowledge acquisition" — do **not**
  cite LEDGAR +84.0 or SPARQL +59.0 as capability gains.
- Headline future work: **"training-as-a-tool"** — an agent that commissions a new adapter on
  demand when none matches.

### Standing

Low-visibility preprint. Non-archival workshop paper, ~18 GitHub stars, 0 forks, no external
reproductions, no OpenReview thread, no substantive discussion found on Reddit/X/HuggingFace,
citation count effectively zero (could not be confirmed — treat as unverified). Code appears
released and runnable with Docker and reproduction scripts for Tables 1–3.

**Implication:** this is a paper to position against, not a wall to route around. Treat it as
a reproducible baseline representing the **K=1 special case** of your planner.

---

## 5. Differentiation: what is already covered vs. what is open

| Intended contribution | Covered by Adaptive Minds? | Verdict |
|---|---|---|
| Dedicated planner selecting/loading/unloading adapters across a multi-step task | **Claimed, not demonstrated.** Entropy gate + ReAct loop, no dedicated planner, no sequential multi-adapter evidence | **OPEN** — lead with this |
| System-level scheduling under serving constraints (merge/unmerge, eviction, prefetch) | **No.** AM reports a placement *ablation* (<5% effect) but makes no scheduling *decisions* | **OPEN at the intersection** |
| Benchmark for adapter selection as planning | **No.** AM's routing eval is 60 hand-crafted single-domain queries | **OPEN, high value** |

**Already covered — reposition, do not claim as novel:**
per-query semantic routing; vLLM multi-adapter serving; adapter hot-swap latency (<1 ms);
the "adapters as callable tools" abstraction itself; the finding that routing works only when
queries surface domain signal.

**Genuinely novel — lead with these:**
1. Sequential multi-step adapter *planning* with a real planner and structured termination.
2. Serving-aware scheduling as a *decision* — when to merge, evict, prefetch the next adapter.
3. A compositional adapter-selection-as-planning benchmark.
4. The options-framework formalization of adapter use, with serving costs as transition costs.

**Verified open as of Aug 2026:** nobody has instantiated a LoRA adapter as an *option* with
initiation set, intra-option policy, and termination condition, nor formulated adapter
scheduling as a semi-MDP. Recent LLM-guided hierarchical RL (LDSC arXiv:2503.19007; HiPER
2602.16165; Select-to-Act/HRLLI 2606.22350; Multi² 2606.03698) treats policies/skills as
temporally-extended actions but never parameter-space adapters. Voyager is the closest
temporal-abstraction analogue but uses executable code skills, not adapters, and does no
serving reasoning. **This bridge is unclaimed.**

---

## 6. Positioning and category

**Not** agentic knowledge management — that is about retrieval and context assembly, and
nothing here concerns what the model *knows*. This is **conditional computation for
sequential decision-making**: the planner chooses computation (which weights are live), not
information.

**Three viable framings, each with different peers and venues:**

| Framing | Venues | Peers | Contribution reads as |
|---|---|---|---|
| Modular DL / conditional computation | ICML, ICLR | MoE, adapter routing | "routing becomes sequential" |
| **Agentic planning** ← recommended | NeurIPS D&B, ACL | tool-use agents | "the action space is weights, not API calls" |
| Systems | MLSys, OSDI | dLoRA, S-LoRA | "scheduling under a planner's demands, not a request trace" |

The planning framing is strongest for a first paper: it is where the benchmark contribution
lands and where the gap is genuinely empty.

**One-line positioning statement:**

> Prior work routes between models (RouteLLM), between experts inside a model (MoE), or
> between adapters for a single query (Arrow, PHATGOOSE, Adaptive Minds); this work routes
> between adapters *across a multi-step task under resource constraints*, which makes
> selection a planning problem rather than a classification problem.

### The four baselines every reviewer will ask about

1. **Tool / function calling.** The one that will always be asked. A LoRA is a skill; so is a
   tool call. The defensible answer: weights should host skills that change the model's
   *style of computation* (a reasoning mode, a formalism, a domain prior), while tools handle
   skills that fetch or compute a discrete result. Benchmarks: BFCL, ToolLLM, τ-bench.
2. **Prompt / context switching.** The cheapest alternative — swap the system prompt or
   few-shot exemplars instead of the weights. **Run this early.** If a prompt swap matches a
   LoRA swap on the benchmark, the whole architecture is unmotivated.
3. **Static merged multi-task model.** Task arithmetic, or one model tuned on the union of
   skills. Task arithmetic gives small but consistent gains; TIES can degrade. If a merge
   matches the router, dynamism buys nothing.
4. **MoE.** Already does conditional computation, learned end-to-end, no external planner.
   The differentiator: adapters are independently trained, post-hoc composable, and
   **addressable by name** — a planner can reason about them; Mixtral's expert 47 is not a
   thing you can plan over.

---

## 7. Related work landscape

### 7a. Adapter routing / selection

| System | Year / venue | Routing unit | Learned or zero-shot | Multi-step? | Planner? |
|---|---|---|---|---|---|
| **Adaptive Minds** (2510.15416) | 2025–26, ICML-W non-arch. | per-query | zero-shot (base LLM over NL metadata) | **No (asserted only)** | No |
| **LoRAHub** (Huang et al., 2307.13269) | 2023, COLM 2024 | per-task | few-shot gradient-free (CMA-ES over adapter coefficients) | No | No |
| **Arrow / library-of-LoRAs** (Ostapenko et al., 2405.11157) | 2024, ICML | per-token, per-layer | zero-shot — routes by \|dot product\| of activation with top right-singular vector of each adapter's ΔW; paired with MBC clustering | No | No |
| **PHATGOOSE** (Muqeeth et al., 2402.05859) | 2024, ICML | per-token, per-module | post-hoc learned sigmoid gates | No | No |
| **X-LoRA** (Buehler & Buehler, 2402.07148) | 2024, APL Mach. Learn. | per-token, per-layer | learned dense gating, dual forward pass (~5M-param scaling head) | No | No |
| **HydraLoRA** (2404.19245) | 2024, NeurIPS Oral | per-sample | trained; 1 shared A + multiple B experts | No | No |
| **LoRAMoE** (2312.09979) | 2024, ACL | per-token per FFN | trained; localized balancing constraint | No | No |
| **Self-MoE / MiXSE** (2406.12034) | 2024 | per-token | self-specialized experts + self-optimized router | No | No |
| **LoRAuter** (2601.21795) | 2026 | per-query | task-representation routing, 1500+ adapters | No | No |
| **S2L (Skill-to-LoRA)** (2606.16769) | 2026 | per-skill | retrieval over skill metadata | swaps per skill, not planned | No |
| **Voyager** (2305.16291) | 2023, TMLR | per-subtask | LLM-written code skills, text retrieval | **Yes — but code skills, not adapters** | curriculum + iterative prompting |

### 7b. Multi-adapter serving substrate (this is your scheduling foundation)

- **Punica** (Chen et al., MLSys 2024, arXiv:2310.18547). Introduces the **SGMV** kernel
  batching xAB for different LoRAs in one launch against a single base copy. **12× throughput**
  vs prior systems, **~2 ms latency per token**.
- **S-LoRA** (Sheng et al., MLSys 2024, arXiv:2311.03285). All adapters in host memory,
  fetched on demand. **Unified Paging** for variable-rank adapter weights + variable-length KV
  cache. **MBGMM/MBGMV** kernels support mixed ranks without padding, so cost scales with
  *average* rank. Serves **thousands of adapters on one GPU**; **up to 30× vs HF PEFT, 4× vs
  naive vLLM**. Adapter count bounded by host memory, not GPU memory.
- **dLoRA** (Wu et al., USENIX OSDI 2024, pp. 911–927). **The closest systems analogue to your
  scheduling contribution.** Key insight: neither always-merged nor always-unmerged is optimal.
  Uses "a credit-based batching algorithm to decide when to merge and unmerge, and a
  request-adapter co-migration algorithm to decide when to migrate." **Up to 57.9× vs vLLM,
  26.0× vs HF PEFT, and up to 1.8× lower average latency than S-LoRA.**
- **V-LoRA** (2411.00915): quantifies mode-switch cost — unmerged inference +27–140 ms,
  mode switch 53 ms. Useful concrete transition costs for a semi-MDP.
- **ForkKV** (2604.06370, 2026): copy-on-write disaggregated KV cache for multi-LoRA *agent*
  serving with LRU eviction, ~2.5× throughput vs prefix caching.
- Also: CaraServe (CPU-assisted, rank-aware), Chameleon (adapter-aware scheduler avoiding
  head-of-line blocking), Cannikin (SLO-tailored), InfiniLoRA.

> **The gap in one sentence:** these systems solve many-adapter serving but treat adapter
> choice as an *exogenous request attribute*. None plans the *sequence* of adapters for a task.
> Your contribution is coupling the planner's adapter sequence to the scheduler's
> merge/evict/prefetch decisions.

### 7c. Production stacks

**vLLM multi-LoRA:** `--enable-lora`, `--lora-modules name=path`; `max_loras` (adapters per
batch), `max_lora_rank` ∈ {8,16,32,…,512}, `max_cpu_loras` (LRU host cache, must be ≥
`max_loras`); `fully_sharded_loras`; per-active-adapter CUDA-graph capture. Cold-start
mitigated by pre-loading with a dummy request.

**HuggingFace PEFT multi-adapter API:** `add_adapter`, `set_adapter`, `disable_adapter`,
`add_weighted_adapter` with combination types `linear`, `cat`, `ties`, `dare_ties`,
`dare_linear`, `magnitude_prune`, and SVD variants.

Also: LoRAX/Predibase, NVIDIA NIM multi-LoRA, TensorRT-LLM LoRA.

### 7d. LoRA variants — mergeability (the property that governs scheduling)

`h = W₀x + (α/r)·BAx`, with `B ∈ ℝ^(d×r)`, `A ∈ ℝ^(r×k)`, `r ≪ min(d,k)`.

Because `ΔW = (α/r)BA` is a plain additive term, it can be **merged** (`W' = W₀ + ΔW`, zero
inference overhead, one base copy per adapter) or kept **unmerged** (separate xAB branch,
enables batching many adapters against one shared `W₀`). **Every serving and scheduling
decision in this project turns on that choice.**

- **Cheap to hot-swap** (reduce to plain additive BA at inference): vanilla LoRA, VeRA,
  rsLoRA, LoRA+, PiSSA.
- **Complicating:** **DoRA** (per-column magnitude vector + normalization breaks clean
  additive merging and unmerged batching), **AdaLoRA** (variable effective rank per layer
  breaks fixed-rank batching kernels), **LoHa/LoKr** (non-additive product structure),
  **QLoRA** (merging into a quantized base is lossy — typically served unmerged).
- **GaLore** is not an adapter at all — low-rank *gradient* projection, nothing to swap.

**Practical default for this project:** vanilla LoRA, fixed rank across the library. It keeps
the Punica/S-LoRA kernels usable and keeps merge/unmerge a clean binary decision.

### 7e. Weight generation (context for the deferred diffusion direction)

- **Text-to-LoRA / T2L** (Charakorn et al., ICML 2025, arXiv:2506.06105). Hypernetwork
  generating a LoRA from a natural-language task description "in a single inexpensive forward
  pass"; reconstructed adapters "match the performance of task-specific adapters" and
  zero-shot generalize to unseen tasks. 55M/34M/5M-parameter variants.
- **Drag-and-Drop LLMs** (2506.16406). Unlabeled prompts → LoRA weights via text encoder +
  hyper-convolutional decoder. Claims "up to 12,000× lower overhead than full fine-tuning"
  and "gains up to 30%" — self-reported, single preprint, "up to" qualifiers, not replicated.
- **LoRA Diffusion** (2412.02352), **DiffLoRA** (2408.06740), **LatentSkill** (2606.06087).
- **Transformer²** (Sakana, 2501.06252): SVD-based singular-value fine-tuning, z-vectors per
  skill trained with RL, two-pass inference.
- **SEAL** (MIT, NeurIPS 2025, 2506.10943): model generates its own finetuning data via an RL
  outer loop; no-context SQuAD 33.5% → 47.0%, few-shot ARC 20% → 72.5%. Each self-edit
  evaluation costs 30–45 s.

Relevance: Adaptive Minds' own future work is "training-as-a-tool" — commissioning a new
adapter when none matches. If that direction is ever pursued, T2L is the mechanism.

### 7f. Planning — empirical limits and algorithms

**Limits.** PlanBench (Valmeekam et al., NeurIPS 2023 D&B, arXiv:2206.10498): classical Fast
Downward 100%; best pre-o1 LLMs barely exceed ~5% on Mystery Blocksworld (obfuscated).
o1-preview: 97.8% standard Blocksworld but **52.8% Mystery Blocksworld**, degrading sharply
beyond ~20 steps (arXiv:2409.13373). NATURAL PLAN (Google, arXiv:2406.04520): Gemini 1.5 Pro
34.8%, GPT-4 31.1% on trip planning; "all models perform below 5% when there are 10 cities."

**Diagnosis** (Kambhampati et al., "LLMs Can't Plan, But Can Help Planning in LLM-Modulo
Frameworks," ICML 2024): autoregressive generation does approximate retrieval, not sound
state-tracking. The field's answer is LLM-as-generator inside verifier-guided loops.

**Algorithms worth knowing:** ReAct, Reflexion, Tree/Graph-of-Thought, Plan-and-Solve, ADaPT
(as-needed decomposition), LLM+P (NL → PDDL → classical planner), LLM-Modulo (external
verifiers), RAP (MCTS with LLM as world model), **LATS** (Zhou et al., ICML 2024,
arXiv:2310.04406 — MCTS unifying reasoning/acting/planning; 94.4% pass@1 HumanEval with
GPT-4; uses on average 3.55 fewer expanded nodes than RAP and 12.12 fewer than ToT).

**Test-time compute:** Snell et al. (arXiv:2408.03314, ICLR 2025) — compute-optimal per-prompt
allocation is >4× more efficient than best-of-N; PRM (process) vs ORM (outcome) reward models.

**Other harnesses:** ACPBench, AutoPlanBench, TravelPlanner, AgentBench, τ-bench/τ²-bench,
WebArena, OSWorld, GAIA, SWE-bench Verified, BFCL.

### 7g. Routing between models (adjacent, not the same problem)

**RouteLLM** (Ong et al., arXiv:2406.18665, ICLR 2025): routes between a strong and weak model
using Chatbot-Arena preference data; "cost savings of up to 3.66×" retaining 95% GPT-4 quality
on MT-Bench. **FrugalGPT** (TMLR 2024): cascades, up to 98% cost reduction. Also Hybrid-LLM,
Zooter, RouterBench, RouterArena. Router overhead: rule-based <1 ms, embedding ~5 ms, ML
classifier 50–100 ms.

**Keep the three-way taxonomy explicit in the paper** — reviewers conflate them:
routing between *models* (RouteLLM) / between *experts inside one model* (MoE) / between
*adapters on one base model* (this work).

### 7h. Known negative results — check against these

- **Routing collapse / load imbalance.** MoE routers collapse to a few experts. DeepSeek-V3's
  auxiliary-loss-free bias (per-expert bias added only to the top-k selection score, updated
  γ=0.001/step) is the current best mitigation; a 2025 study finds it insufficient alone under
  very high sparsity in lower layers.
- **Cross-adapter merge degradation.** On LLMs only task arithmetic consistently gives
  constructive gains, typically <1% average; TIES can degrade; DARE fails when merging many
  models. Rank and sign interference are real.
- **Zero-shot routing has no free lunch.** Arrow/PHATGOOSE depend on adapter directions being
  separable in activation/SVD space. On MBC-clustered libraries "routing is less important" —
  meaning part of the reported gain comes from the clustering, not the routing.
- **The interruption problem** (from options theory): free switching causes thrashing, and
  thrashing here has a hard systems cost.

---

## 8. Staged plan

### Stage 1 — Baseline (weeks 1–3)
Reproduce Adaptive Minds v2 as the one-shot-routing lower bound. Run its 30-adapter router
and Table 2 specialists; confirm the <5% placement-latency claim and ~98% routing accuracy.
Establishes the "AM = K=1 special case of our planner" narrative.
**Also run the prompt-switching baseline here** — it is nearly free and it is the one that can
invalidate the project.

*Threshold:* reproduce routing accuracy within ±3 pp → treat AM as subsumed. If not, the
routing layer itself becomes a contribution.

### Stage 2 — Benchmark (weeks 3–8) ← highest leverage
Construct tasks that **provably require** sequential use of ≥2 distinct adapters — e.g.
extract → translate → format pipelines, or SQL-generation → PII-redaction → summarization
chains. Provide gold adapter-sequences, a completion checker, and per-step serving-cost
logging. Vary plan length and domain-cue obfuscation the way PlanBench does.

*Kill criterion:* if no task family can be built where multi-adapter chaining beats the best
single adapter by a meaningful margin, the multi-step premise is weak — pivot to
serving-scheduling as the lead contribution.

### Stage 3 — Planner + scheduler (weeks 8–18)
Planner emits an adapter *plan*; scheduler turns the plan into merge/unmerge/prefetch/evict
operations on a dLoRA- or vLLM-style backend. Formalize each option as
`(adapter, initiation condition, sub-policy, termination)` with transition costs = measured
serving latencies (borrow dLoRA's credit-based accounting, V-LoRA's measured switch costs).

Run the **small planner vs. self-routing** ablation here — Stage 1 already provides the
self-routing arm for free.

---

## 9. Implementation notes

**Stack:** vLLM (`--enable-lora`, tune `max_loras`, `max_lora_rank`, `max_cpu_loras`) as the
batching backbone; a dLoRA-style merge/unmerge controller for skewed workloads; HF PEFT for
composition experiments. Planning harnesses: PlanBench (`karthikv792/LLMs-Planning`),
NATURAL PLAN (`google-deepmind/natural-plan`), LM-Eval-Harness, the LATS repo.

**Hardware:** a single A100/H100 80 GB suffices for multi-LoRA serving of thousands of
small-rank adapters. Adaptive Minds ran on one L40S 48 GB.

**Real engineering bottlenecks:**
1. Heterogeneous-rank batching — SGMV/BGMV pad to max rank; S-LoRA's MBGMV avoids padding.
   Choose accordingly, or just fix the rank across the library.
2. Adapter cache eviction — vLLM LRU over `max_cpu_loras`; cold start mitigated by dummy-request
   preloading. **This is where your scheduling contribution lives.**
3. Merge/unmerge overhead — dLoRA's credit-based decision is the reference design.
4. X-LoRA-style dual forward pass doubles the KV cache.
5. DoRA/AdaLoRA break fixed-rank kernels.

---

## 10. Working preferences

- Technical and precise; skip introductory framing of LoRA or agents.
- Flag self-reported "up to" benchmark numbers as such — most headline multipliers in this
  literature are best-case, single-workload, single-seed.
- Note when English phrasing would read wrong to reviewers.

---

## 11. Open questions carried forward

1. **Where does the termination signal come from?** The hardest problem, and the one the
   options framing names. Currently unanswered.
2. **Should adapters have learned initiation sets (preconditions)?** Possibly a contribution
   in its own right.
3. **With RL dropped, what trains the planner?** Supervised imitation on gold plans, or
   prompted planning with no training. Decide before building the benchmark — it determines
   whether gold plans are required.
4. **Does adapter-direction separability hold as library size N grows?** Relevant if the
   planner ever uses SVD-style signals rather than metadata. Flagged earlier as a thread worth
   its own investigation; lower priority now that adapter-internals research is out of scope.
5. **How many adapters before routing degrades?** Adaptive Minds tested 30 and explicitly
   flags scale as open.
