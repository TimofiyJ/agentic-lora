# Research Results, Problems & Theories

Running log of empirical findings, blockers, and hypotheses as the system is built and tested.

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

## Experiments

### E1 — Doc-to-LoRA: Training Purpose-Built Adapters from Document Corpora
**Status:** Planned

Train 5-6 LoRA adapters from scratch on domain-specific document collections, giving the system a controlled adapter pool with unambiguous domain boundaries. Motivated by P1 (vLLM compatibility) and P2 (sparse public adapters) — both problems disappear when we train our own.

**Proposed domains:** Medical (PubMed), Legal (FreeLaw), Code (GitHub), Finance (SEC filings), Science (arXiv), Multilingual (OPUS translation pairs).

**Training stack:** PEFT + trl `SFTTrainer` on Qwen2.5-3B-Instruct. LoRA target modules restricted to `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` to guarantee vLLM compatibility. Rank 16-64 depending on domain complexity.

**What to measure:**
- Domain accuracy on held-out test sets (does the medical adapter outperform base model on medical Q&A?)
- Routing clarity: can the planner reliably pick the correct adapter from description alone?
- Weight space structure: do Doc-to-LoRA adapters cluster by domain in RQ3 experiments?

---

## Theories

### T1 — Scheduling Story May Shift to Memory Efficiency
The thesis plan flags this as a key decision point: if adapter load/unload latency is negligible (<10ms), the RQ1 narrative should shift from latency optimisation to GPU memory efficiency. This has not yet been measured. The GPU-Time Efficiency metric defined in the plan is well-suited to capture this regardless of which dimension dominates.

### T2 — Plan Regeneration Frequency Will Be the Core Tradeoff of RQ1
The continuous planner can regenerate the task plan after every step, but doing so adds an LLM call on the critical path. The interesting research question is not whether to replan, but *when* — triggered by low-confidence output, unexpected format, or just always. The naive orchestrator (fixed plan, no replanning) will be the baseline that makes this tradeoff measurable.

### T3 — Parallel Step Execution Requires Careful Adapter Slot Management
If the plan contains independent steps that can run in parallel, they may require different adapters to be resident simultaneously. With `--max-loras 4`, a plan with 4 parallel steps using 4 different adapters saturates the GPU slot budget. The scheduler must either execute parallel steps within the slot budget or serialise them — and that choice affects latency in a way the naive orchestrator won't capture.
