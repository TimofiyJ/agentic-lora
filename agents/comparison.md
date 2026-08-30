# Evaluation Framework & Benchmark Design

## The Core Claim

The primary claim is **accuracy and task quality on multi-step cross-domain tasks**, not efficiency. Resource gains (GPU memory, latency) are a side-effect, not the headline.

## The Fairness Principle

The system's LoRA adapters contain domain knowledge baked into weights. Any baseline must access **equivalent knowledge** via a different mechanism — not less knowledge. Comparing against a raw base LLM would be unfair.

**This system** = `Base LLM + adapter pool (domain-specialized LoRA weights) + dedicated planner`

---

## Baselines (same knowledge budget, different mechanism)

| Baseline | Knowledge source | What varies | What it proves |
|---|---|---|---|
| **Self-routing (Adaptive Minds)** | Same adapter pool | Base LLM selects adapters itself; no dedicated planner | Value of a dedicated small planner over self-routing |
| **Parametric RAG (PRAG)** | Same domain corpora, encoded into base weights (full fine-tune or all-adapter merge) | No dynamic per-step routing; all knowledge statically present | Whether dynamic routing beats statically-merged parametric knowledge |
| **Standard RAG** | Same domain corpora, retrieved at query time into context | Knowledge in context vs. knowledge in weights | Whether adapter weights beat retrieval for specialized multi-step tasks |
| **Best-single-adapter** | One adapter selected once per task | Static selection, no per-step routing | Value of multi-step dynamic routing vs. static expert selection |
| **All-adapters merged** | Task arithmetic of all LoRA weights merged into base | No switching at all | Whether dynamic routing beats blending all domain knowledge together |
| **Oracle routing** | Same adapter pool; correct adapter always selected | Perfect planner, no routing errors | Upper bound — ceiling on what the planner can achieve |

The most critical comparison is **self-routing (Adaptive Minds)**: same base model, same adapters, same task — only the planning mechanism differs. This directly isolates the value of the dedicated planner.

The second most critical is **PRAG**: same domain knowledge, stored monolithically. Tests whether dynamic per-step specialization beats always-on parametric knowledge.

---

## What Each Comparison Argues

```
vs. self-routing (Adaptive Minds)  → dedicated planner outperforms self-routing on task quality
vs. Parametric RAG                 → dynamic per-step routing beats statically merged domain knowledge
vs. standard RAG                   → adapter weights beat retrieval for specialized multi-step tasks
vs. best-single-adapter            → multi-step routing beats static expert selection
vs. all-adapters-merged            → dynamic switching beats blending all knowledge together
vs. oracle                         → gap shows how close the planner gets to the theoretical ceiling
```

---

## Metrics

### Primary (task quality — these lead all reporting)

| Metric | What it measures | Notes |
|---|---|---|
| **Task completion rate** | Did the full multi-step task succeed end-to-end? | Headline metric |
| **Step accuracy** | Correct output at each individual step | Requires step-level ground truth |
| **Adapter selection accuracy** | Did the planner pick the right adapter per step? | Requires oracle ground-truth adapter sequences |
| **Error propagation rate** | How often does a wrong step corrupt all downstream steps? | Reveals compounding failures specific to multi-step tasks |

### Domain-Specific Quality (per adapter type)

| Adapter domain | Benchmark |
|---|---|
| Medical | MedQA, PubMedQA |
| Legal | LegalBench, CUAD |
| Code | HumanEval, MBPP |
| Finance | FinQA |
| Science | MMLU science subsets, SciQ |
| Translation | BLEU on WMT pairs |

### Secondary (resource efficiency — reported as bonus)

GPU-Time Efficiency, peak GPU memory, end-to-end latency — reported alongside quality numbers to show the resource side-effect, but not used as the primary argument.

---

## Benchmark Task Design Requirement

Tasks must be genuinely multi-step and cross-domain. A task solvable by one adapter alone lets best-single-adapter tie or win — those instances are invalid for the comparison. Tasks should require at least 2–3 distinct domain steps.

**Example valid tasks:**
- *"Read this medical paper → extract the statistical findings → write Python to plot them → translate the summary to Spanish"*
- *"Classify this legal clause → summarize it in plain English → estimate financial exposure → generate a code stub to flag it in a contract parser"*

---

## Benchmark Instance Format

A single benchmark instance (JSON):

```json
{
  "id": "bench_001",
  "query": "I have this abstract from a clinical trial. Extract the key numerical findings, write Python to plot treatment vs control results, then summarize the findings in Spanish.",
  "context": {
    "abstract": "A randomized controlled trial of 240 patients evaluated drug X for hypertension. The treatment group (n=120) achieved a mean systolic reduction of 18.3 mmHg (SD 4.1, p<0.001) versus 6.2 mmHg (SD 3.8) in the control group (n=120) over 12 weeks. Secondary endpoints: diastolic reduction 11.2 vs 4.3 mmHg; adverse events 8.3% vs 7.1%."
  },
  "steps": [
    {
      "id": 1,
      "description": "Extract key numerical findings from the abstract",
      "oracle_adapter": "medical",
      "input": "{{context.abstract}}",
      "expected_output": {
        "findings": {
          "treatment_systolic": 18.3,
          "control_systolic": 6.2,
          "treatment_diastolic": 11.2,
          "control_diastolic": 4.3,
          "n_treatment": 120,
          "n_control": 120,
          "p_value": 0.001,
          "duration_weeks": 12
        }
      },
      "metric": "entity_recall",
      "metric_config": {
        "required_fields": ["treatment_systolic", "control_systolic", "treatment_diastolic", "control_diastolic"],
        "threshold": 0.8
      }
    },
    {
      "id": 2,
      "description": "Write Python to plot treatment vs control results",
      "oracle_adapter": "code",
      "input": "{{steps[0].output}}",
      "expected_output": {
        "code_must_contain": ["matplotlib", "bar", "systolic", "diastolic"],
        "must_execute": true,
        "must_produce_plot": true
      },
      "metric": "code_execution",
      "metric_config": {
        "timeout_s": 10,
        "required_imports": ["matplotlib"]
      }
    },
    {
      "id": 3,
      "description": "Summarize the findings in Spanish",
      "oracle_adapter": "multilingual",
      "input": "{{steps[0].output}}",
      "expected_output": {
        "reference": "El grupo de tratamiento logró una reducción sistólica media de 18,3 mmHg frente a 6,2 mmHg en el grupo de control durante 12 semanas (p<0,001). Las reducciones diastólicas fueron 11,2 frente a 4,3 mmHg.",
        "must_be_spanish": true,
        "must_contain_numbers": [18.3, 6.2, 11.2, 4.3]
      },
      "metric": "bleu_plus_number_recall",
      "metric_config": {
        "bleu_weight": 0.6,
        "number_recall_weight": 0.4
      }
    }
  ],
  "final_output": {
    "reference": "{{steps[2].expected_output.reference}}",
    "metric": "bleu_plus_number_recall"
  },
  "oracle_adapter_sequence": ["medical", "code", "multilingual"],
  "metadata": {
    "domains": ["medical", "code", "multilingual"],
    "n_steps": 3,
    "complexity": "medium",
    "requires_step_chaining": true,
    "solvable_by_single_adapter": false
  }
}
```

### How Each Baseline Is Scored Against This Instance

| System | What it sees | What's measured |
|---|---|---|
| **Your system** | `query` + `context` | Adapter sequence vs `oracle_adapter_sequence`, per-step metric, final BLEU |
| **Self-routing (Adaptive Minds)** | Same | Same — but LLM picks adapters; compare adapter sequence accuracy |
| **PRAG / merged model** | Same | No adapter sequence to score; only step metrics + final BLEU |
| **Standard RAG** | Same | Retrieves from same corpora at runtime; only step metrics + final BLEU |
| **Best-single-adapter** | Same | Forced to use one adapter for all steps; collapses on out-of-domain steps |
| **Oracle** | Same + forced `oracle_adapter_sequence` | Upper bound — best possible quality with correct routing |

### The Key Validity Filter

`"solvable_by_single_adapter": false` is the filter that makes an instance valid for this benchmark. Any instance where this is `true` must be excluded or kept only as a sanity check — those are the cases where best-single-adapter ties your system, making the routing contribution invisible.

---

## Dataset Composition (~20 instances)

```
5 instances  — 2-step tasks  (easy, one domain transition)
10 instances — 3-step tasks  (medium, the primary evaluation set)
5 instances  — 4-step tasks  (hard, tests replanning on failure)
```

No two instances should share the same adapter sequence — otherwise you are testing one routing pattern repeatedly rather than generalization across domain transitions.

**Domain pair coverage to target:**

| Step 1 | Step 2 | Step 3 |
|---|---|---|
| medical | code | — |
| medical | multilingual | — |
| legal | finance | — |
| science | code | multilingual |
| legal | code | multilingual |
| finance | legal | multilingual |
| medical | science | code |
| legal | medical | multilingual |
