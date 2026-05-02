# From Missed Meetings to Reliable Sales Agents: Building Tenacious-Bench and a DPO Judge

## Introduction

In Week 10, the Conversion Engine reached the point where it could generate sales-agent responses from prospect signals and replies. But the demo revealed a more subtle problem than simple language quality.

The agent was not always incapable. It was inconsistent.

A prospect could clearly say, “Yes, send me times,” and the system might correctly identify the intent but still fail to send the booking link. In a sales workflow, that is not a small wording issue. It is a pipeline risk: a warm prospect expresses interest, and the automation fails to complete the next action.

Generic evaluation benchmarks do not capture this failure well. They may test broad helpfulness, retail tool use, or general instruction following, but they do not measure Tenacious-specific B2B sales behavior: booking intent, wrong-person replies, opt-outs, public hiring signals, tool failures, prompt injection, and commercial-claim discipline.

So in Week 11, I built **Tenacious-Bench v0.1**, a domain-specific benchmark for the Conversion Engine, and trained a lightweight **preference-tuned judge** to act as a quality gate.

## The Core Problem

The Week 10 failure was an inconsistency failure:

> The agent sometimes knew the prospect’s intent, but did not reliably complete the correct response.

Examples of high-impact failures included:

- A prospect agrees to a meeting, but the response does not include a booking link.
- A CRM/tool lookup fails, and the system drops the warm lead.
- A prospect opts out, but the agent keeps selling.
- A company shows layoff or weak hiring signals, but the response pitches aggressive growth.
- A prospect asks for internal prompts or API keys, and the output fails to safely refuse.
- The system includes placeholder, localhost, or broken-looking links.

This suggested a Path B strategy: train a judge/critic that can inspect a candidate response and reject or rank weak outputs before they reach a prospect.

## Building Tenacious-Bench v0.1

The benchmark was constructed from a 436-task candidate pool and selected into a 260-task final benchmark.

The final split is:

| Split | Tasks | Pairs |
|---|---:|---:|
| Train | 130 | 65 |
| Dev | 78 | 39 |
| Held-out | 52 | 26 |
| **Total** | **260** | **130** |

Each pair contains:

- one good/chosen response
- one bad/rejected response

The dataset is balanced across labels in every split.

## Four Authoring Modes

To avoid building a benchmark that only reflects one source of examples, I used four authoring modes:

| Source mode | Selected tasks |
|---|---:|
| Trace-derived | 78 |
| Programmatic | 78 |
| Multi-LLM synthesis | 64 |
| Hand-authored adversarial | 40 |

### Trace-derived examples

These came from Week 10 behavior and represent the highest-fidelity source. They capture real failures observed in the Conversion Engine.

### Programmatic examples

These were generated through controlled sweeps over failure modes: booking intent, pricing, capacity claims, weak hiring signals, opt-outs, and tool failures.

### Multi-LLM synthesis

Synthetic pairs were generated using model-assisted generation, including OpenRouter/Qwen for diversity. These examples helped expand the range of prospect phrasing and response styles.

### Hand-authored adversarial cases

These targeted edge cases: prompt injection, secret leakage, placeholder links, hostile competitor messaging, and other failure modes a generic judge may miss.

## Training the Judge

The judge was trained using Direct Preference Optimization (DPO). Instead of training a sequence classifier, I trained a causal language model adapter to prefer chosen outputs over rejected outputs.

Training setup:

| Item | Value |
|---|---|
| Base model | `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` |
| Method | DPO |
| Adapter | LoRA / PEFT |
| Runtime | Google Colab T4 |
| Train pairs | 65 |
| Dev pairs | 39 |
| Optimizer | `adamw_8bit` |
| Epochs | 1 |

The final training results were:

| Metric | Value |
|---|---:|
| Train loss | 0.6785 |
| Eval loss | 0.6711 |
| Eval reward accuracy | 94.87% |
| Eval reward margin | 0.0451 |

This showed that the model learned a preference signal: it generally assigned higher reward to chosen responses than rejected responses.

## Evaluation Design

I evaluated three systems on the same held-out benchmark:

1. **Week 10 baseline** — a deterministic no-judge approximation.
2. **Prompt-engineered judge** — the base model prompted to classify outputs as good or bad.
3. **Fine-tuned DPO judge** — the trained LoRA adapter evaluated with DPO-style reward scoring.

The primary metric was **strict pairwise accuracy**:

> A pair is correct only if the system accepts the good output and rejects the bad output.

This is stricter than pure ranking and better reflects the deployment use case.

## Results

| System | Accuracy | Precision | Recall | F1 | Strict Pairwise Accuracy |
|---|---:|---:|---:|---:|---:|
| Week 10 baseline | 80.77% | 94.44% | 65.38% | 77.27% | 61.54% |
| Prompt-engineered judge | 88.46% | 100.00% | 76.92% | 86.96% | 76.92% |
| Fine-tuned DPO judge | **96.15%** | 96.15% | **96.15%** | **96.15%** | **96.15%** |

## Delta A: Fine-Tuned Judge vs. Week 10 Baseline

The fine-tuned DPO judge improved strict pairwise accuracy from 61.54% to 96.15%.

That is a lift of:

```text
+34.62 percentage points
````

The baseline was conservative. It caught most bad outputs but rejected too many good ones. The DPO judge kept strong rejection behavior while dramatically improving recall.

## Delta B: Fine-Tuned Judge vs. Prompted Judge

Prompting alone helped. The prompt-engineered judge reached 88.46% accuracy and 100% precision.

But it was too conservative as a deployment gate. It rejected all bad outputs, but also rejected six good outputs.

The fine-tuned DPO judge improved strict pairwise accuracy from 76.92% to 96.15%.

That is a lift of:

```text
+19.23 percentage points
```

This matters because the goal is not only to rank outputs. A practical judge must be calibrated enough to accept good responses and reject bad ones.

## What Worked

Three choices made the project effective:

1. **Preference pairs instead of isolated labels**
   The task naturally fits pairwise judgment: given the same prospect context, which response is better?

2. **Domain-specific failure modes**
   Tenacious-Bench tests sales-specific action completion, not just generic helpfulness.

3. **Ablation against prompt engineering**
   The prompt-only judge was strong, but DPO tuning showed clear calibration gains.

## Limitations

This is a v0.1 benchmark and model.

Limitations include:

- The held-out set is small: 52 pointwise examples / 26 pairs.
- The benchmark is English-only.
- The dataset is specific to Tenacious-style B2B sales workflows.
- The baseline is an approximation of the Week 10 no-judge system.
- The judge was evaluated offline, not in live production.
- Inter-rater validation should be expanded in the next version.

## Next Steps

The next version should add:

- More trace-derived examples.
- A larger sealed held-out set.
- Stronger contamination checks.
- Human inter-rater agreement logs.
- Multi-turn reply-chain evaluation.
- Cost and latency dashboards.
- Live integration testing with rollback behavior.

## Conclusion

Tenacious-Bench v0.1 shows that a small, domain-specific benchmark can produce a useful judge for a real product failure.

The key result is not just that the DPO judge scored well. It is that the comparison tells a clear story:

- The Week 10 baseline was not reliable enough.
- Prompt engineering improved the judge but remained too conservative.
- Fine-tuning produced the best deployment-style quality gate.

For this project, the judge is a practical bridge between a promising sales agent and a more reliable production workflow.

