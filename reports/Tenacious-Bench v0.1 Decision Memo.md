# Tenacious-Bench v0.1 Decision Memo  

**Project:** The Conversion Engine Ground Truth  
**Path:** B — Preference-Tuned Judge  
**Author:** Nuhamin Alemayehu  
**Date:** 2026-05-02  

---

## Page 1 — The Decision

### Three-Sentence Executive Summary

I built **Tenacious-Bench v0.1**, a 260-task B2B sales-agent benchmark with 130 preference pairs, then trained a lightweight LoRA/DPO judge to catch Tenacious-specific output failures before they reach prospects. On the sealed held-out set of 26 pairs / 52 pointwise examples, the fine-tuned DPO judge improved strict pairwise accuracy from **61.54%** for the Week 10 no-judge baseline to **96.15%**, a **+34.62 percentage-point lift** with 95% CI **[+15.38 pp, +53.85 pp]** and one-sided p = **0.0013**. Recommendation: **deploy with caveat, in shadow mode only**, because the judge is strong on v0.1 but the held-out set is small, public-signal ground truth is lossy, and one held-out pair still failed.

### What Was Built

The Week 10 failure was an inconsistency problem: the agent could often identify prospect intent but could not reliably tell when its own final response was incomplete or unsafe. I therefore chose Path B: a preference-tuned judge/critic rather than a generator fine-tune. The benchmark was built from four source modes and then converted into DPO preference pairs:

| Source mode | Candidate tasks | Selected tasks | Held-out tasks |
|---|---:|---:|---:|
| Trace-derived | 176 | 78 | 22 |
| Programmatic | 102 | 78 | 14 |
| Multi-LLM synthesis | 96 | 64 | 6 |
| Hand-authored adversarial | 62 | 40 | 10 |
| **Total** | **436** | **260** | **52** |

The final split is exactly balanced: **130 train / 78 dev / 52 held-out tasks**, with 65/39/26 complete pairs respectively and zero broken pairs. Each split has equal good/bad labels. :contentReference[oaicite:1]{index=1}

### Training Summary

The judge used `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` as the base model and was trained as a LoRA adapter with DPO. Training used **65 train pairs** and **39 validation pairs** with `prompt`, `chosen`, and `rejected` fields. The run completed with final train loss **0.6785**, eval loss **0.6711**, eval reward accuracy **94.87%**, and positive eval reward margin **0.0451**. :contentReference[oaicite:2]{index=2}

| Training item | Value |
|---|---:|
| Base model | `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` |
| Train pairs | 65 |
| Validation pairs | 39 |
| Train loss | 0.6785 |
| Eval loss | 0.6711 |
| Eval reward accuracy | 94.87% |
| Runtime | 122.17 seconds |
| Direct training compute cost | $0 on Colab T4 |

### Held-Out Evaluation

Primary metric: **strict pairwise accuracy**. A pair is counted correct only when the system accepts the good output and rejects the bad output.

| System | Accuracy | Precision | Recall | F1 | Strict pairwise accuracy |
|---|---:|---:|---:|---:|---:|
| Week 10 baseline | 80.77% | 94.44% | 65.38% | 77.27% | 61.54% |
| Prompted judge | 88.46% | 100.00% | 76.92% | 86.96% | 76.92% |
| Fine-tuned DPO judge | **96.15%** | 96.15% | **96.15%** | **96.15%** | **96.15%** |

The fine-tuned DPO judge correctly handled **25/26** held-out pairs. Its mean reward margin was **0.0615**, but the minimum margin was **-0.01875**, which corresponds to the single unresolved held-out failure. :contentReference[oaicite:3]{index=3}

### Delta A — Trained Judge vs. Week 10 Baseline

| Metric | Baseline | Fine-tuned DPO judge | Delta |
|---|---:|---:|---:|
| Strict pairwise accuracy | 61.54% | 96.15% | **+34.62 pp** |

Paired bootstrap result:

- Observed delta: **+0.3462**
- 95% CI: **[+0.1538, +0.5385]**
- One-sided p-value for delta ≤ 0: **0.0013**
- Interpretation: positive with CI separation. :contentReference[oaicite:4]{index=4}

### Delta B — Trained Judge vs. Prompted Judge

| Metric | Prompted judge | Fine-tuned DPO judge | Delta |
|---|---:|---:|---:|
| Strict pairwise accuracy | 76.92% | 96.15% | **+19.23 pp** |

Paired bootstrap result:

- Observed delta: **+0.1923**
- 95% CI: **[+0.0385, +0.3462]**
- One-sided p-value for delta ≤ 0: **0.0039**
- Interpretation: prompting helped, but fine-tuning produced better deployment-style calibration. :contentReference[oaicite:5]{index=5}

### Cost Reporting

The challenge requires cost per task with and without the trained component, and cost discipline is explicitly graded. :contentReference[oaicite:6]{index=6}

Measured project costs:

| Cost bucket | Measured cost | Unit calculation |
|---|---:|---|
| Dataset authoring via OpenRouter/Qwen | **$0.2051** | 51 calls, 804,998 total tokens |
| Average OpenRouter call | **$0.0040** | $0.2051 / 51 calls |
| Generated synthetic task cost | **$0.0021** | $0.2051 / 96 synthetic tasks |
| Selected benchmark task amortized cost | **$0.0008** | $0.2051 / 260 selected tasks |
| Training compute | **$0.00** | Free Colab T4 |
| Held-out evaluation direct API cost | **$0.00** | local/Colab model inference, no eval-tier API |
| Baseline direct evaluation cost | **$0.00** | deterministic baseline |

Cost per held-out task in this experiment:

| Condition | Direct API cost/task | Notes |
|---|---:|---|
| Week 10 no-judge baseline | $0.0000 | deterministic script |
| Prompted judge | $0.0000 | local/Colab inference; no API |
| Fine-tuned DPO judge | $0.0000 | local/Colab inference; training was free |

Important caveat: these are **measured experiment costs**, not production hosting costs. Production deployment still needs p50/p95 latency and GPU-dollar cost measured under the actual serving stack.

### Production Recommendation

**Recommendation: deploy with caveat, shadow mode only.**

Quantitative basis:

- Fine-tuned judge improved strict pairwise accuracy to **96.15%** on held-out.
- It still made **1 error out of 26 pairs**.
- It implies one false allow-through and one false block on the 52 pointwise examples.
- Held-out size is small, so the point estimate is promising but not sufficient for autonomous blocking.

Shadow-mode entry criteria:

1. Run judge on live candidate outputs without blocking.
2. Human-review at least **500 live judged outputs**.
3. Move to blocking only if:
   - high-severity bad-output allow-through is **≤2%**,
   - false block rate on human-approved outputs is **≤5%**,
   - p95 added latency is **≤1.5 seconds**,
   - marginal scoring cost is **≤$0.005/output**.

---

## Page 2 — Skeptic’s Appendix

### Four Benchmark Coverage Gaps

1. **Sparse failure-code tails.**  
   The candidate pool has strong coverage for some failures, but several failure codes are thin: F2.3 has only 5 candidate tasks, F1.4 has 6, F4.1 has 6, and F4.3/F1.1 have 8 each. These slices are too small to support confident per-failure claims. :contentReference[oaicite:7]{index=7}  
   **v0.2 fix:** add at least 20 examples per failure code and report per-slice accuracy.

2. **Held-out multi-LLM synthesis is underrepresented.**  
   Multi-LLM synthesis contributes 64 selected tasks overall but only 6 held-out tasks. This makes the held-out estimate less sensitive to synthetic-style artifacts. :contentReference[oaicite:8]{index=8}  
   **v0.2 fix:** stratify the held-out set so each source mode has at least 20 held-out tasks.

3. **No full live multi-turn state coverage.**  
   v0.1 evaluates candidate outputs, not full production trajectories across email, CRM, calendar, and follow-up state. A judge can score a response but may not catch multi-step workflow drift.  
   **v0.2 fix:** add trajectory tasks where the ground truth includes prior state, tool output, and expected next action.

4. **No measured production latency or serving cost.**  
   The experiment used Colab/local inference. That is sufficient for Week 11 evidence but not enough for production SLOs. The challenge asks for cost/latency reporting with and without the trained component. :contentReference[oaicite:9]{index=9}  
   **v0.2 fix:** run the same 52 held-out examples through the intended serving path and report p50/p95 latency, GPU memory, and cost per scored output.

### Ground-Truth Faithfulness and Public-Signal Lossiness

Tenacious-Bench v0.1 is faithful to the Week 10 failure structure but not a perfect representation of production truth.

What is faithful:

- Trace-derived tasks anchor the benchmark to observed Week 10 behavior.
- Programmatic tasks systematically test known failure modes.
- Adversarial tasks encode failure modes that generic sales benchmarks miss.
- Every selected pair is complete and label-balanced across train/dev/held-out. :contentReference[oaicite:10]{index=10}

What is lossy:

- Public company signals are simplified into task fields; they do not preserve the full uncertainty of Crunchbase, layoffs, job-post velocity, or leadership-change data.
- Some “good” responses are corrected or synthetic preference targets, not unique real-world ground truth.
- The labels are comparative: `chosen` is better than `rejected`, not necessarily the only correct prospect-facing response.
- Formal contamination checks and independent inter-rater agreement should be expanded. The challenge specifically calls for n-gram overlap, embedding similarity, time-shift verification, and a delayed 30-task hand-labeling agreement check. :contentReference[oaicite:11]{index=11}

### One Honest Unresolved Training Failure

The fine-tuned judge did not achieve perfect held-out performance. It got **25/26** pairs correct and had a minimum reward margin of **-0.01875**, meaning at least one held-out pair was ranked in the wrong direction. :contentReference[oaicite:12]{index=12}

Interpretation:

- The model learned the overall preference signal.
- The decision boundary is still fragile for at least one adversarial or low-margin pair.
- The current data volume is too small to know whether that failure is isolated or a repeated blind spot.

v0.2 action:

1. Inspect the failed pair in `fine_tuned_judge_pair_scores.jsonl`.
2. Add 10–20 near-neighbor adversarial tasks around that failure.
3. Rerun Delta A/B with a larger held-out set.
4. Tune the rejection threshold against dev, not held-out.

### Statistical Rigor Caveat

The ablation result is statistically positive, but the held-out size is small:

- held-out pairs: **26**
- pointwise examples: **52**
- fine-tuned judge errors: **2 pointwise / 1 pair**
- Delta A CI: **[+15.38 pp, +53.85 pp]**
- Delta B CI: **[+3.85 pp, +34.62 pp]**

This supports a Week 11 result and shadow-mode recommendation, but not a full production launch.

### Kill-Switch Clause

The trained judge should be paused or rolled back if any of the following are observed during shadow mode or controlled rollout:

1. **Safety allow-through trigger:** more than **1 high-severity bad output per 50 human-reviewed allowed outputs**.
2. **Over-blocking trigger:** more than **10%** of human-approved outputs are rejected by the judge in a weekly review sample.
3. **Latency trigger:** p95 added latency exceeds **1.5 seconds** for 24 hours.
4. **Cost trigger:** marginal inference cost exceeds **$0.005 per scored output**.
5. **Coverage trigger:** more than **20%** of live failures fall outside the current Tenacious-Bench taxonomy.

### Final Decision

The trained DPO judge should not yet be an autonomous production blocker. It should be deployed first as a shadow-mode critic and measurement layer. If it maintains the observed held-out gains under 500+ live reviewed outputs while staying within the cost and latency thresholds above, it becomes a credible candidate for a controlled blocking rollout.