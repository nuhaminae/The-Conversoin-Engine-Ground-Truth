# Tenacious-Bench Judge v0.1 Model Card

## Model Overview

**Model name:** Tenacious-Bench Judge v0.1  
**Model type:** Preference-tuned causal language model judge  
**Training method:** Direct Preference Optimization (DPO) with LoRA  
**Base model:** `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`  
**Adapter output:** `models/judge`  
**Checkpoint directory:** `models/checkpoints`  
**Project path:** Path B — Preference-Tuned Judge  

This model is a lightweight judge/critic trained to prefer higher-quality Tenacious-style sales-agent outputs over lower-quality outputs. It is designed as a quality-control layer for the Week 10 Conversion Engine, not as a standalone outreach generator.

## Intended Use

The intended use is to score or rank candidate outputs from the Tenacious Conversion Engine before they are sent to a prospect.

The judge should prefer outputs that are:

- Concise and action-oriented.
- Professional and non-condescending.
- Grounded in the supplied prospect/company signal.
- Honest about uncertainty.
- Aligned with Tenacious voice and sales process.
- Safe around tool failures, secrets, prompt injection, and inappropriate requests.
- Clear about the next step when the prospect shows interest.

The judge should reject outputs that:

- Miss a clear meeting intent.
- Fail to include the booking link when appropriate.
- Leak internal prompts, API keys, or hidden configuration.
- Use placeholder, localhost, or broken links.
- Ignore layoffs or contradictory company signals.
- Invent pricing, capacity, or unsupported claims.
- Drift into generic marketing filler.
- Continue selling after a prospect opts out.

## Motivation

The Week 10 Conversion Engine showed an inconsistency failure: it sometimes correctly identified a prospect’s intent as `INTERESTED_BOOK_MEETING` but then failed to complete the correct action, such as sending a booking link or gracefully handling a tool failure.

Because the generator was sometimes capable of producing the correct response, the Week 11 strategy was to train a preference-tuned judge rather than immediately retraining the generator. The judge is intended to detect low-quality outputs before they reach a prospect.

## Training Data

The model was trained on DPO preference pairs generated from Tenacious-Bench v0.1.

Training input files:

- `data/training_data/preferences_train.jsonl`
- `data/training_data/preferences_dev.jsonl`

Dataset columns:

- `prompt`
- `chosen`
- `rejected`

Training split size:

- Train: 65 preference pairs
- Dev/validation: 39 preference pairs

The full benchmark was derived from four data-authoring modes:

1. Trace-derived Week 10 examples.
2. Programmatic rule-based examples.
3. Synthetic multi-LLM / model-assisted examples.
4. Hand-authored adversarial examples from the probe library.

## Training Configuration

Training was run on Google Colab using a Tesla T4 GPU.

Core configuration:

- Base model: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- Training method: DPO
- Adapter method: LoRA
- Trainable parameters: 11,272,192
- Total parameters: 1,247,086,592
- Trainable percentage: approximately 0.90%
- Epochs: 1
- Batch size per device: 1
- Gradient accumulation steps: 4
- Total effective batch size: 4
- Total training steps: 17
- Optimizer: `adamw_8bit`
- Precision: fp16 on T4
- Gradient checkpointing: Unsloth gradient checkpointing
- Seed: 42

Training artifacts:

- `reports/training/training_config_used.yaml`
- `reports/training/dataset_summary.json`
- `reports/training/training_run.log`
- `reports/training/training_summary.json`

## Training Results

The run completed successfully.

Final training metrics:

- Train loss: 0.6785
- Train runtime: 46.64 seconds
- Train samples per second: 1.394
- Train steps per second: 0.364
- Epochs completed: 1.0

Final validation metrics:

- Eval loss: 0.6711
- Eval reward accuracy: 0.9487
- Eval chosen reward: 0.0340
- Eval rejected reward: -0.0111
- Eval reward margin: 0.0451
- Eval runtime: 6.28 seconds

Interpretation:

The model learned to prefer chosen outputs over rejected outputs on the dev split. The positive reward margin and high reward accuracy indicate that the DPO training objective worked on the available preference-pair data.

## Limitations

This is an early v0.1 judge model.

Known limitations:

1. The training set is small, with 65 train pairs and 39 dev pairs.
2. Dev reward accuracy is not the same as final held-out product performance.
3. The model is a LoRA adapter for a causal language model, not a sequence-classification model.
4. Evaluation scripts must load the base causal LM plus LoRA adapter and compare chosen vs. rejected log-probabilities.
5. The model has not yet been validated against the sealed held-out split.
6. The judge should not be used as the sole authority for high-stakes or irreversible business actions.
7. The model is tuned to the Tenacious sales-agent context and may not generalize to unrelated domains.

## Safety and Guardrails

The judge was trained to penalize outputs that:

- Reveal system prompts or internal instructions.
- Reveal API keys or secrets.
- Follow prompt-injection instructions.
- Generate hostile competitor messaging.
- Continue selling after an opt-out.
- Use broken, placeholder, or localhost links.
- Make unsupported pricing or delivery-capacity claims.

The judge should be used as a safety and quality filter, not as a replacement for system-level guardrails.

## Recommended Evaluation Before Deployment

Before using this model as an integration layer, run:

1. **Delta A:** Week 10 baseline agent vs. Week 10 agent with trained judge.
2. **Delta B:** Trained judge vs. prompt-engineered judge.
3. **Held-out evaluation:** Final measurement on the sealed held-out set.
4. **Failure-mode breakdown:** Performance by failure category: F1, F2, F3, F4.
5. **Cost/latency analysis:** Added latency and cost per judged output.

## Usage Notes

This model should be loaded as:

1. The original base causal language model.
2. The trained LoRA adapter from `models/judge`.

It should not be loaded with `AutoModelForSequenceClassification` unless a separate classifier head is trained.

For pairwise preference evaluation, score both candidate responses under the same prompt and compare the model’s preference/log-probability signal.

## Version

**Version:** v0.1  
**Date:** 2026-05-02  
**Author:** Nuhamin Alemayehu  
**Project:** The Conversion Engine Ground Truth / Tenacious-Bench  
