# Methodology for Tenacious-Bench v0.1

This document outlines the methodology for creating and utilising `Tenacious-Bench v0.1`, a benchmark designed to evaluate the Tenacious Conversion Engine.

## 1. Path Declaration

For this challenge, we are officially committing to **Path B: Preference-Tuned Judge**.

## 2. Justification for Path B

The decision to build a preference-tuned judge is based on a critical failure mode observed in the Week 10 Conversion Engine: **inconsistency**. The agent demonstrated that it is *capable* of producing the correct output but frequently fails to do so, and more importantly, does not recognise its own errors.

**Core Problem:** The agent suffers from **Action and Reasoning Failures (F2 & F3)**. It can correctly perceive a prospect's intent but then fail to execute the corresponding action due to unhandled errors or flawed logic.

**Evidence of Inconsistency:**

- **Failure Case (`traceId: a4c80bf5655f83320fca0e7a794e384f`):** In this trace, the prospect replied with clear buying intent. The agent correctly classified this intent as `INTERESTED_BOOK_MEETING`. However, a subsequent lookup in our CRM failed, and the system errored out with "Contact not found in HubSpot." The agent did not handle this failure and never sent the crucial booking link, effectively dropping a warm lead.
- **Success Case (`traceId: d5c0035384a396442e00c56769f0bc4a`):** In this trace, faced with a nearly identical input, the agent performed perfectly. It classified the intent as `INTERESTED_BOOK_MEETING` and successfully generated a correct response that included the booking link.

This inconsistency proves that simply retraining the generator model (as in Path A) is an inefficient approach. The agent *already knows how* to generate the correct response. The problem is that its internal logic and error handling are not robust, causing it to fail silently.

Therefore, the most direct and effective solution is to build a "judge" model. This judge will act as a safety net, scoring the generator's final output against a learned preference for "good" Tenacious-style responses. It can then be used to reject or flag low-quality outputs before they ever reach a prospect, directly addressing the inconsistency problem.

## 3. Dataset Partitioning and Contamination Protocol

To ensure a rigorous evaluation, `Tenacious-Bench` will be partitioned as follows:

- **Training Set (50%):** Used exclusively for training the preference model.
- **Public Dev Set (30%):** Used for iteration, hyperparameter tuning, and public leaderboard scores.
- **Sealed Held-out Set (20%):** A private, final evaluation set used only once to measure the true performance of the trained judge.

To prevent data contamination between the training and held-out sets, we will implement three specific checks:

1. **N-gram Overlap:** We will calculate n-gram overlap scores between the datasets to ensure there are no verbatim or near-verbatim copies of tasks.
2. **Embedding Similarity:** Using a cost-effective embedding model, we will check for high semantic similarity between task descriptions in the training and held-out sets.
3. **Time-Shift Verification:** We will confirm that any real-world data used to generate tasks (e.g., from news articles or company reports) was published *before* the project's start date to prevent a model from having "seen the answers".

A full contamination report will be committed before training commences.

## 4. Inter-Rater Agreement (IRA)

To ensure the rubric is objective and consistent, a subset of 30 tasks will be hand-labeled by a single evaluator twice, with a 24-hour gap. If the Cohen's Kappa score for any rubric dimension is below 0.80, the rubric will be revised and the process repeated until the target agreement is met. The final agreement matrix will be published.
