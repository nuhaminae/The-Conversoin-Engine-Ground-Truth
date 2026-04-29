Thanks for clarifying — I’ve gone back through the **Week 11 challenge document** carefully. The key is that Week 11 doesn’t start from scratch: it explicitly **imports Week 10’s artifacts** as inputs to build the evaluation bench. Here’s how they slot into your Week 11 repo and pipeline:

---

### 📂 Where to Put Week 10 Artifacts in Week 11 Repo

- **`data/raw/trace_log.jsonl`**  
  → Raw conversation traces from Week 10. Used in Week 11 to mine evaluation tasks and adversarial probes.

- **`docs/probe_library.md`**  
  → Probe library authored in Week 10. Week 11 uses it to generate evaluation prompts and stress‑test the agent.

- **`docs/failure_taxonomy.md`**  
  → Failure taxonomy defined in Week 10. Week 11 uses it to categorise evaluation tasks and ensure coverage across failure types.

- **`docs/hiring_signal_brief.md`** and **`docs/competitor_gap.md`**  
  → Strategic briefs from Week 10. Week 11 references them to contextualise evaluation tasks so the bench reflects Tenacious’s positioning.

- **`outputs/brief_outputs/`**  
  → Generated outputs from briefs. Week 11 uses them as reference material for evaluation and comparison.

- **`checks/style_guide_adherence.md`**  
  → Tenacious style guide adherence checks. Week 11 integrates these into Path A evaluation (stylistic quality).

---

### 🔑 How They Integrate into Week 11’s Build

- **Trace log + probe library + failure taxonomy** → Feed into Week 11’s `data_prep` scripts to generate the evaluation dataset splits (`train.json`, `dev.json`, `heldout.json`).  
- **Hiring signal brief + competitor gap + brief outputs** → Provide **contextual grounding** for evaluation tasks, ensuring the bench reflects Tenacious’s business needs.  
- **Style guide adherence checks** → Applied during evaluation (Path A) to measure stylistic quality alongside correctness.  
- **Synthetic preference pairs + adversarial cases** (also Week 10 deliverables) → Imported into Week 11’s `data/splits/` to train and evaluate the Judge (Path B).  
- **Datasheet.md** → Documents dataset provenance and reliability, required in Week 11.  
- **Baseline outputs + Judge model + model_card.md** → Week 10 evaluation artifacts are imported into Week 11’s `models/` and `reports/` for comparison.

---

### 📌 Integration Flow in Week 11
1. **Data prep scripts** read `trace_log.jsonl`, `probe_library.md`, and `failure_taxonomy.md` to generate evaluation tasks.  
2. **Contextual briefs** (hiring signal, competitor gap, brief outputs) are referenced when constructing tasks to ensure relevance.  
3. **Checks/style guide adherence** are applied during evaluation (Path A).  
4. **Judge model + baseline outputs** from Week 10 are run against the held‑out set in Week 11 evaluation scripts.  
5. **Datasheet.md + model_card.md** are carried forward as documentation deliverables.

---

👉 In short: Week 10 artifacts live under `data/raw/`, `docs/`, `outputs/`, and `checks/` in the Week 11 repo. They are imported into Week 11’s data prep, evaluation, and reporting pipeline to form the foundation of the Sales Agent Evaluation Bench.  

Would you like me to draft a **visual integration diagram** (boxes/arrows showing Week 10 artifacts flowing into Week 11’s pipeline) so you can drop it straight into your docs?

===
Here’s a clear **Week 11 Inputs Manifest** you can drop into your repo (`docs/week11_inputs.md`). It explicitly lists each artifact from Week 10, where to store it in the Week 11 repo, and how it is used in the pipeline.

---

# 📑 Week 11 Inputs Manifest

This document inventories all Week 10 deliverables imported into Week 11’s Sales Agent Evaluation Bench and explains their role.

---

## Week 10 Artifacts → Week 11 Integration

| Artifact | Repo Location (Week 11) | Role in Week 11 |
|----------|--------------------------|-----------------|
| **trace_log.jsonl** | `data/raw/trace_log.jsonl` | Seed corpus for trace‑derived tasks. Converted into (input, candidate output) pairs for evaluation dataset. |
| **probe_library.md** | `docs/probe_library.md` | Source of adversarial task seeds. Each probe expands into 3–8 task variants. |
| **failure_taxonomy.md** | `docs/failure_taxonomy.md` | Provides schema dimensions for Tenacious‑Bench. Used to categorise tasks and ensure coverage. |
| **Hiring signal brief & competitor gap** | `docs/hiring_signal_brief.md`, `docs/competitor_gap.md` | Contextual input templates. Tasks require grounding in these briefs to score correctly. |
| **Brief outputs** | `outputs/brief_outputs/` | Reference material for evaluation tasks. Used to compare agent outputs against expected brief‑grounded responses. |
| **Tenacious style guide adherence checks** | `checks/style_guide_adherence.md` | Alignment objective for trained component. Integrated into Path A evaluation for stylistic quality. |
| **Synthetic preference pairs** | `data/splits/train.json` (training partition) | Used in Path B judge training. Constructed from probe‑triggered failures vs corrected outputs. |
| **Adversarial cases** | `data/splits/dev.json` + `data/splits/heldout.json` | Stress‑test tasks in evaluation dataset. Carry originality weight in Tenacious‑Bench. |
| **Datasheet.md** | `data/datasheet.md` | Documents dataset provenance, contamination checks, and inter‑rater agreement. Required for HuggingFace publication. |
| **Baseline outputs** | `reports/baseline_metrics.json` | Comparison point against Judge model on Tenacious‑Bench held‑out. |
| **Judge model + model_card.md** | `models/judge/`, `models/model_card.md` | Imported from Week 10 Path B deliverable. Evaluated on Tenacious‑Bench and compared to baseline. |

---

## 🔑 Integration Notes
- **Data Prep (Acts I–II):** Trace logs, probes, taxonomy, briefs, and style checks feed into dataset authoring scripts.  
- **Training (Act III–IV):** Synthetic pairs and adversarial cases form the training/dev/held‑out partitions. Judge model fine‑tuned here.  
- **Evaluation (Act IV):** Baseline outputs and Judge model are run against held‑out tasks. Confusion matrices and ablation results logged.  
- **Documentation (Acts II–V):** Datasheet.md and model_card.md are carried forward, updated with Week 11 results.  
- **Publication (Act V):** HuggingFace dataset + model, blog post, and community artifact reference these Week 10 inputs as lineage.  

