Here’s a **step‑by‑step guide** to produce the Week 11 interim submission artifacts and package them correctly for grading.

---

## 🖥️ What to Run

### 1. Prepare datasets (Acts I–II)
Run the pipeline up to dataset generation:

```bat
python src\data_prep\trace_tasks.py
python src\data_prep\synthetic_pairs.py
python src\data_prep\adversarial_cases.py
python src\data_prep\split_dataset.py
```

This will create:
- `tenacious_bench_v0.1/train/`
- `tenacious_bench_v0.1/dev/`
- `tenacious_bench_v0.1/held_out/`

### 2. Generate contamination check + datasheet
- Run your contamination‑check script (if separate) or add logic in `split_dataset.py` to output `contamination_check.json`.
- Write `data/datasheet.md` following Gebru & Pushkarna templates.

### 3. Schema + audit
- Ensure `schema.json` is in root (defines dimensions, signals, failure taxonomy).
- Write `audit_memo.md` summarising Week 10 artifacts imported (trace_log, probe_library, failure_taxonomy, briefs, style guide).

### 4. Inter‑rater agreement
- Collect annotator scores on a sample of tasks.
- Save results in `inter_rater_agreement.md`.

### 5. Cost log
- Update `cost_log.md` with compute time, API usage, and human annotation hours.

---

## 📂 Repo Structure Checklist

- `README.md` → overview, setup, status, next steps.  
- `audit_memo.md` → audit of Week 10 artifacts.  
- `schema.json` → schema dimensions.  
- `scoring_evaluator.py` → rubric application script.  
- `tenacious_bench_v0.1/` → train/dev/held_out partitions.  
- `datasheet.md` → dataset documentation.  
- `methodology.md` → path declaration, justification, contamination check.  
- `generation_scripts/` → reproducible authoring code.  
- `inter_rater_agreement.md` → annotator agreement results.  
- `synthesis_memos/` → at least two memos.  
- `cost_log.md` → compute + human cost.

---

## 📑 Interim PDF Report (content outline)

**Title:** Interim Submission – Tenacious Bench v0.1

**1. Bench Composition**
- Counts per dimension (failure taxonomy categories, ICP segments, hiring signals).
- Partition sises (train/dev/held_out).
- Source modes (trace‑derived, synthetic, adversarial).

**2. Inter‑Rater Agreement**
- Method (Cohen’s κ or % agreement).
- Results table.
- Interpretation (strengths, weaknesses).

**3. Example Tasks**
- **Programmatic:** synthetic preference pair with rubric application.  
- **Trace‑derived:** task from `trace_log.jsonl`.  
- **Adversarial:** probe‑seeded failure case.  
- For each: show rubric scoring (tone markers, style guide checks, taxonomy category).

**4. What’s Working**
- Dataset partitions reproducible.
- Schema dimensions cover Week 10 taxonomy.
- Style guide checks integrated.

**5. What’s Not Working**
- Some signals weak → need more adversarial coverage.
- Inter‑rater agreement below target in one dimension.

**6. Plan for Days 4–7**
- Expand adversarial cases.
- Improve contamination check automation.
- Push dataset + Judge model to HuggingFace in final submission.

---

## 🚀 Submission Steps

1. Run `scripts/run_pipeline.bat` **through Step 1 only** (Acts I–II).  
2. Verify dataset partitions and contamination check outputs.  
3. Fill in `datasheet.md`, `audit_memo.md`, `methodology.md`, `inter_rater_agreement.md`.  
4. Export PDF report with the outline above.  
5. Push repo to GitHub (public).  
6. Upload PDF to Google Drive, share public link.  
7. Submit both links by **Wednesday, 21:00 UTC**.

---

This workflow ensures you meet the rubric: dataset present with three partitions, datasheet, contamination check, schema, audit, inter‑rater agreement, synthesis memos, and cost log — plus the PDF report covering composition, agreement, examples, and forward plan.  

Would you like me to draft the **actual LaTeX template for the PDF report** so you can compile it directly into a polished submission?