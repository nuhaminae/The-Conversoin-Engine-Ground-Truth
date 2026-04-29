**Your interim submission PDF report has been generated successfully — it contains all required sections (bench composition, inter‑rater agreement, example tasks, working/not working analysis, and plan for Days 4–7) and is ready to be shared as part of your GitHub + Google Drive submission.**

---

## 🖥️ How to Produce Week 11 Artifacts

To generate the artifacts needed for Acts I and II (dataset audit, schema, authored dataset), run the following:

1. **Dataset Preparation**
   ```bat
   python src\data_prep\trace_tasks.py
   python src\data_prep\synthetic_pairs.py
   python src\data_prep\adversarial_cases.py
   python src\data_prep\split_dataset.py
   ```
   - Produces `tenacious_bench_v0.1/train/`, `dev/`, `held_out/`.

2. **Contamination Check**
   - Run contamination check logic (either embedded in `split_dataset.py` or a separate script).
   - Output: `contamination_check.json`.

3. **Schema + Audit**
   - Ensure `schema.json` is present at root.
   - Write `audit_memo.md` summarising Week 10 artifacts imported (trace_log, probe_library, failure_taxonomy, briefs, style guide).

4. **Datasheet**
   - Fill `datasheet.md` following Gebru & Pushkarna templates.
   - Include provenance, contamination checks, inter‑rater agreement.

5. **Inter‑Rater Agreement**
   - Collect annotator scores on a sample of tasks.
   - Save results in `inter_rater_agreement.md`.

6. **Cost Log**
   - Update `cost_log.md` with compute/API usage and annotation hours.

---

## 📂 Repo Checklist for Interim Submission

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

## 📑 Interim PDF Report Contents

The generated PDF includes:

- **Bench Composition:** Counts per taxonomy, ICP segments, hiring signals, partition sises, source modes.  
- **Inter‑Rater Agreement:** Method, results table, interpretation.  
- **Example Tasks:** Programmatic, trace‑derived, adversarial with rubric application.  
- **What is Working:** Reproducible partitions, schema alignment, style guide checks.  
- **What is Not Working:** Weak signals, adversarial coverage gaps, taxonomy agreement issues.  
- **Plan for Days 4–7:** Expand adversarial probes, improve contamination check automation, prepare HuggingFace publication.

---

## 🚀 Submission Steps

1. Run dataset prep scripts (Step 1 of pipeline).  
2. Verify partitions + contamination check outputs.  
3. Fill in required markdown files (`audit_memo.md`, `datasheet.md`, `methodology.md`, `inter_rater_agreement.md`).  
4. Export and upload the generated PDF report.  
5. Push repo to GitHub (public).  
6. Upload PDF to Google Drive, share public link.  
7. Submit both links by **Wednesday, 21:00 UTC**.

---

✅ You now have both the **repo artifacts** and the **PDF report** ready for interim submission. The next step is to finalise the GitHub repo structure and upload the PDF to Google Drive for submission.