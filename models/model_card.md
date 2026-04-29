# 📑 Model Card – Tenacious Judge Model (Path B)

## Model Details

- **Model name:** Tenacious Judge (Path B)  
- **Version:** v1.0  
- **Architecture:** DistilBERT (fine‑tuned for binary classification)  
- **Training method:** Preference optimisation with LoRA adapters  
- **Developed by:** [Your Team Name]  
- **Date:** April 2026  

---

## Intended Use

- **Primary:** Evaluate agent outputs in B2B sales conversations.  
- **Function:** Score outputs as *good* (consistent with prospect intent) or *bad* (missed intent).  
- **Integration:** Agent generates → Judge scores → weak outputs rejected or corrected.  
- **Not intended for:** General retail benchmarks, medical, legal, or consumer chat contexts.  

---

## Training Data

- **Dataset size:** ~200–300 tasks.  
- **Sources:**  
  - Trace‑derived tasks (Week 10 demo transcripts).  
  - Synthetic preference pairs.  
  - Multi‑LLM synthesis with judge filtering.  
  - Adversarial cases.  
- **Languages:** English (Tenacious sales context).  
- **Splits:** Train (70%), Dev (15%), Held‑out (15%).  

---

## Evaluation

- **Metrics:** Accuracy, precision, recall, F1.  
- **Baseline comparison:** Keyword heuristic (Week 10 agent outputs).  
- **Results:** Judge model improved consistency by [X%] over baseline.  
- **Visuals:** Confusion matrix and comparison charts included in `/reports`.  

---

## Limitations

- **Scope:** Focused only on Tenacious‑specific failure mode (missed scheduling after prospect “yes”).  
- **Biases:** Synthetic data may not capture full diversity of real conversations.  
- **Generalisation:** Not validated outside Tenacious B2B context.  

---

## Ethical Considerations

- **Transparency:** Datasheet.md documents dataset sources and contamination checks.  
- **Reliability:** Judge enforces consistency but does not guarantee stylistic quality (Path A needed).  
- **Privacy:** No personal prospect data included; all traces anonymised.  

---

## Maintenance

- **Maintainers:** [Your Team Name]  
- **Update frequency:** Iterative updates as new failure modes are discovered.  
- **Contact:** [Your email or GitHub handle]  

