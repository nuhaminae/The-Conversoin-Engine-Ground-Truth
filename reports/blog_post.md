# Blog Post Draft Structure – Tenacious Sales Evaluation Benchmark (Path B)

## Title

**From Missed Meetings to Reliable Sales Agents: Building a Tenacious‑Specific Evaluation Benchmark**

---

## 1. Introduction

- Context: Week 10 demo revealed thin outputs and missed intent (prospects said “yes,” agent didn’t schedule).  
- Problem: Existing benchmarks (τ²‑Bench retail) don’t capture Tenacious’s B2B sales context.  
- Goal: Build a Tenacious‑specific evaluation benchmark and train a Judge/Critic model to enforce consistency.

---

## 2. Dataset Construction

- **Sources:**  
  - Trace‑derived tasks (Week 10 transcripts).  
  - Synthetic preference pairs (prospect “yes” → good vs. bad agent outputs).  
  - Multi‑LLM synthesis with judge filtering.  
  - Adversarial cases (edge scenarios like “maybe later”).  
- **Size:** ~200–300 tasks.  
- **Schema:** Prospect input, agent output, label (0/1), failure mode tag, metadata.  
- **Quality checks:** Contamination checks, inter‑rater agreement, judge filtering.

---

## 3. Model Training (Path B)

- **Approach:** Preference optimisation for binary classification (good vs. bad outputs).  
- **Base model:** DistilBERT (lightweight, efficient).  
- **Fine‑tuning:** LoRA adapter for preference scoring.  
- **Integration:** Agent generates → Judge scores → weak outputs rejected.  

---

## 4. Evaluation

- **Metrics:** Accuracy, precision, recall, F1.  
- **Baseline heuristic:** Keyword detection (“schedule,” “invite”).  
- **Results:** Judge model improved consistency by [X%] over baseline.  
- **Visuals:** Confusion matrix, comparison chart (baseline vs. judge).  

---

## 5. Business Impact

- **Reliability:** Prospects saying “yes” now consistently trigger scheduling.  
- **Pipeline protection:** No lost meetings due to missed intent.  
- **Scalability:** Framework can extend to tone/style (Path A) and reasoning (Path C).  

---

## 6. Community Contribution

- **Open dataset:** HuggingFace upload with datasheet.  
- **Model card:** Judge model documentation.  
- **Workshop submission:** Sharing lessons learned with evaluation community.  

---

## 7. Conclusion

- Tenacious‑specific benchmark fills a critical gap.  
- Judge model enforces consistency, lifting reliability.  
- Next steps: expand dataset, layer tone/style improvements, contribute back to community.  
