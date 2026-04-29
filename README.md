# The-Conversion-Engine-Ground-Truth

The goal of the project is to build a **Tenacious-specific evaluation benchmark** and train a **Judge/Critic model (Path B)** to improve reliability in B2B sales conversations.

```bash
The-Conversion-Engine-Ground-Truth/
│
├── data/
│   ├── raw/                # Seed corpus from Tenacious (style guide, case studies, transcripts)
│   ├── processed/          # Cleaned & normalised datasets
│   ├── tasks/              # 200–300 evaluation tasks (trace-derived, synthetic, adversarial)
│   ├── splits/             # Train/dev/held-out partitions
│   └── datasheet.md        # Dataset documentation (sources, contamination checks, IRR logs)
│
├── models/
│   ├── judge/              # Judge model artifacts (LoRA weights, configs)
│   ├── checkpoints/        # Training checkpoints
│   └── model_card.md       # Model documentation
│
├── src/
│   ├── data_prep/          # Scripts for dataset construction & preprocessing
│   │   ├── trace_tasks.py
│   │   ├── synthetic_pairs.py
│   │   ├── adversarial_cases.py
│   │   └── split_dataset.py
│   │
│   ├── training/           # Training scripts for preference optimisation
│   │   ├── train_judge.py
│   │   └── utils.py
│   │
│   ├── evaluation/         # Ablation & benchmark evaluation
│   │   ├── eval_judge.py
│   │   └── metrics.py
│   │
│   └── integration/        # Pipeline integration (agent + judge)
│       └── run_with_judge.py
│
├── notebooks/
│   ├── exploratory_data.ipynb   # Inspect seed corpus & tasks
│   ├── training_logs.ipynb      # Track judge training
│   └── evaluation_results.ipynb # Visualise ablations
│
├── reports/
│   ├── executive_memo.pdf       # 2-page memo for Tenacious leadership
│   └── blog_post.md             # Technical blog post for community
│
├── community/
│   ├── github_issue.md          # Contribution to open evaluation repo
│   └── workshop_submission.md   # Submission draft for evaluation workshop
│
├── configs/
│   ├── training_config.yaml     # Hyperparameters, LoRA settings
│   └── eval_config.yaml         # Evaluation parameters
│
├── scripts/
│   └── run_pipeline.sh          # Shell script to run end-to-end pipeline
│
├── requirements.txt             # Dependencies (mirrors uv install list)
├── README.md                    # Project overview & instructions
└── LICENSE                      # License for dataset & model artifacts
```

---

## Setup

Install dependencies using `uv`:

```bash
uv pip install -r requirements.txt
```

Or directly:

```bash
uv pip install torch transformers datasets accelerate peft bitsandbytes \
sentencepiece evaluate scikit-learn pandas numpy matplotlib tqdm huggingface-hub \
safetensors wandb jupyter ipywidgets
```

---

## Workflow

1. **Dataset Construction**  
   - Build preference pairs (good vs. bad agent outputs).  
   - Partition into train/dev/held-out sets.  
   - Document in `datasheet.md`.

2. **Judge Training**  
   - Run `src/training/train_judge.py` with configs in `configs/training_config.yaml`.  
   - Save checkpoints in `models/checkpoints/`.

3. **Evaluation**  
   - Use `src/evaluation/eval_judge.py` to run ablations.  
   - Compare against Week 10 baseline.  
   - Log results in `notebooks/evaluation_results.ipynb`.

4. **Integration**  
   - Run `src/integration/run_with_judge.py` to test agent + judge pipeline.  

5. **Deliverables**  
   - Dataset (HuggingFace upload + datasheet).  
   - Judge model (LoRA adapter + model card).  
   - Blog post (`reports/blog_post.md`).  
   - Executive memo (`reports/executive_memo.pdf`).  
   - Community contribution (`community/github_issue.md`).  

---

## Evaluation Metrics

- **Consistency score** – Judge preference accuracy.  
- **Lift over baseline** – Improvement vs. Week 10 agent.  
- **Inter-rater agreement** – Dataset quality validation.  

---

## License

Dataset and model licensing terms will be specified here.
