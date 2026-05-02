@echo off
REM ============================================================
REM Run: scripts\run_pipeline.bat at the root of the repo
REM
REM Full Tenacious Judge (Path B) Pipeline
REM NOTE: Python scripts are responsible for loading .env secrets.
REM ============================================================

echo [1/5] Preparing datasets...
REM Ensure you have created all six types of task generation scripts.
python src\data_prep\trace_tasks.py
python src\data_prep\programmatic_tasks.py
python src\data_prep\synthetic_pairs.py
python scripts\summarise_openrouter_costs.py
python src\data_prep\adversarial_cases.py
python src\data_prep\split_dataset.py
python src\data_prep\create_preference_pairs.py

echo [2/5] Training Preference-Tuned Judge (DPO)...
REM The training script takes the config file as its primary argument. (Run on unsloth)
python src\training\train_judge.py --config configs\training_config.yaml

echo [3/5] Evaluating agents against the held-out benchmark...
REM --- Delta A: Fine-Tuned Judge vs. Baseline ---
echo    -> Evaluating Fine-Tuned Judge (your final model) (Run on unsloth)
python src\evaluation\eval_judge.py --config configs\eval_config.yaml
echo    -> Evaluating Baseline (Week 10 agent, no judge)
python src\evaluation\eval_baseline.py --config configs\eval_config.yaml
REM --- Delta B: Fine-Tuned Judge vs. Prompting (Honesty Check) ---
echo    -> Evaluating Prompt-Engineered Judge (honesty check)
python src\evaluation\eval_prompted_judge.py --config configs\eval_config.yaml

echo [4/5] Generating comparison report...
REM This notebook compares all three metric files.
jupyter nbconvert --execute notebooks\evaluation_result.ipynb --to html --output-dir reports --output evaluation_results
jupyter nbconvert --execute notebooks\training_logs.ipynb --to html --output-dir reports --output training_logs
jupyter nbconvert --execute notebooks\exploratory_data.ipynb --to html --output-dir reports --output exploratory_data

echo [5/5] Packaging final artifacts...
REM This step prepares the benchmark for HuggingFace upload (Act V).
python scripts\package_final_artifacts.py
echo    -> Final artifacts prepared in dist\Act_v_package
echo    -> Zip package prepared at dist\Act_v_package.zip

echo ============================================================
echo Pipeline complete! Results saved in /reports
echo - fine_tuned_judge_metrics.json
echo - prompted_judge_metrics.json
echo - baseline_metrics.json
echo - ablation_comparison_chart.png
echo - evaluation_results.html
echo ============================================================

pause
