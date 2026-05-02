# src/training/train_judge.py

"""
Train a Path B preference-tuned judge/critic for Tenacious-Bench using DPO.

Input:
  data/training_data/preferences_train.jsonl
  data/training_data/preferences_dev.jsonl

Each row must contain:
  prompt, chosen, rejected

Outputs:
  models/checkpoints/
  models/judge/
  reports/training/training_summary.json
  reports/training/training_config_used.yaml
"""

import unsloth
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

PatchDPOTrainer()

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import wandb
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from trl import DPOConfig, DPOTrainer


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_yaml(data: Dict[str, Any], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def get_optional(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# -----------------------------
# Environment and auth
# -----------------------------
def authenticate_services(config: Dict[str, Any]) -> Tuple[str, str]:
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN", "").strip()
    wandb_api_key = os.getenv("WANDB_API_KEY", "").strip()
    use_wandb = bool(get_optional(config, "reporting", "use_wandb", default=False))

    if hf_token:
        # Current huggingface_hub authentication path.
        # This makes the token available to model downloads and Hub operations.
        login(token=hf_token, add_to_git_credential=False)
        print("Hugging Face Hub authenticated with login().")
    else:
        print("No HF_TOKEN found. This is OK for public models and no Hub push.")

    if wandb_api_key and use_wandb:
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=get_optional(config, "reporting", "wandb_project", default="tenacious-judge"),
            name=get_optional(config, "reporting", "wandb_run_name", default="dpo-judge"),
            config=config,
        )
        print("W&B tracking initialised.")
    else:
        os.environ["WANDB_DISABLED"] = "true"
        print("W&B disabled.")

    return hf_token, wandb_api_key


# -----------------------------
# Data loading
# -----------------------------
def load_dpo_dataset(data_config: Dict[str, Any]):
    """
    Loads explicit train/dev preference-pair JSONL files.
    Required columns: prompt, chosen, rejected.
    """
    data_files = {
        "train": data_config["train_file"],
        "validation": data_config["dev_file"],
    }

    for split, path in data_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{split} file not found: {path}. "
                "Expected data/training_data/preferences_train.jsonl and "
                "data/training_data/preferences_dev.jsonl after Stage 1."
            )

    print(f"Loading DPO dataset from: {data_files}")
    dataset = load_dataset("json", data_files=data_files)

    required = {"prompt", "chosen", "rejected"}

    for split_name in ["train", "validation"]:
        missing = required - set(dataset[split_name].column_names)
        if missing:
            raise ValueError(
                f"{split_name} split is missing required DPO columns: {missing}. "
                f"Columns found: {dataset[split_name].column_names}"
            )

        bad_rows = []
        for idx, row in enumerate(dataset[split_name]):
            for key in required:
                if not isinstance(row.get(key), str) or not row[key].strip():
                    bad_rows.append((idx, key))
                    break
            if len(bad_rows) >= 5:
                break

        if bad_rows:
            raise ValueError(
                f"{split_name} has empty/non-string prompt/chosen/rejected fields. "
                f"Examples: {bad_rows}"
            )

    print(f"Train pairs: {len(dataset['train'])}")
    print(f"Validation pairs: {len(dataset['validation'])}")

    return dataset["train"], dataset["validation"]


# -----------------------------
# Main training
# -----------------------------
def main(config_path: str) -> None:
    started_at = time.time()

    print("Loading configuration from:", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    output_config = config["output"]

    report_dir = output_config.get("report_dir", "reports/training")
    checkpoint_dir = output_config["checkpoint_dir"]
    model_dir = output_config["model_dir"]

    ensure_dir(report_dir)
    ensure_dir(checkpoint_dir)
    ensure_dir(model_dir)

    save_yaml(config, os.path.join(report_dir, "training_config_used.yaml"))

    hf_token, wandb_api_key = authenticate_services(config)

    print(f"Loading base model with Unsloth: {model_config['base_model']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["base_model"],
        max_seq_length=data_config["max_length"],
        dtype=None,
        load_in_4bit=True,
        token=hf_token or None,
    )

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Configuring LoRA adapter.")
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_config["lora"]["r"],
        target_modules=model_config["lora"]["target_modules"],
        lora_alpha=model_config["lora"]["alpha"],
        lora_dropout=model_config["lora"]["dropout"],
        bias="none",
        use_gradient_checkpointing=True,
        random_state=config.get("seed", 42),
        max_seq_length=data_config["max_length"],
    )

    print("Loading DPO preference data.")
    train_dataset, eval_dataset = load_dpo_dataset(data_config)

    dataset_summary = {
        "train_file": data_config["train_file"],
        "dev_file": data_config["dev_file"],
        "train_pairs": len(train_dataset),
        "validation_pairs": len(eval_dataset),
        "required_columns": ["prompt", "chosen", "rejected"],
    }
    save_json(dataset_summary, os.path.join(report_dir, "dataset_summary.json"))

    report_to = (
        "wandb"
        if wandb_api_key and get_optional(config, "reporting", "use_wandb", default=False)
        else "none"
    )

    # DPOConfig is the current TRL config object for DPOTrainer.
    dpo_args = DPOConfig(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config.get(
            "eval_batch_size", training_config["batch_size"]
        ),
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        num_train_epochs=training_config["num_epochs"],
        learning_rate=training_config["learning_rate"],
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        weight_decay=training_config.get("weight_decay", 0.0),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "linear"),
        logging_steps=training_config.get("logging_steps", 1),
        save_strategy=training_config.get("save_strategy", "epoch"),
        eval_strategy=training_config.get(
            "eval_strategy",
            training_config.get("evaluation_strategy", "epoch"),
        ),
        seed=config.get("seed", 42),
        optim=training_config.get("optim", "adamw_8bit"),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        beta=training_config.get("beta", 0.1),
        max_length=data_config["max_length"],
        max_prompt_length=data_config.get("max_prompt_length", data_config["max_length"] // 2),
        report_to=report_to,
        remove_unused_columns=False,
    )

    print("Initialising DPOTrainer.")
    try:
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
    except TypeError:
        # Compatibility fallback for older TRL releases.
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

    print("\n--- Starting DPO Training ---\n")
    train_result = dpo_trainer.train()
    print("\n--- DPO Training Complete ---\n")

    print("Running final validation evaluation.")
    eval_metrics = dpo_trainer.evaluate()

    print(f"Saving final LoRA adapter to {model_dir}")
    dpo_trainer.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Copy trainer state if available.
    trainer_state = Path(checkpoint_dir) / "trainer_state.json"
    if trainer_state.exists():
        shutil.copy2(trainer_state, Path(report_dir) / "trainer_state.json")

    summary = {
        "status": "complete",
        "base_model": model_config["base_model"],
        "model_dir": model_dir,
        "checkpoint_dir": checkpoint_dir,
        "report_dir": report_dir,
        "train_pairs": len(train_dataset),
        "validation_pairs": len(eval_dataset),
        "train_metrics": getattr(train_result, "metrics", {}),
        "eval_metrics": eval_metrics,
        "runtime_seconds": round(time.time() - started_at, 2),
        "seed": config.get("seed", 42),
    }

    save_json(summary, os.path.join(report_dir, "training_summary.json"))

    if wandb_api_key and get_optional(config, "reporting", "use_wandb", default=False):
        wandb.finish()

    print("\nTraining summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DPO preference-tuned judge for Tenacious-Bench."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training_config.yaml.",
    )
    args = parser.parse_args()
    main(args.config)