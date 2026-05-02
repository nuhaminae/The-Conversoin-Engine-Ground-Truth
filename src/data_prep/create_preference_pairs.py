# src/data_prep/create_preference_pairs.py

"""
Convert pointwise Tenacious-Bench splits into Path B preference-pair JSONL.

Reads rows with label=1/0 grouped by pair_id and writes:
  {prompt, chosen, rejected, pair_id, source_mode, failure_mode_tag, metadata}

Default input splits:
  tenacious_bench/train/train.jsonl
  tenacious_bench/dev/dev.jsonl
  tenacious_bench/held_out/held_out.jsonl
Default output:
  tenacious_bench/dpo/{split}_dpo.jsonl
  data/training_data/preferences_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Reads a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed line {line_no} in {path}: {exc}")


def build_prompt(task: Dict[str, Any]) -> str:
    """Builds the prompt for a given task."""
    prospect_input = str(task.get("prospect_input", "")).strip()
    scenario_type = task.get("scenario_type", "unknown")
    source_mode = task.get("source_mode", "unknown")
    return (
        "You are a Tenacious-Bench judge. Prefer the response that best follows the Tenacious requirements: "
        "grounded in the supplied signal, honest about uncertainty, concise, professional, non-condescending, "
        "one clear next step, no banned phrases, no unsupported capacity/pricing claims, and safe tool-error handling.\n\n"
        f"Source mode: {source_mode}\nScenario type: {scenario_type}\n"
        f"Task input:\n{prospect_input}"
    )


def create_dpo_pairs_from_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Creates a list of preference-pair JSON objects from a list of rows."""
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: {"good": [], "bad": []}
    )
    for row in rows:
        pair_id = row.get("pair_id") or row.get("prospect_input")
        if row.get("label") == 1:
            grouped[pair_id]["good"].append(row)
        elif row.get("label") == 0:
            grouped[pair_id]["bad"].append(row)

    pairs: List[Dict[str, Any]] = []
    for pair_id, bucket in grouped.items():
        if not bucket["good"] or not bucket["bad"]:
            print(
                f"[WARN] Skipping incomplete DPO pair {pair_id}: good={len(bucket['good'])}, bad={len(bucket['bad'])}"
            )
            continue
        # Keep only the first good/bad per pair_id. The split step already deduplicates.
        chosen_row = bucket["good"][0]
        rejected_row = bucket["bad"][0]
        chosen = str(chosen_row.get("agent_output", "")).strip()
        rejected = str(rejected_row.get("agent_output", "")).strip()
        if not chosen or not rejected:
            print(f"[WARN] Skipping blank DPO pair {pair_id}")
            continue
        pairs.append(
            {
                "pair_id": pair_id,
                "prompt": build_prompt(chosen_row),
                "chosen": chosen,
                "rejected": rejected,
                "source_mode": chosen_row.get("source_mode"),
                "scenario_type": chosen_row.get("scenario_type"),
                "failure_code": rejected_row.get("failure_code"),
                "failure_mode_tag": rejected_row.get("failure_mode_tag"),
                "metadata": {
                    "chosen_task_id": chosen_row.get("task_id"),
                    "rejected_task_id": rejected_row.get("task_id"),
                    "chosen_metadata": chosen_row.get("metadata", {}),
                    "rejected_metadata": rejected_row.get("metadata", {}),
                },
            }
        )
    return pairs


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    """Writes a list of JSON objects to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def create_dpo_pairs_for_split(
    input_path: str, output_path: str
) -> List[Dict[str, Any]]:
    """Creates a list of preference-pair JSON objects from a JSONL file."""
    if not os.path.exists(input_path):
        print(f"[WARN] Input split not found: {input_path}")
        return []
    rows = list(read_jsonl(input_path))
    pairs = create_dpo_pairs_from_rows(rows)
    count = write_jsonl(output_path, pairs)
    print(f"Created {count} DPO pairs -> {output_path}")
    return pairs


def create_all_dpo_pairs(
    base_dir: str, output_dir: str, training_data_dir: str
) -> Dict[str, Any]:
    """Creates a list of preference-pair JSON objects from all splits."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(training_data_dir, exist_ok=True)
    report: Dict[str, Any] = {"splits": {}}

    for split_name in ["train", "dev", "held_out"]:
        input_path = os.path.join(base_dir, split_name, f"{split_name}.jsonl")
        output_path = os.path.join(output_dir, f"{split_name}_dpo.jsonl")
        pairs = create_dpo_pairs_for_split(input_path, output_path)
        report["splits"][split_name] = {
            "pairs": len(pairs),
            "output_path": output_path,
            "source_modes": dict(Counter(pair.get("source_mode") for pair in pairs)),
            "failure_codes": dict(Counter(pair.get("failure_code") for pair in pairs)),
        }
        if split_name == "train":
            train_out = os.path.join(training_data_dir, "preferences_train.jsonl")
            write_jsonl(train_out, pairs)
            report["training_data_path"] = train_out
        if split_name == "dev":
            dev_out = os.path.join(training_data_dir, "preferences_dev.jsonl")
            write_jsonl(dev_out, pairs)
            report["dev_data_path"] = dev_out

    report_path = os.path.join(output_dir, "dpo_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote DPO report -> {report_path}")
    print(
        "Note: held_out_dpo.jsonl is for evaluation sanity checks only. Do not train on held_out."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="tenacious_bench")
    parser.add_argument("--output-dir", default="tenacious_bench/dpo")
    parser.add_argument("--training-data-dir", default="data/training_data")
    args = parser.parse_args()
    create_all_dpo_pairs(args.base_dir, args.output_dir, args.training_data_dir)


if __name__ == "__main__":
    main()
