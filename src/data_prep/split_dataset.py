# src/data_prep/split_dataset.py
"""
Combine generated task files, cap to the Week 11 target size, and split by pair_id.

Why pair-level splitting matters:
- Path B needs chosen/rejected rows for the same prompt in the same partition.
- Random row-level splitting can separate pairs and destroy DPO data.

Default output:
  tenacious_bench/train/train.jsonl
  tenacious_bench/dev/dev.jsonl
  tenacious_bench/held_out/held_out.jsonl
  tenacious_bench/dataset_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

DEFAULT_INPUT_FILES = [
    "data/tasks/trace_tasks.json",
    "data/tasks/programmatic_tasks.json",
    "data/tasks/synthetic_pairs.json",
    "data/tasks/adversarial_cases.json",
]

SOURCE_TARGET_SHARES = {
    "trace-derived": 0.30,
    "programmatic": 0.30,
    "multi-LLM synthesis": 0.25,
    "hand-authored adversarial": 0.15,
}


def load_json_array(path: str) -> List[Dict[str, Any]]:
    """Load a JSON array from a file."""
    if not os.path.exists(path):
        print(f"[WARN] Missing input file: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def validate_task(task: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Validate a task row."""
    required = [
        "pair_id",
        "source_mode",
        "prospect_input",
        "agent_output",
        "label",
        "failure_mode_tag",
    ]
    missing = [k for k in required if k not in task]
    if missing:
        raise ValueError(f"Task #{index} missing required keys: {missing}")
    if task["label"] not in (0, 1):
        raise ValueError(f"Task #{index} has invalid label {task['label']!r}")
    if not str(task.get("agent_output", "")).strip():
        raise ValueError(f"Task #{index} has blank agent_output")
    return task


def group_valid_pairs(
    tasks: Iterable[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for i, task in enumerate(tasks):
        """Validate and group tasks by pair_id."""
        task = validate_task(task, i)
        groups[task["pair_id"]].append(task)

    valid: Dict[str, List[Dict[str, Any]]] = {}
    for pair_id, rows in groups.items():
        labels = {row["label"] for row in rows}
        if labels == {0, 1}:
            # Keep one chosen and one rejected if duplicates exist.
            chosen = next(row for row in rows if row["label"] == 1)
            rejected = next(row for row in rows if row["label"] == 0)
            valid[pair_id] = [chosen, rejected]
        else:
            print(f"[WARN] Dropping incomplete pair {pair_id}: labels={labels}")
    return valid


def source_mode_for_pair(rows: List[Dict[str, Any]]) -> str:
    """Get the source mode for a pair of tasks."""
    return rows[0].get("source_mode", "unknown")


def choose_pairs_by_source(
    groups: Dict[str, List[Dict[str, Any]]], target_tasks: int, seed: int
) -> List[str]:
    """Choose pairs by source mode, with a seed for reproducibility."""
    rng = random.Random(seed)
    target_pairs = target_tasks // 2
    by_source: Dict[str, List[str]] = defaultdict(list)
    for pair_id, rows in groups.items():
        by_source[source_mode_for_pair(rows)].append(pair_id)
    for ids in by_source.values():
        rng.shuffle(ids)

    selected: List[str] = []
    selected_set = set()

    # First pass: honor source-mode shares as closely as possible.
    for source, share in SOURCE_TARGET_SHARES.items():
        wanted = round(target_pairs * share)
        available = by_source.get(source, [])
        for pair_id in available[:wanted]:
            if pair_id not in selected_set:
                selected.append(pair_id)
                selected_set.add(pair_id)

    # Second pass: fill any remainder from any source with available pairs.
    all_remaining = [
        pid for ids in by_source.values() for pid in ids if pid not in selected_set
    ]
    rng.shuffle(all_remaining)
    for pair_id in all_remaining:
        if len(selected) >= target_pairs:
            break
        selected.append(pair_id)
        selected_set.add(pair_id)

    if len(selected) < target_pairs:
        print(
            f"[WARN] Only {len(selected)} complete pairs available; target was {target_pairs}. Using all available pairs."
        )
    return selected


def split_pair_ids(pair_ids: List[str], seed: int) -> Dict[str, List[str]]:
    """Split a list of pair_ids into train/dev/held_out sets, with a seed for reproducibility."""
    rng = random.Random(seed)
    ids = list(pair_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = round(n * 0.50)
    n_dev = round(n * 0.30)
    train = ids[:n_train]
    dev = ids[n_train : n_train + n_dev]
    held_out = ids[n_train + n_dev :]
    return {"train": train, "dev": dev, "held_out": held_out}


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    """Write a list of tasks to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def split_dataset(
    input_files: List[str],
    output_base_dir: str,
    target_tasks: int = 260,
    seed: int = 42,
) -> Dict[str, Any]:
    """Split a dataset into train/dev/held_out sets by source mode."""
    all_tasks: List[Dict[str, Any]] = []
    for path in input_files:
        loaded = load_json_array(path)
        print(f"Loaded {len(loaded)} tasks from {path}")
        all_tasks.extend(loaded)

    groups = group_valid_pairs(all_tasks)
    print(f"Found {len(groups)} complete chosen/rejected pairs before capping.")

    if target_tasks % 2 != 0:
        target_tasks -= 1
        print(f"[WARN] target_tasks must be even for pairs. Using {target_tasks}.")

    selected_pair_ids = choose_pairs_by_source(groups, target_tasks, seed)
    split_ids = split_pair_ids(selected_pair_ids, seed)

    report: Dict[str, Any] = {
        "seed": seed,
        "requested_target_tasks": target_tasks,
        "available_complete_pairs": len(groups),
        "selected_pairs": len(selected_pair_ids),
        "selected_tasks": len(selected_pair_ids) * 2,
        "splits": {},
        "source_mode_counts": {},
    }

    selected_rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for split_name, ids in split_ids.items():
        rows: List[Dict[str, Any]] = []
        for pair_id in ids:
            rows.extend(groups[pair_id])
        # Stable order: chosen before rejected within each pair_id.
        rows.sort(key=lambda r: (r["pair_id"], -int(r["label"])))
        selected_rows_by_split[split_name] = rows
        output_path = os.path.join(output_base_dir, split_name, f"{split_name}.jsonl")
        count = write_jsonl(output_path, rows)
        report["splits"][split_name] = {
            "pairs": len(ids),
            "tasks": count,
            "path": output_path,
            "labels": dict(Counter(row["label"] for row in rows)),
            "source_modes": dict(Counter(row["source_mode"] for row in rows)),
        }
        print(f"Saved {count} tasks ({len(ids)} pairs) to {output_path}")

    all_selected_rows = [
        row for rows in selected_rows_by_split.values() for row in rows
    ]
    report["source_mode_counts"] = dict(
        Counter(row["source_mode"] for row in all_selected_rows)
    )
    report["failure_code_counts"] = dict(
        Counter(
            row.get("failure_code", "None")
            for row in all_selected_rows
            if row["label"] == 0
        )
    )

    os.makedirs(output_base_dir, exist_ok=True)
    report_path = os.path.join(output_base_dir, "dataset_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote dataset report -> {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_INPUT_FILES)
    parser.add_argument("--output-dir", default="tenacious_bench")
    parser.add_argument("--target-tasks", type=int, default=260)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_dataset(args.inputs, args.output_dir, args.target_tasks, args.seed)


if __name__ == "__main__":
    main()
