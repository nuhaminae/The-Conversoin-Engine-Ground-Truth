# src/data_prep/synthetic_pairs.py
#
# Generate synthetic preference pairs.

import json


def build_synthetic_pairs(output_file):
    examples = [
        (
            "Yes, Tuesday at 3 PM works for me.",
            "Perfect, I’ll schedule the meeting for Tuesday at 3 PM and send you a calendar invite.",
            1,
        ),
        ("Yes, Tuesday at 3 PM works for me.", "Okay, thanks for letting me know.", 0),
    ]

    tasks = []
    for prospect, agent, label in examples:
        tasks.append(
            {
                "prospect_input": prospect,
                "agent_output": agent,
                "label": label,
                "failure_mode_tag": "Missed intent: scheduling",
                "metadata": "Synthetic preference pair",
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)


if __name__ == "__main__":
    build_synthetic_pairs("data/tasks/synthetic_pairs.json")
