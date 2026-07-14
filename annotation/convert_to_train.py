#!/usr/bin/env python3
"""
Convert annotated reaction data to TRL messages format for training.

Input: flower_reaction_annotated (from annotate_reaction.py)
Output: TRL SFTTrainer compatible JSON with messages format
"""

import json
import yaml
import argparse
import random
from pathlib import Path


def load_prompts(prompts_path: Path) -> dict:
    """Load prompts from YAML file."""
    with open(prompts_path, "r") as f:
        return yaml.safe_load(f)


def convert_annotation_to_messages(
    annotation: dict,
    system_prompt: str,
    template_prompt: str
) -> dict | None:
    """
    Convert a single annotation to TRL messages format.

    Returns None if annotation failed or is invalid.
    """
    # Skip failed annotations
    if annotation.get("status") != "success":
        return None

    # Skip if no annotation text
    if not annotation.get("annotation"):
        return None

    # Get starting reactants from first step
    steps = annotation.get("steps", [])
    if not steps:
        return None

    starting_reactants = steps[0]["reactants"]

    # Format conditions
    conditions = annotation.get("conditions", [])
    if conditions:
        parts = []
        for c in conditions:
            role = c.get("role", "")
            text = c.get("text", "")
            smiles = c.get("smiles", "")
            if text:
                entry = f"{role}: {text}" if role else text
                if smiles:
                    entry += f" ({smiles})"
                parts.append(entry)
        conditions_str = "; ".join(parts) if parts else "Not specified"
    else:
        conditions_str = "Not specified"

    # Format the user message (template prompt with reactants and conditions)
    user_content = template_prompt.replace("{starting_reactants}", starting_reactants)
    user_content = user_content.replace("{conditions}", conditions_str)

    return {
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": annotation["annotation"]}
        ],
        "chat_template_kwargs": {"enable_thinking": False}
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert annotated reactions to TRL messages format"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input JSON file (annotated reactions from annotate_reaction.py)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSON file (TRL messages format for SFTTrainer)"
    )
    parser.add_argument(
        "--prompts", type=str, default="pathway_student_prompts.yaml",
        help="Path to student prompts YAML file"
    )
    parser.add_argument(
        "--system-key", type=str, default="pathway_student_system_prompt",
        help="Key for system prompt in YAML (default: pathway_student_system_prompt)"
    )
    parser.add_argument(
        "--template-key", type=str, default="pathway_student_template_prompt",
        help="Key for template prompt in YAML (default: pathway_student_template_prompt)"
    )
    parser.add_argument(
        "--shuffle", action="store_true", default=True,
        help="Shuffle the output data (default: True)"
    )
    parser.add_argument(
        "--no-shuffle", dest="shuffle", action="store_false",
        help="Do not shuffle the output data"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling (default: 42)"
    )

    args = parser.parse_args()

    # Load prompts
    prompts_path = Path(args.prompts)
    print(f"Loading prompts from {prompts_path}...")
    prompts = load_prompts(prompts_path)

    system_prompt = prompts.get(args.system_key, "")
    if not system_prompt:
        raise ValueError(f"Could not find '{args.system_key}' in {prompts_path}")

    template_prompt = prompts.get(args.template_key, "")
    if not template_prompt:
        raise ValueError(f"Could not find '{args.template_key}' in {prompts_path}")

    # Load annotated data
    input_path = Path(args.input)
    print(f"Loading annotated data from {input_path}...")
    with open(input_path, "r") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    print(f"Found {len(annotations)} annotations")

    # Convert to TRL messages format
    train_data = []
    skipped = 0

    for ann in annotations:
        result = convert_annotation_to_messages(ann, system_prompt, template_prompt)
        if result:
            train_data.append(result)
        else:
            skipped += 1

    print(f"Converted {len(train_data)} annotations to TRL messages format")
    print(f"Skipped {skipped} failed/invalid annotations")

    # Shuffle data
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(train_data)
        print(f"Shuffled data with seed={args.seed}")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(train_data, f, indent=2)

    print(f"Saved to {output_path}")

    # Print sample
    if train_data:
        print("\n--- Sample (first entry) ---")
        sample = train_data[0]
        messages = sample["messages"]
        print(f"System: {messages[0]['content'][:100]}...")
        print(f"User: {messages[1]['content'][:100]}...")
        print(f"Assistant length: {len(messages[2]['content'])} chars")


if __name__ == "__main__":
    main()
