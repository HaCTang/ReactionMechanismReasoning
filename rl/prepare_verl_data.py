#!/usr/bin/env python3
"""
Prepare verl-compatible parquet data from the johnny-w/flower "unlabeled" subset.

Each reaction is one training sample (reaction-wise, NOT step-wise).
The model receives starting reactants and must predict the complete multi-step pathway.

Output columns (verl format):
  - data_source: str
  - prompt: list[dict] (chat messages)
  - ability: str
  - reward_model.ground_truth: str (JSON string of ground truth steps)
  - extra_info: dict

Usage:
    python rl/prepare_verl_data.py
    python rl/prepare_verl_data.py --max_steps 20 --output_dir data/verl_flower/
"""

import json
import argparse
from pathlib import Path

import yaml
import pandas as pd
from datasets import load_dataset


def load_prompts(prompts_path: str) -> tuple[str, str]:
    """Load system and template prompts from YAML."""
    with open(prompts_path, "r") as f:
        prompts = yaml.safe_load(f)
    system_prompt = prompts["pathway_student_system_prompt"].strip()
    template_prompt = prompts["pathway_student_template_prompt"].strip()
    return system_prompt, template_prompt


def build_prompt_messages(system_prompt: str, template_prompt: str, starting_reactants: str, conditions: str = "Not specified") -> list[dict]:
    """Build chat messages for a single reaction."""
    user_content = template_prompt.replace("{starting_reactants}", starting_reactants)
    user_content = user_content.replace("{conditions}", conditions)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def process_split(dataset, system_prompt: str, template_prompt: str, max_steps: int) -> pd.DataFrame:
    """Convert a HuggingFace dataset split to verl parquet format."""
    records = []
    skipped = 0

    for sample in dataset:
        steps = json.loads(sample["steps"])
        num_steps = sample["num_steps"]

        # Filter out extreme outliers
        if num_steps > max_steps:
            skipped += 1
            continue

        if not steps:
            skipped += 1
            continue

        # Starting reactants = reactants of the first step
        starting_reactants = steps[0]["reactants"]

        # Build ground truth: list of step products for reward computation
        ground_truth_steps = []
        for step in steps:
            ground_truth_steps.append({
                "step_id": step["step_id"],
                "products": step["products"],
            })

        prompt = build_prompt_messages(system_prompt, template_prompt, starting_reactants)

        records.append({
            "data_source": "flower_unlabeled",
            "prompt": prompt,
            "ability": "reaction_mechanism",
            "reward_model": {
                "ground_truth": json.dumps(ground_truth_steps),
            },
            "extra_info": {
                "reaction_id": sample["reaction_id"],
                "num_steps": num_steps,
            },
        })

    print(f"  Processed {len(records)} samples, skipped {skipped} (>{max_steps} steps)")
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Prepare verl parquet data from flower dataset")
    parser.add_argument("--output_dir", type=str, default="data/verl_flower/",
                        help="Output directory for parquet files")
    parser.add_argument("--prompts", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "train" / "pathway_student_prompts.yaml"),
                        help="Path to student prompts YAML")
    parser.add_argument("--max_steps", type=int, default=20,
                        help="Max steps per reaction (filter outliers)")
    parser.add_argument("--dataset_name", type=str, default="johnny-w/flower",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default="unlabeled",
                        help="HuggingFace dataset config")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    print(f"Loading prompts from {args.prompts}...")
    system_prompt, template_prompt = load_prompts(args.prompts)
    print(f"  System prompt: {len(system_prompt)} chars")
    print(f"  Template prompt: {len(template_prompt)} chars")

    # Load dataset
    print(f"Loading dataset {args.dataset_name} ({args.dataset_config})...")
    ds = load_dataset(args.dataset_name, args.dataset_config)

    for split_name in ["train", "validation"]:
        if split_name not in ds:
            print(f"  Skipping {split_name} (not found)")
            continue

        print(f"\nProcessing {split_name} split ({len(ds[split_name])} samples)...")
        df = process_split(ds[split_name], system_prompt, template_prompt, args.max_steps)

        out_path = output_dir / f"{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved {len(df)} samples to {out_path}")

    # Print summary
    for split_name in ["train", "validation"]:
        path = output_dir / f"{split_name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            print(f"\n{split_name}: {len(df)} samples, columns: {df.columns.tolist()}")
            # Show a sample prompt
            sample_prompt = df.iloc[0]["prompt"]
            if isinstance(sample_prompt, str):
                sample_prompt = json.loads(sample_prompt)
            print(f"  Sample system prompt: {sample_prompt[0]['content'][:80]}...")
            print(f"  Sample user prompt: {sample_prompt[1]['content'][:80]}...")


if __name__ == "__main__":
    main()
