#!/usr/bin/env python3
"""Export johnny-w/flower:rl_v6 to verl parquet under data/verl_v6/.

If the HF config already stores verl-ready rows, writes them through.
Otherwise falls back to building prompts from pathway_student YAML.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset


def load_prompts(path: Path) -> tuple[str, str]:
    prompts = yaml.safe_load(path.read_text(encoding="utf-8"))
    return (
        prompts["pathway_student_system_prompt"].strip(),
        prompts["pathway_student_template_prompt"].strip(),
    )


def row_to_record(sample: dict, system_prompt: str, template_prompt: str) -> dict | None:
    if "prompt" in sample and "reward_model" in sample:
        return {
            "data_source": sample.get("data_source", "flower_rl_v6"),
            "prompt": sample["prompt"],
            "ability": sample.get("ability", "mechanism"),
            "reward_model": sample["reward_model"],
            "extra_info": sample.get("extra_info", {}),
        }

    steps = sample.get("steps")
    if isinstance(steps, str):
        steps = json.loads(steps)
    if not steps:
        return None
    starting = steps[0]["reactants"]
    user = template_prompt.replace("{starting_reactants}", starting).replace(
        "{conditions}", sample.get("conditions", "Not specified")
    )
    gt = [{"step_id": s.get("step_id", i + 1), "products": s["products"]} for i, s in enumerate(steps)]
    return {
        "data_source": "flower_rl_v6",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        "ability": "mechanism",
        "reward_model": {"ground_truth": json.dumps(gt)},
        "extra_info": {"reaction_id": sample.get("reaction_id")},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="johnny-w/flower")
    ap.add_argument("--config", default="rl_v6")
    ap.add_argument("--out", type=Path, default=Path("data/verl_v6"))
    ap.add_argument(
        "--prompts",
        type=Path,
        default=Path("train/pathway_student_prompts.yaml"),
    )
    args = ap.parse_args()

    system_prompt, template_prompt = load_prompts(args.prompts)
    ds = load_dataset(args.dataset, args.config)
    args.out.mkdir(parents=True, exist_ok=True)

    for split_name, split in ds.items():
        records = []
        for sample in split:
            rec = row_to_record(sample, system_prompt, template_prompt)
            if rec is not None:
                records.append(rec)
        out_name = "train.parquet" if split_name in ("train", "train_data") else f"{split_name}.parquet"
        if split_name == "validation":
            out_name = "validation.parquet"
        path = args.out / out_name
        pd.DataFrame(records).to_parquet(path, index=False)
        print(f"{split_name}: {len(records)} -> {path}")


if __name__ == "__main__":
    main()
