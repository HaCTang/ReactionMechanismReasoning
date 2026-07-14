#!/usr/bin/env python3
"""Compare Fukuyama step-wise states with Flower training subsets by fingerprints."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_CONFIGS = [
    "sft_hq",
    "rl_v6",
    "sft",
    "sft_hq_v2",
    "sft_named_reactions",
    "annotated",
    "unlabeled",
    "rl_v7",
    "rl_pericyclic",
    "rl_named_reactions",
]


REACTION_TYPES = [
    "Addition",
    "Cleavage",
    "Coordination",
    "Electron Transfer",
    "Elimination",
    "Pericyclic",
    "Proton Transfer",
    "Radical",
    "Rearrangement",
    "Substitution",
]


SMILES_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(?:\[?[A-Z][A-Za-z0-9@+\-\[\]\(\)=#$:/\\.%]*)(?:\.[A-Za-z0-9@+\-\[\]\(\)=#$:/\\.%]+)*")


@dataclass(frozen=True)
class BenchmarkStep:
    split: str
    case_id: str
    step: int
    reaction_type: str
    state: str


def normalize_reaction_type(value: str) -> str:
    """Normalize reaction-type labels to the 10 coarse classes."""
    label = " ".join(str(value).replace("_", " ").split()).title()
    aliases = {
        "Proton Transfer": "Proton Transfer",
        "Electron Transfer": "Electron Transfer",
    }
    return aliases.get(label, label)


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string after removing atom-map IDs."""
    cleaned = re.sub(r":\d+", "", smiles.strip())
    if not cleaned:
        return None
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def canonicalize_state(value: Any) -> str | None:
    """Canonicalize a multi-component molecular state."""
    if value is None:
        return None
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("smiles", "")
            parts.extend(str(item).split("."))
    else:
        parts = str(value).split(".")

    canonical: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        mol = mol_from_smiles(part)
        if mol is None:
            continue
        canonical.append(Chem.MolToSmiles(mol, canonical=True))
    if not canonical:
        return None
    return ".".join(sorted(canonical))


def state_fingerprint(state: str, generator: rdFingerprintGenerator.FingerprintGenerator64) -> DataStructs.ExplicitBitVect | None:
    """Create a union Morgan fingerprint for a multi-component state."""
    fp = None
    for part in state.split("."):
        mol = mol_from_smiles(part)
        if mol is None:
            continue
        mol_fp = generator.GetFingerprint(mol)
        if fp is None:
            fp = mol_fp
        else:
            fp |= mol_fp
    return fp


def parse_json_value(value: Any) -> Any:
    """Parse JSON strings when needed."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def load_reaction_types(type_dir: Path) -> dict[tuple[str, int], str]:
    """Load step-level reaction classes for Fukuyama cases."""
    type_map: dict[tuple[str, int], str] = {}
    for path in sorted(type_dir.glob("*.csv")):
        case_id = path.stem
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    step = int(row["step"])
                except (KeyError, TypeError, ValueError):
                    continue
                reaction_type = normalize_reaction_type(row.get("type", "Unknown"))
                type_map[(case_id, step)] = reaction_type
    return type_map


def load_benchmark_steps(repo_root: Path) -> list[BenchmarkStep]:
    """Load Fukuyama benchmark elementary-step states."""
    mechanisms_dir = repo_root / "fukuyama_bench" / "mechanisms"
    type_map = load_reaction_types(repo_root / "fukuyama_bench" / "analyze_result" / "reaction_type")
    steps: list[BenchmarkStep] = []

    for mechanism_path in sorted(mechanisms_dir.glob("*/mechanism.json")):
        case_id = mechanism_path.parent.name
        split = case_id[0].upper()
        if split not in {"A", "B", "C"}:
            continue
        data = json.loads(mechanism_path.read_text(encoding="utf-8"))
        mechanism = data.get("mechanism", [])
        for idx, step_data in enumerate(mechanism, start=1):
            state = canonicalize_state(step_data.get("reactants", []))
            if not state:
                continue
            reaction_type = type_map.get((case_id, idx), "Unknown")
            if reaction_type not in REACTION_TYPES:
                reaction_type = "Unknown"
            steps.append(BenchmarkStep(split, case_id, idx, reaction_type, state))
    return steps


def extract_json_blocks(text: str) -> Iterable[Any]:
    """Yield JSON objects or arrays embedded in text."""
    if not text:
        return
    blocks = re.findall(r"```json\s*([\s\S]*?)\s*```", text)
    blocks.extend(re.findall(r"```\s*([\s\S]*?)\s*```", text))
    for block in blocks:
        try:
            yield json.loads(block)
        except json.JSONDecodeError:
            continue


def extract_starting_reactants_from_messages(messages: Any) -> str | None:
    """Extract the starting reactant state from chat messages."""
    messages = parse_json_value(messages)
    if not isinstance(messages, list):
        return None
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = str(msg.get("content", ""))
        match = re.search(r"Starting Reactants:\s*(.+?)(?:\n\s*\n|Reaction Conditions:|$)", content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"Reactants:\s*(.+?)(?:\n\s*\n|$)", content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_assistant_steps_from_messages(messages: Any) -> list[dict[str, Any]]:
    """Extract product steps from a chat-formatted assistant answer."""
    messages = parse_json_value(messages)
    if not isinstance(messages, list):
        return []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for block in extract_json_blocks(str(msg.get("content", ""))):
            if isinstance(block, list):
                return [item for item in block if isinstance(item, dict)]
            if isinstance(block, dict) and isinstance(block.get("steps"), list):
                return [item for item in block["steps"] if isinstance(item, dict)]
    return []


def products_from_step(step: dict[str, Any]) -> Any:
    """Return product strings from one structured step."""
    if "products" in step:
        return step["products"]
    if "product_smiles" in step:
        return step["product_smiles"]
    if "next_state" in step:
        return step["next_state"]
    return None


def reconstruct_states_from_products(starting_state: Any, product_steps: list[dict[str, Any]]) -> list[str]:
    """Build step-wise current states from a starting state and product list."""
    states: list[str] = []
    current = canonicalize_state(starting_state)
    if not current:
        return states
    for step in product_steps:
        states.append(current)
        next_state = canonicalize_state(products_from_step(step))
        if not next_state:
            break
        current = next_state
    return states


def extract_states_from_structured_steps(steps_value: Any) -> list[str]:
    """Extract reactant states from a structured steps column."""
    steps_value = parse_json_value(steps_value)
    if not isinstance(steps_value, list):
        return []
    states: list[str] = []
    for step in steps_value:
        if not isinstance(step, dict):
            continue
        state = canonicalize_state(step.get("reactants"))
        if state:
            states.append(state)
    if states:
        return states
    if steps_value:
        return reconstruct_states_from_products(None, steps_value)
    return []


def extract_states_from_reward(row: dict[str, Any]) -> list[str]:
    """Extract step-wise states from verl-style reward data."""
    reward_model = parse_json_value(row.get("reward_model"))
    if not isinstance(reward_model, dict):
        return []
    ground_truth = parse_json_value(reward_model.get("ground_truth"))
    if not isinstance(ground_truth, list):
        return []
    starting_state = extract_starting_reactants_from_messages(row.get("prompt"))
    return reconstruct_states_from_products(starting_state, [item for item in ground_truth if isinstance(item, dict)])


def extract_states_from_text(row: dict[str, Any]) -> list[str]:
    """Fallback text extraction for configs without structured state fields."""
    text_parts = []
    for value in row.values():
        if isinstance(value, (str, list, dict)):
            text_parts.append(str(value))
    states: set[str] = set()
    for match in SMILES_PATTERN.finditer(" ".join(text_parts)):
        state = canonicalize_state(match.group(0))
        if state:
            states.add(state)
    return sorted(states)


def extract_training_states(row: dict[str, Any]) -> list[str]:
    """Extract candidate step-wise molecular states from one training row."""
    if "steps" in row:
        states = extract_states_from_structured_steps(row["steps"])
        if states:
            return states
    if "messages" in row:
        starting_state = extract_starting_reactants_from_messages(row["messages"])
        product_steps = extract_assistant_steps_from_messages(row["messages"])
        states = reconstruct_states_from_products(starting_state, product_steps)
        if states:
            return states
    if "reward_model" in row:
        states = extract_states_from_reward(row)
        if states:
            return states
    return extract_states_from_text(row)


def load_hf_token(repo_root: Path) -> str | None:
    """Load a HuggingFace token from the environment or local key files."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip()
    for path in [repo_root / "hf_token.key", repo_root / "hf.key", repo_root.parent / "hf_token.key", repo_root.parent / "hf.key"]:
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
    return None


def iter_hf_rows(config: str, split: str, token: str | None) -> Iterable[dict[str, Any]]:
    """Stream rows from one HuggingFace config."""
    from datasets import load_dataset

    kwargs = {"token": token} if token else {}
    dataset = load_dataset("johnny-w/flower", config, split=split, streaming=True, **kwargs)
    for row in dataset:
        yield dict(row)


def load_or_build_training_states(
    output_dir: Path,
    config: str,
    split: str,
    max_records: int | None,
    token: str | None,
    show_progress: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """Extract and cache unique training states for one config."""
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{max_records}" if max_records else ""
    cache_path = cache_dir / f"{config}_{split}{suffix}_states.json"
    meta_path = cache_dir / f"{config}_{split}{suffix}_meta.json"
    if cache_path.exists() and meta_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8")), json.loads(meta_path.read_text(encoding="utf-8"))

    states: set[str] = set()
    rows_seen = 0
    rows_with_states = 0
    row_iter = iter_hf_rows(config, split, token)
    if tqdm is not None and show_progress:
        row_iter = tqdm(row_iter, desc=f"{config} rows", unit="row", dynamic_ncols=True)

    for row in row_iter:
        rows_seen += 1
        row_states = extract_training_states(row)
        if row_states:
            rows_with_states += 1
            states.update(row_states)
        if max_records and rows_seen >= max_records:
            break

    sorted_states = sorted(states)
    meta = {
        "config": config,
        "split": split,
        "rows_seen": rows_seen,
        "rows_with_states": rows_with_states,
        "unique_states": len(sorted_states),
        "max_records": max_records,
    }
    cache_path.write_text(json.dumps(sorted_states, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return sorted_states, meta


def summarize_config(
    config: str,
    split: str,
    max_records: int | None,
    output_dir_str: str,
    token: str | None,
    benchmark_steps: list[BenchmarkStep],
    show_row_progress: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Process one training config and return all summary rows."""
    output_dir = Path(output_dir_str)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_states, meta = load_or_build_training_states(
        output_dir,
        config,
        split,
        max_records,
        token,
        show_progress=show_row_progress,
    )
    similarities = compute_nearest_similarities(benchmark_steps, train_states, generator)

    overall_rows: list[dict[str, Any]] = []
    by_type_rows: list[dict[str, Any]] = []

    for benchmark_split in ["A", "B", "C", "All"]:
        indices = [
            idx
            for idx, step in enumerate(benchmark_steps)
            if benchmark_split == "All" or step.split == benchmark_split
        ]
        values = [similarities[idx] for idx in indices if not math.isnan(similarities[idx])]
        summary = summarize(values)
        overall_rows.append(
            {
                "training_config": config,
                "benchmark_split": benchmark_split,
                **{key: round(value, 4) if isinstance(value, float) else value for key, value in summary.items()},
            }
        )

    for benchmark_split in ["A", "B", "C", "All"]:
        for reaction_type in REACTION_TYPES:
            indices = [
                idx
                for idx, step in enumerate(benchmark_steps)
                if (benchmark_split == "All" or step.split == benchmark_split) and step.reaction_type == reaction_type
            ]
            values = [similarities[idx] for idx in indices if not math.isnan(similarities[idx])]
            summary = summarize(values)
            by_type_rows.append(
                {
                    "training_config": config,
                    "benchmark_split": benchmark_split,
                    "reaction_type": reaction_type,
                    **{key: round(value, 4) if isinstance(value, float) else value for key, value in summary.items()},
                }
            )

    return overall_rows, by_type_rows, meta


def percentile(values: list[float], pct: float) -> float:
    """Compute a percentile with linear interpolation."""
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, Any]:
    """Summarize a list of nearest-neighbor similarities."""
    if not values:
        return {
            "n_steps": 0,
            "mean_max_tanimoto": math.nan,
            "median_max_tanimoto": math.nan,
            "p90_max_tanimoto": math.nan,
        }
    return {
        "n_steps": len(values),
        "mean_max_tanimoto": mean(values),
        "median_max_tanimoto": median(values),
        "p90_max_tanimoto": percentile(values, 0.9),
    }


def compute_nearest_similarities(
    benchmark_steps: list[BenchmarkStep],
    train_states: list[str],
    generator: rdFingerprintGenerator.FingerprintGenerator64,
) -> list[float]:
    """Compute nearest training-state similarity for each benchmark step."""
    train_fps = []
    for state in train_states:
        fp = state_fingerprint(state, generator)
        if fp is not None:
            train_fps.append(fp)
    if not train_fps:
        return [math.nan for _ in benchmark_steps]

    similarities: list[float] = []
    for step in benchmark_steps:
        query_fp = state_fingerprint(step.state, generator)
        if query_fp is None:
            similarities.append(math.nan)
            continue
        scores = DataStructs.BulkTanimotoSimilarity(query_fp, train_fps)
        similarities.append(max(scores) if scores else math.nan)
    return similarities


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(value: Any) -> str:
    """Format floats for LaTeX tables."""
    if isinstance(value, float):
        if math.isnan(value):
            return "--"
        return f"{value:.3f}"
    return str(value)


def write_latex_table(path: Path, rows: list[dict[str, Any]], columns: list[str], caption: str) -> None:
    """Write a compact LaTeX tabular."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(columns).replace("_", r"\_") + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(format_float(row.get(col, "")) for col in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(output_dir: Path, overall_rows: list[dict[str, Any]], by_type_rows: list[dict[str, Any]]) -> None:
    """Write summary figures."""
    for plot_name in [
        "fingerprint_similarity_overall_heatmap.png",
        "fingerprint_similarity_overall_heatmap.svg",
        "fingerprint_similarity_by_type_heatmap.png",
        "fingerprint_similarity_by_type_heatmap.svg",
    ]:
        plot_path = output_dir / plot_name
        if plot_path.exists():
            plot_path.unlink()
    if not overall_rows or not by_type_rows:
        return
    import matplotlib.pyplot as plt

    configs = DEFAULT_CONFIGS
    splits = ["A", "B", "C", "All"]
    matrix = []
    for config in configs:
        row = []
        for split in splits:
            match = next((item for item in overall_rows if item["training_config"] == config and item["benchmark_split"] == split), None)
            row.append(float(match["mean_max_tanimoto"]) if match else math.nan)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(splits)), splits)
    ax.set_yticks(range(len(configs)), configs)
    ax.set_xlabel("Fukuyama split")
    ax.set_ylabel("Flower training config")
    ax.set_title("Mean nearest-neighbor Morgan similarity")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if not math.isnan(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.55 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="Mean max Tanimoto")
    fig.tight_layout()
    fig.savefig(output_dir / "fingerprint_similarity_overall_heatmap.png", dpi=300)
    fig.savefig(output_dir / "fingerprint_similarity_overall_heatmap.svg")
    plt.close(fig)

    all_type_rows = [row for row in by_type_rows if row["benchmark_split"] == "All"]
    selected_configs = [cfg for cfg in configs if any(row["training_config"] == cfg for row in all_type_rows)]
    if not selected_configs:
        return
    type_matrix = []
    for config in selected_configs:
        row_values = []
        for reaction_type in REACTION_TYPES:
            match = next(
                (
                    item
                    for item in all_type_rows
                    if item["training_config"] == config and item["reaction_type"] == reaction_type
                ),
                None,
            )
            row_values.append(float(match["mean_max_tanimoto"]) if match else math.nan)
        type_matrix.append(row_values)

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(type_matrix, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(REACTION_TYPES)), REACTION_TYPES, rotation=45, ha="right")
    ax.set_yticks(range(len(selected_configs)), selected_configs)
    ax.set_xlabel("Reaction type")
    ax.set_ylabel("Flower training config")
    ax.set_title("Mean nearest-neighbor similarity by reaction type")
    fig.colorbar(im, ax=ax, label="Mean max Tanimoto")
    fig.tight_layout()
    fig.savefig(output_dir / "fingerprint_similarity_by_type_heatmap.png", dpi=300)
    fig.savefig(output_dir / "fingerprint_similarity_by_type_heatmap.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Flower/Fukuyama fingerprint similarity.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(4, max(1, (os.cpu_count() or 2) - 1)),
        help="Number of parallel training configs to process.",
    )
    parser.add_argument(
        "--no_row_progress",
        action="store_true",
        help="Disable per-config row progress bars.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(__file__).resolve().parent / "fingerprint"
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_token = load_hf_token(repo_root)

    benchmark_steps = load_benchmark_steps(repo_root)

    overall_rows: list[dict[str, Any]] = []
    by_type_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []

    worker_count = max(1, min(args.num_workers, len(args.configs)))
    show_row_progress = not args.no_row_progress
    print(f"Processing {len(args.configs)} configs with {worker_count} worker(s)...")

    if worker_count == 1:
        config_iter = args.configs
        if tqdm is not None:
            config_iter = tqdm(config_iter, desc="Configs", unit="config", dynamic_ncols=True)
        for config in config_iter:
            try:
                config_overall, config_by_type, meta = summarize_config(
                    config,
                    args.split,
                    args.max_records,
                    str(output_dir),
                    hf_token,
                    benchmark_steps,
                    show_row_progress=show_row_progress,
                )
                overall_rows.extend(config_overall)
                by_type_rows.extend(config_by_type)
                meta_rows.append(meta)
            except Exception as exc:
                meta_rows.append({"config": config, "split": args.split, "error": f"{type(exc).__name__}: {exc}"})
                print(f"  Skipped {config}: {type(exc).__name__}: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    summarize_config,
                    config,
                    args.split,
                    args.max_records,
                    str(output_dir),
                    hf_token,
                    benchmark_steps,
                    show_row_progress,
                ): config
                for config in args.configs
            }
            future_iter = as_completed(futures)
            if tqdm is not None:
                future_iter = tqdm(future_iter, total=len(futures), desc="Configs", unit="config", dynamic_ncols=True)
            for future in future_iter:
                config = futures[future]
                try:
                    config_overall, config_by_type, meta = future.result()
                    overall_rows.extend(config_overall)
                    by_type_rows.extend(config_by_type)
                    meta_rows.append(meta)
                except Exception as exc:
                    meta_rows.append({"config": config, "split": args.split, "error": f"{type(exc).__name__}: {exc}"})
                    print(f"  Skipped {config}: {type(exc).__name__}: {exc}")

    write_csv(
        output_dir / "fingerprint_similarity_overall.csv",
        overall_rows,
        ["training_config", "benchmark_split", "n_steps", "mean_max_tanimoto", "median_max_tanimoto", "p90_max_tanimoto"],
    )
    write_csv(
        output_dir / "fingerprint_similarity_by_reaction_type.csv",
        by_type_rows,
        [
            "training_config",
            "benchmark_split",
            "reaction_type",
            "n_steps",
            "mean_max_tanimoto",
            "median_max_tanimoto",
            "p90_max_tanimoto",
        ],
    )
    write_csv(
        output_dir / "fingerprint_similarity_dataset_metadata.csv",
        meta_rows,
        sorted({key for row in meta_rows for key in row.keys()}),
    )

    write_latex_table(
        output_dir / "fingerprint_similarity_overall.tex",
        overall_rows,
        ["training_config", "benchmark_split", "n_steps", "mean_max_tanimoto", "median_max_tanimoto", "p90_max_tanimoto"],
        "Nearest-neighbor Morgan fingerprint similarity between Fukuyama step-wise states and Flower training subsets.",
    )
    all_rows = [row for row in by_type_rows if row["benchmark_split"] == "All"]
    write_latex_table(
        output_dir / "fingerprint_similarity_by_reaction_type_all.tex",
        all_rows,
        ["training_config", "reaction_type", "n_steps", "mean_max_tanimoto", "median_max_tanimoto", "p90_max_tanimoto"],
        "Nearest-neighbor Morgan fingerprint similarity by reaction type, aggregated over Sets A/B/C.",
    )
    write_plots(output_dir, overall_rows, by_type_rows)
    print(f"Wrote fingerprint similarity analysis to {output_dir}")


if __name__ == "__main__":
    main()
