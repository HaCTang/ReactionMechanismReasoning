#!/usr/bin/env python3
"""Select unlabeled states similar to Fukuyama B/C step-wise states."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from fingerprint_similarity_analysis import canonicalize_state, load_reaction_types, mol_from_smiles


SKIP_BENCHMARK_LABELS = {"reagent", "catalyst", "solvent", "additive", "leaving"}


@dataclass(frozen=True)
class BenchmarkComponent:
    split: str
    case_id: str
    step: int
    reaction_type: str
    state: str
    component: str
    fp: object


def parse_state_line(line: str) -> str | None:
    """Parse one JSON-list line from the cached state file."""
    text = line.strip()
    if not text or text in {"[", "]"}:
        return None
    if text.endswith(","):
        text = text[:-1]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, str) and value else None


def iter_sampled_states(cache_path: Path, sample_size: int, total_states: int, seed: int):
    """Yield a deterministic random sample of cached states without loading all states."""
    sample_size = min(sample_size, total_states)
    selected_indices = set(random.Random(seed).sample(range(total_states), sample_size))
    current_idx = -1
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            state = parse_state_line(line)
            if state is None:
                continue
            current_idx += 1
            if current_idx in selected_indices:
                yield current_idx, state


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows as CSV."""
    fieldnames = [
        "benchmark_split",
        "reaction_type",
        "similarity",
        "benchmark_case_id",
        "benchmark_step",
        "cache_index",
        "unlabeled_component",
        "benchmark_component",
        "unlabeled_state",
        "benchmark_state",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(path: Path, rows: list[dict]) -> None:
    """Write split/type summary counts."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark_split"], row["reaction_type"])].append(float(row["similarity"]))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["benchmark_split", "reaction_type", "n_examples", "mean_similarity", "max_similarity"],
        )
        writer.writeheader()
        for (split, reaction_type), values in sorted(grouped.items()):
            writer.writerow(
                {
                    "benchmark_split": split,
                    "reaction_type": reaction_type,
                    "n_examples": len(values),
                    "mean_similarity": round(sum(values) / len(values), 4),
                    "max_similarity": round(max(values), 4),
                }
            )


def component_fingerprints(state: str, generator, min_heavy_atoms: int):
    """Return fingerprints for chemically substantive components in a state."""
    results = []
    for component in state.split("."):
        mol = mol_from_smiles(component)
        if mol is None:
            continue
        if mol.GetNumHeavyAtoms() < min_heavy_atoms:
            continue
        if not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
            continue
        fp = generator.GetFingerprint(mol)
        results.append((component, fp))
    return results


def load_benchmark_components(repo_root: Path, generator, min_heavy_atoms: int) -> list[BenchmarkComponent]:
    """Load B/C substrate and intermediate components from Fukuyama step-wise states."""
    type_map = load_reaction_types(repo_root / "fukuyama_bench" / "analyze_result" / "reaction_type")
    targets: list[BenchmarkComponent] = []
    for mechanism_path in sorted((repo_root / "fukuyama_bench" / "mechanisms").glob("*/mechanism.json")):
        case_id = mechanism_path.parent.name
        split = case_id[0].upper()
        if split not in {"B", "C"}:
            continue
        data = json.loads(mechanism_path.read_text(encoding="utf-8"))
        for step_idx, step_data in enumerate(data.get("mechanism", []), start=1):
            state = canonicalize_state(step_data.get("reactants", []))
            if not state:
                continue
            reaction_type = type_map.get((case_id, step_idx), "Unknown")
            for reactant in step_data.get("reactants", []):
                if not isinstance(reactant, dict):
                    continue
                label = str(reactant.get("label", "")).strip().lower()
                if label in SKIP_BENCHMARK_LABELS:
                    continue
                component_state = canonicalize_state([reactant])
                if not component_state:
                    continue
                component_fps = component_fingerprints(component_state, generator, min_heavy_atoms)
                for component, fp in component_fps:
                    targets.append(
                        BenchmarkComponent(
                            split=split,
                            case_id=case_id,
                            step=step_idx,
                            reaction_type=reaction_type,
                            state=state,
                            component=component,
                            fp=fp,
                        )
                    )
    return targets


def best_component_match(train_components, benchmark_components):
    """Find the best component-level Tanimoto match."""
    benchmark_fps = [item.fp for item in benchmark_components]
    best = None
    for train_component, train_fp in train_components:
        similarities = DataStructs.BulkTanimotoSimilarity(train_fp, benchmark_fps)
        if not similarities:
            continue
        idx, similarity = max(enumerate(similarities), key=lambda item: item[1])
        if best is None or similarity > best[0]:
            benchmark = benchmark_components[idx]
            best = (similarity, train_component, benchmark, benchmark.component)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Select unlabeled states similar to Fukuyama B/C states.")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--sample_size", type=int, default=200_000)
    parser.add_argument("--max_per_split_type", type=int, default=25)
    parser.add_argument("--min_heavy_atoms", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260530)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cache_path = repo_root / "rebuttal" / "new_figures" / "fingerprint" / "cache" / "unlabeled_train_states.json"
    meta_path = repo_root / "rebuttal" / "new_figures" / "fingerprint" / "cache" / "unlabeled_train_meta.json"
    output_dir = repo_root / "data" / "mech_infer_train_singlestep"
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    total_states = int(meta["unique_states"])

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    benchmark_components = load_benchmark_components(repo_root, generator, args.min_heavy_atoms)

    selected: list[dict] = []
    selected_by_group: dict[tuple[str, str], int] = defaultdict(int)
    max_total = args.max_per_split_type * 2 * 10

    for cache_index, state in iter_sampled_states(cache_path, args.sample_size, total_states, args.seed):
        train_components = component_fingerprints(state, generator, args.min_heavy_atoms)
        if not train_components:
            continue
        match = best_component_match(train_components, benchmark_components)
        if match is None:
            continue
        best_similarity, train_component, benchmark, benchmark_component = match
        if best_similarity <= args.threshold:
            continue

        group = (benchmark.split, benchmark.reaction_type)
        if selected_by_group[group] >= args.max_per_split_type:
            continue

        selected_by_group[group] += 1
        selected.append(
            {
                "source": "johnny-w/flower:unlabeled cached train states",
                "cache_index": cache_index,
                "unlabeled_state": state,
                "unlabeled_component": train_component,
                "similarity": round(best_similarity, 4),
                "benchmark_split": benchmark.split,
                "benchmark_case_id": benchmark.case_id,
                "benchmark_step": benchmark.step,
                "reaction_type": benchmark.reaction_type,
                "benchmark_state": benchmark.state,
                "benchmark_component": benchmark_component,
            }
        )
        if len(selected) >= max_total:
            break

    selected.sort(key=lambda row: (row["benchmark_split"], row["reaction_type"], -row["similarity"]))

    prefix = f"unlabeled_bc_similar_gt_{str(args.threshold).replace('.', 'p')}"
    write_jsonl(output_dir / f"{prefix}.jsonl", selected)
    write_csv(output_dir / f"{prefix}.csv", selected)
    write_summary(output_dir / f"{prefix}_summary.csv", selected)
    print(f"Selected {len(selected)} examples with similarity > {args.threshold}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
