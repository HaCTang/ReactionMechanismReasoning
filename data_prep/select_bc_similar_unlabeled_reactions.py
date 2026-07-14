#!/usr/bin/env python3
"""Select complete unlabeled reactions similar to Fukuyama B/C step-wise states."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from fingerprint_similarity_analysis import (
    canonicalize_state,
    load_hf_token,
    mol_from_smiles,
)
from select_bc_similar_unlabeled_states import load_benchmark_components

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    load_dataset = None
    tqdm = None


def component_fingerprints(state: str, generator, min_heavy_atoms: int):
    """Return fingerprints for substantial organic components in a state."""
    results = []
    for component in state.split("."):
        mol = mol_from_smiles(component)
        if mol is None:
            continue
        if mol.GetNumHeavyAtoms() < min_heavy_atoms:
            continue
        if not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
            continue
        results.append((component, generator.GetFingerprint(mol)))
    return results


def parse_steps(value: Any) -> list[dict[str, Any]]:
    """Parse a Flower steps column."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def find_step_matches(row: dict[str, Any], benchmark_components, threshold: float, min_heavy_atoms: int) -> list[dict[str, Any]]:
    """Find all step-level component matches above the threshold for one reaction."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    benchmark_fps = [item.fp for item in benchmark_components]
    steps = parse_steps(row.get("steps"))
    matches: list[dict[str, Any]] = []

    for step_idx, step in enumerate(steps, start=1):
        state = canonicalize_state(step.get("reactants"))
        if not state:
            continue
        components = component_fingerprints(state, generator, min_heavy_atoms)
        for component, fp in components:
            similarities = DataStructs.BulkTanimotoSimilarity(fp, benchmark_fps)
            if not similarities:
                continue
            best_idx, best_similarity = max(enumerate(similarities), key=lambda item: item[1])
            if best_similarity <= threshold:
                continue
            benchmark = benchmark_components[best_idx]
            matches.append(
                {
                    "similarity": round(best_similarity, 4),
                    "unlabeled_step": step_idx,
                    "unlabeled_state": state,
                    "unlabeled_component": component,
                    "benchmark_split": benchmark.split,
                    "benchmark_case_id": benchmark.case_id,
                    "benchmark_step": benchmark.step,
                    "reaction_type": benchmark.reaction_type,
                    "benchmark_state": benchmark.state,
                    "benchmark_component": benchmark.component,
                }
            )
    return matches


def compact_row(row: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep the complete reaction plus match metadata."""
    steps = parse_steps(row.get("steps"))
    best = max(matches, key=lambda item: item["similarity"])
    return {
        "source": "johnny-w/flower:unlabeled",
        "reaction_id": row.get("reaction_id"),
        "num_steps": row.get("num_steps") or len(steps),
        "best_similarity": best["similarity"],
        "best_benchmark_split": best["benchmark_split"],
        "best_reaction_type": best["reaction_type"],
        "best_benchmark_case_id": best["benchmark_case_id"],
        "best_benchmark_step": best["benchmark_step"],
        "matches": sorted(matches, key=lambda item: -item["similarity"]),
        "steps": steps,
        "raw_row": row,
    }


def trim_grouped(records: list[dict[str, Any]], max_per_split_type: int) -> list[dict[str, Any]]:
    """Keep top-N records in every B/C x reaction-type group."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group = (record["best_benchmark_split"], record["best_reaction_type"])
        grouped[group].append(record)

    trimmed: list[dict[str, Any]] = []
    for group_records in grouped.values():
        group_records.sort(key=lambda item: (-float(item["best_similarity"]), str(item.get("reaction_id", ""))))
        trimmed.extend(group_records[:max_per_split_type])
    trimmed.sort(key=lambda item: (item["best_benchmark_split"], item["best_reaction_type"], -float(item["best_similarity"])))
    return trimmed


def process_chunk(
    rows: list[dict[str, Any]],
    benchmark_components,
    threshold: float,
    min_heavy_atoms: int,
    max_per_split_type: int,
) -> list[dict[str, Any]]:
    """Process a chunk of reactions and return trimmed candidates."""
    candidates = []
    for row in rows:
        matches = find_step_matches(row, benchmark_components, threshold, min_heavy_atoms)
        if matches:
            candidates.append(compact_row(row, matches))
    return trim_grouped(candidates, max_per_split_type)


def iter_chunks(dataset, chunk_size: int):
    """Yield row chunks from a HuggingFace dataset."""
    chunk = []
    for row in dataset:
        chunk.append(dict(row))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write selected complete reactions as JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact CSV index for selected reactions."""
    fieldnames = [
        "reaction_id",
        "num_steps",
        "best_similarity",
        "best_benchmark_split",
        "best_reaction_type",
        "best_benchmark_case_id",
        "best_benchmark_step",
        "n_matches",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "reaction_id": row.get("reaction_id"),
                    "num_steps": row.get("num_steps"),
                    "best_similarity": row.get("best_similarity"),
                    "best_benchmark_split": row.get("best_benchmark_split"),
                    "best_reaction_type": row.get("best_reaction_type"),
                    "best_benchmark_case_id": row.get("best_benchmark_case_id"),
                    "best_benchmark_step": row.get("best_benchmark_step"),
                    "n_matches": len(row.get("matches", [])),
                }
            )


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write selected count summary by B/C split and reaction type."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["best_benchmark_split"], row["best_reaction_type"])].append(float(row["best_similarity"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["benchmark_split", "reaction_type", "n_reactions", "mean_best_similarity", "max_best_similarity"],
        )
        writer.writeheader()
        for (split, reaction_type), values in sorted(grouped.items()):
            writer.writerow(
                {
                    "benchmark_split": split,
                    "reaction_type": reaction_type,
                    "n_reactions": len(values),
                    "mean_best_similarity": round(sum(values) / len(values), 4),
                    "max_best_similarity": round(max(values), 4),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select complete unlabeled reactions similar to B/C benchmark states.")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--max_per_split_type", type=int, default=20)
    parser.add_argument("--min_heavy_atoms", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    if load_dataset is None:
        raise SystemExit("Missing dependency: datasets. Install it in the active environment.")

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "data" / "mech_infer_train_singlestep"
    output_dir.mkdir(parents=True, exist_ok=True)
    token = load_hf_token(repo_root)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    benchmark_components = load_benchmark_components(repo_root, generator, args.min_heavy_atoms)
    dataset = load_dataset("johnny-w/flower", "unlabeled", split="train", token=token)
    if args.max_rows is not None:
        dataset = dataset.select(range(min(args.max_rows, len(dataset))))

    selected: list[dict[str, Any]] = []
    total_chunks = math.ceil(len(dataset) / args.chunk_size)
    chunk_iter = iter_chunks(dataset, args.chunk_size)
    progress = tqdm(total=total_chunks, desc="Unlabeled chunks", unit="chunk", dynamic_ncols=True) if tqdm else None

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = set()
        max_inflight = max(1, args.num_workers * 2)

        def submit_chunk(chunk_rows):
            futures.add(
                executor.submit(
                    process_chunk,
                    chunk_rows,
                    benchmark_components,
                    args.threshold,
                    args.min_heavy_atoms,
                    args.max_per_split_type,
                )
            )

        for chunk in chunk_iter:
            submit_chunk(chunk)
            if len(futures) >= max_inflight:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    selected.extend(future.result())
                    selected = trim_grouped(selected, args.max_per_split_type)
                    if progress:
                        progress.update(1)

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                selected.extend(future.result())
                selected = trim_grouped(selected, args.max_per_split_type)
                if progress:
                    progress.update(1)
    if progress:
        progress.close()

    selected = trim_grouped(selected, args.max_per_split_type)
    prefix = f"unlabeled_bc_similar_complete_gt_{str(args.threshold).replace('.', 'p')}"
    write_jsonl(output_dir / f"{prefix}.jsonl", selected)
    write_csv(output_dir / f"{prefix}.csv", selected)
    write_summary(output_dir / f"{prefix}_summary.csv", selected)
    print(f"Selected {len(selected)} complete reactions with component similarity > {args.threshold}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
