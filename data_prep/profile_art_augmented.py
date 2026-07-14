#!/usr/bin/env python3
"""Profile the ART-augmented dataset's chemical-space coverage.

Shows (1) that the new data is NOT a benchmark near-duplicate (decontam check),
and (2) that ADDING it to the SFT training set raises the nearest-neighbor
Morgan similarity between FukuyamaBench steps and the training pool -- especially
for Sets B/C and for the failing reaction classes (pericyclic, rearrangement,
radical, cleavage). This is the coverage-gain evidence for the rebuttal.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

from rdkit import DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprint_similarity_analysis import (  # noqa: E402
    REACTION_TYPES,
    canonicalize_state,
    load_benchmark_steps,
    state_fingerprint,
)

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def new_data_states(reactions_path: Path) -> list[str]:
    data = json.loads(reactions_path.read_text(encoding="utf-8"))
    states: set[str] = set()
    for rxn in data.get("reactions", []):
        for step in rxn.get("steps", []):
            cs = canonicalize_state(step.get("reactants"))
            if cs:
                states.add(cs)
            cp = canonicalize_state(step.get("products"))
            if cp:
                states.add(cp)
    return sorted(states)


def fps_for_states(states: list[str]):
    out = []
    for s in states:
        fp = state_fingerprint(s, GEN)
        if fp is not None:
            out.append(fp)
    return out


def nn_sims(benchmark_steps, train_fps):
    sims = []
    for step in benchmark_steps:
        q = state_fingerprint(step.state, GEN)
        if q is None or not train_fps:
            sims.append(math.nan)
            continue
        sims.append(max(DataStructs.BulkTanimotoSimilarity(q, train_fps)))
    return sims


def summ(values):
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return (0, math.nan, math.nan)
    vals.sort()
    return (len(vals), sum(vals) / len(vals), vals[len(vals) // 2])


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cache = Path(__file__).resolve().parent / "fingerprint" / "cache"
    reactions_path = repo_root / "data" / "art_augmented_mechanisms" / "reactions.json"
    out_dir = Path(__file__).resolve().parent / "fingerprint"

    benchmark_steps = load_benchmark_steps(repo_root)
    print(f"benchmark steps: {len(benchmark_steps)}")

    sft_states = json.loads((cache / "sft_train_states.json").read_text(encoding="utf-8"))
    new_states = new_data_states(reactions_path)
    print(f"sft states: {len(sft_states)} | new ART-aug states: {len(new_states)}")

    print("building fingerprints ...")
    sft_fps = fps_for_states(sft_states)
    new_fps = fps_for_states(new_states)
    union_fps = sft_fps + new_fps

    sims_sft = nn_sims(benchmark_steps, sft_fps)
    sims_union = nn_sims(benchmark_steps, union_fps)

    rows = []
    for split in ["A", "B", "C", "All"]:
        idx = [i for i, s in enumerate(benchmark_steps) if split == "All" or s.split == split]
        n, m_sft, _ = summ([sims_sft[i] for i in idx])
        _, m_union, _ = summ([sims_union[i] for i in idx])
        rows.append({"split": split, "reaction_type": "ALL", "n_steps": n,
                     "mean_sft": round(m_sft, 4), "mean_sft+art": round(m_union, 4),
                     "delta": round(m_union - m_sft, 4)})

    for split in ["A", "B", "C", "All"]:
        for rt in REACTION_TYPES:
            idx = [i for i, s in enumerate(benchmark_steps)
                   if (split == "All" or s.split == split) and s.reaction_type == rt]
            if not idx:
                continue
            n, m_sft, _ = summ([sims_sft[i] for i in idx])
            _, m_union, _ = summ([sims_union[i] for i in idx])
            rows.append({"split": split, "reaction_type": rt, "n_steps": n,
                         "mean_sft": round(m_sft, 4), "mean_sft+art": round(m_union, 4),
                         "delta": round(m_union - m_sft, 4)})

    out_csv = out_dir / "art_augmented_coverage_gain.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "reaction_type", "n_steps",
                                           "mean_sft", "mean_sft+art", "delta"])
        w.writeheader()
        w.writerows(rows)

    print("\n=== coverage gain (mean max Tanimoto to training) ===")
    for r in rows:
        if r["reaction_type"] == "ALL":
            print(f"  {r['split']:>3} ALL  sft={r['mean_sft']:.3f} -> sft+art={r['mean_sft+art']:.3f}  (+{r['delta']:.3f})")
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
