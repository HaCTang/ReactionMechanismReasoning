#!/usr/bin/env python3
"""Rank unlabeled FlowER states by molecular similarity to FukuyamaBench.

For each unique unlabeled state we score the *most benchmark-like substantive
component* (max Tanimoto of any heavy component to any benchmark molecule),
ignoring tiny reagents/solvents. We then keep the highest-scoring states as
domain-targeted (benchmark-genre) candidates for training/RL.

DECONTAMINATION: states whose best component exceeds --ceiling are treated as
near-duplicates of the benchmark and EXCLUDED (reported separately) -- this is a
domain-targeting selection, not test-set leakage.

NOTE: this file holds states only (no reaction IDs); the output is a ranked list
of states/components. Mapping the kept states back to full unlabeled reaction
trajectories requires the HF `johnny-w/flower:unlabeled` rows.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprint_similarity_analysis import mol_from_smiles  # noqa: E402

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def heavy(mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def load_benchmark_mol_fps(repo_root: Path):
    fps = []
    seen = set()
    for mech in sorted((repo_root / "fukuyama_bench" / "mechanisms").glob("*/mechanism.json")):
        data = json.loads(mech.read_text(encoding="utf-8"))
        for step in data.get("mechanism", []):
            for role in ("reactants", "products"):
                for sp in step.get(role, []):
                    smi = sp.get("smiles", "") if isinstance(sp, dict) else str(sp)
                    for part in str(smi).split("."):
                        m = mol_from_smiles(part)
                        if m is None or heavy(m) < 4:
                            continue
                        c = Chem.MolToSmiles(m)
                        if c in seen:
                            continue
                        seen.add(c)
                        fps.append(GEN.GetFingerprint(m))
    return fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3000)
    ap.add_argument("--ceiling", type=float, default=0.7, help="exclude near-duplicates above this")
    ap.add_argument("--min-heavy", type=int, default=6)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = Path(__file__).resolve().parent / "fingerprint" / "cache"
    out_dir = Path(__file__).resolve().parent / "fingerprint"

    print("loading benchmark fingerprints ...")
    bench_fps = load_benchmark_mol_fps(repo_root)
    print(f"  benchmark molecules: {len(bench_fps)}")

    states = json.loads((cache_dir / "unlabeled_train_states.json").read_text(encoding="utf-8"))
    print(f"unlabeled unique states: {len(states)}")

    comp_cache: dict[str, float] = {}

    def comp_sim(smi: str) -> float:
        if smi in comp_cache:
            return comp_cache[smi]
        m = mol_from_smiles(smi)
        if m is None or heavy(m) < args.min_heavy:
            comp_cache[smi] = -1.0
            return -1.0
        sims = DataStructs.BulkTanimotoSimilarity(GEN.GetFingerprint(m), bench_fps)
        v = max(sims) if sims else 0.0
        comp_cache[smi] = v
        return v

    kept_heap: list = []  # min-heap of (score, idx)
    n_excluded = 0
    n_scored = 0
    for i, state in enumerate(states):
        best = -1.0
        best_comp = ""
        for part in state.split("."):
            s = comp_sim(part)
            if s > best:
                best, best_comp = s, part
        if best < 0:
            continue
        n_scored += 1
        if best > args.ceiling:
            n_excluded += 1
            continue
        item = (best, i, best_comp, state)
        if len(kept_heap) < args.top:
            heapq.heappush(kept_heap, item)
        elif best > kept_heap[0][0]:
            heapq.heapreplace(kept_heap, item)
        if (i + 1) % 200000 == 0:
            print(f"  scored {i+1}/{len(states)} | comp_cache={len(comp_cache)} | kept_min={kept_heap[0][0]:.3f}")

    ranked = sorted(kept_heap, key=lambda x: -x[0])
    out_csv = out_dir / "unlabeled_topk_by_benchmark_similarity.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "best_component_tanimoto", "best_component_smiles", "state"])
        for r, (score, _idx, comp, state) in enumerate(ranked, 1):
            w.writerow([r, round(score, 4), comp, state])

    print(f"\nscored states (with >= {args.min_heavy} heavy-atom component): {n_scored}")
    print(f"excluded as near-duplicates (> {args.ceiling}): {n_excluded}")
    print(f"kept top {len(ranked)} (best component sim {ranked[0][0]:.3f} .. {ranked[-1][0]:.3f})")
    print(f"-> {out_csv}")


if __name__ == "__main__":
    main()
