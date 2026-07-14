#!/usr/bin/env python3
"""Assemble the molecularly-diverse mechanism training set (mech-infer-train-v1).

Sources (independent of FukuyamaBench; decontaminated against it):
  * oMe silver -- organic-synthesis mechanism trajectories
  * ART (Grossman) augmented -- textbook mechanism analogs
  * HumanBenchmark -- Clayden/literature pathways
  * PMechDB single-step -- polar mechanisms (optional top-up)

Published set: https://huggingface.co/datasets/Haocheng1/mech-infer-train-v1

Rebuild from seeds under data_prep/sources/ (or --sources-dir).
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

from rdkit import RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprint_similarity_analysis import mol_from_smiles  # noqa: E402
from generate_art_augmented import (  # noqa: E402
    load_benchmark_fps,
    traj_fingerprint,
    variant_contaminated,
)

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = Path(__file__).resolve().parent / "sources"


def steps_valid(steps: list[dict]) -> bool:
    for s in steps:
        for side in (s["reactants"], s["products"]):
            parts = [p for p in side.split(".") if p]
            if not parts:
                return False
            for p in parts:
                if mol_from_smiles(p) is None:
                    return False
    return True


def load_ome_silver(sources: Path) -> list[dict]:
    data = json.loads((sources / "oMe_Combined_mapped.json").read_text(encoding="utf-8"))["data"]
    out = []
    for case in data:
        if str(case.get("source", "")).lower() != "silver":
            continue
        mech = case.get("mechanism", [])
        if not mech:
            continue
        steps, types = [], []
        ok = True
        for st in mech:
            react = [sp.get("smiles", "") for sp in st.get("reactants", []) if sp.get("smiles")]
            prod = [sp.get("smiles", "") for sp in st.get("products", []) if sp.get("smiles")]
            if not react or not prod:
                ok = False
                break
            steps.append({"reactants": ".".join(react), "products": ".".join(prod)})
            types.append(st.get("type", "unknown"))
        if not ok or not steps:
            continue
        conds = []
        for rxn in case.get("reactions", []):
            for c in rxn.get("conditions", []):
                txt = c.get("text", "")
                if txt:
                    conds.append({"role": c.get("role", "reagent"), "text": txt})
        out.append({
            "reaction_id": f"oMe_{case.get('id')}",
            "source": "oMe_silver",
            "level": case.get("level", ""),
            "reaction_types": types,
            "conditions": conds,
            "steps": steps,
        })
    return out


def load_humanbenchmark(sources: Path) -> list[dict]:
    p = sources / "350_HumanBenchmark_mapped.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))["data"]
    out = []
    for case in data:
        mech = case.get("mechanism", [])
        if not mech:
            continue
        steps, ok = [], True
        for st in mech:
            react = [sp.get("smiles", "") for sp in st.get("reactants", []) if sp.get("smiles")]
            prod = [sp.get("smiles", "") for sp in st.get("products", []) if sp.get("smiles")]
            if not react or not prod:
                ok = False
                break
            steps.append({"reactants": ".".join(react), "products": ".".join(prod)})
        if not ok or not steps:
            continue
        conds = []
        for rxn in case.get("reactions", []):
            for c in rxn.get("conditions", []):
                if c.get("text"):
                    conds.append({"role": c.get("role", "reagent"), "text": c["text"]})
        out.append({
            "reaction_id": f"HB_{case.get('id')}",
            "source": "human_benchmark",
            "reaction_types": [],
            "conditions": conds,
            "steps": steps,
        })
    return out


def load_pmechdb(sources: Path) -> list[dict]:
    import csv as _csv

    p = sources / "manually_curated_train.csv"
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as fh:
        for i, row in enumerate(_csv.DictReader(fh)):
            cell = row.get("SMIRKS and Arrow Codes", "")
            rxn = cell.split(" ")[0] if cell else ""
            if ">>" not in rxn:
                continue
            lhs, rhs = rxn.split(">>", 1)
            if not lhs.strip() or not rhs.strip():
                continue
            out.append({
                "reaction_id": f"PMechDB_mc_{i}",
                "source": "pmechdb_singlestep",
                "reaction_types": [],
                "conditions": [],
                "steps": [{"reactants": lhs.strip(), "products": rhs.strip()}],
            })
    return out


def load_art_aug(sources: Path) -> list[dict]:
    p = sources / "art_augmented_reactions.json"
    if not p.exists():
        return []
    rxns = json.loads(p.read_text(encoding="utf-8"))["reactions"]
    for r in rxns:
        r.setdefault("reaction_types", [])
    return rxns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources-dir", type=Path, default=DEFAULT_SOURCES)
    ap.add_argument("--decontam-thresh", type=float, default=0.7)
    ap.add_argument(
        "--decontam-min-heavy",
        type=int,
        default=8,
        help="only substantive molecules (>= this many heavy atoms) count for contamination",
    )
    ap.add_argument(
        "--target",
        type=int,
        default=10000,
        help="total target size; PMechDB single-step entries top up to this",
    )
    ap.add_argument("--no-pmechdb", action="store_true", help="multi-step core only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sources = args.sources_dir
    out_dir = Path(args.out) if args.out else REPO_ROOT / "data" / "mech_infer_train_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading benchmark fingerprints for decontamination ...")
    bench_fps, bench_keys = load_benchmark_fps(REPO_ROOT)
    print(f"  benchmark molecules: {len(bench_fps)} fps")

    core = []
    core += load_ome_silver(sources)
    print(f"oMe_silver loaded: {len(core)}")
    n = len(core)
    core += load_humanbenchmark(sources)
    print(f"HumanBenchmark loaded: {len(core) - n}")
    n = len(core)
    core += load_art_aug(sources)
    print(f"ART augmented loaded: {len(core) - n}")

    kept, meta = [], []
    seen, cache = set(), {}
    drop_invalid = drop_dup = drop_contam = 0

    def consider(r, stop_at=None) -> bool:
        nonlocal drop_invalid, drop_dup, drop_contam
        if stop_at is not None and len(kept) >= stop_at:
            return False
        steps = r["steps"]
        if not steps_valid(steps):
            drop_invalid += 1
            return True
        th = traj_fingerprint(steps)
        if th in seen:
            drop_dup += 1
            return True
        if variant_contaminated(
            steps,
            bench_fps,
            bench_keys,
            args.decontam_thresh,
            cache,
            min_heavy=args.decontam_min_heavy,
        ):
            drop_contam += 1
            return True
        seen.add(th)
        kept.append(r)
        meta.append({
            "reaction_id": r["reaction_id"],
            "source": r["source"],
            "n_steps": len(steps),
            "reaction_types": "|".join(r.get("reaction_types", [])),
        })
        return True

    for r in core:
        consider(r)
    n_core = len(kept)
    print(f"multi-step core kept: {n_core}")

    if not args.no_pmechdb and len(kept) < args.target:
        for r in load_pmechdb(sources):
            if not consider(r, stop_at=args.target):
                break
    print(f"after PMechDB top-up: {len(kept)} (single-step added: {len(kept) - n_core})")

    (out_dir / "reactions.json").write_text(
        json.dumps({"reactions": kept}, indent=2), encoding="utf-8"
    )
    with (out_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["reaction_id", "source", "n_steps", "reaction_types"])
        w.writeheader()
        w.writerows(meta)

    by_src = collections.Counter(r["source"] for r in kept)
    type_cov = collections.Counter()
    for r in kept:
        for t in r.get("reaction_types", []):
            type_cov[t] += 1
    print(
        f"\nKept {len(kept)} trajectories "
        f"(dropped: invalid={drop_invalid}, dup={drop_dup}, contaminated={drop_contam})"
    )
    print("by source:", dict(by_src))
    print("reaction-type step coverage:", dict(type_cov.most_common()))
    print(f"-> {out_dir / 'reactions.json'}")


if __name__ == "__main__":
    main()
