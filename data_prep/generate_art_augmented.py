#!/usr/bin/env python3
"""Generate molecularly-augmented mechanism trajectories from the independent
ART (Grossman, *The Art of Writing Reasonable Organic Reaction Mechanisms*)
corpus, to broaden the chemical-space coverage of the SFT training set toward
the FukuyamaBench organic-synthesis genre.

Design (deliberately NON-contaminating):
  * Seeds come ONLY from ART_mechanisms_gemini/ (an independent textbook,
    disjoint from the Fukuyama workbook). The benchmark is NEVER used as a seed.
  * Each ART mechanism is augmented by decorating a *conserved spectator site*
    (a CH on the skeleton common to every species in the mechanism, found via
    MCS) with a small functional group, applied consistently to every molecule
    in every elementary step.
  * A variant is kept only if, for every step, the per-step molecular-formula
    delta and net-charge delta are byte-identical to the original step. This
    guarantees the elementary transformation is chemically unchanged (the
    substituent is a true spectator) -> the mechanism stays balanced/reasonable.
  * Fingerprints are used for DECONTAMINATION ONLY (exclusion): any variant
    whose molecules are an exact match (InChIKey) or near-duplicate
    (Tanimoto > --decontam-thresh) of any benchmark molecule is dropped.

Output: annotation-ready reaction trajectories in the schema consumed by
annotate_reaction.py  ->  convert_to_train.py  ->  SFT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFMCS, rdFingerprintGenerator

# Reuse the exact fingerprint / canonicalization methodology already used for
# the rebuttal's similarity analysis so decontamination is consistent.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprint_similarity_analysis import (  # noqa: E402
    canonicalize_state,
    mol_from_smiles,
)

RDLogger.DisableLog("rdApp.*")

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Spectator decorations. Each value is a SMILES fragment whose atom 0 is bonded
# to the chosen skeletal site (replacing one H there). Chosen to be small,
# common organic substituents that read as reasonable substrate analogs.
# Restricted to common, genre-appropriate substituents so decorated analogs stay
# close to the FukuyamaBench organic-synthesis chemical space (mono / light-double
# substitution only -- heavy multi-substitution pushes molecules out of genre).
SUBSTITUENTS = {
    "F": "F",
    "Cl": "Cl",
    "Br": "Br",
    "Me": "C",
    "Et": "CC",
    "iPr": "C(C)C",
    "OMe": "OC",
    "OEt": "OCC",
}
# Palette for the 2nd site of (light) disubstituted variants.
SECOND_SITE_SUBS = ["F", "Cl", "Me", "OMe"]


def heavy_atoms(mol: Chem.Mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def parse_step(rxn_smiles: str) -> tuple[list[str], list[str]] | None:
    """Split 'A.B>>C.D' into ([A,B],[C,D]) raw component SMILES."""
    if ">>" not in rxn_smiles:
        return None
    lhs, rhs = rxn_smiles.split(">>", 1)
    react = [s for s in lhs.split(".") if s.strip()]
    prod = [s for s in rhs.split(".") if s.strip()]
    if not react or not prod:
        return None
    return react, prod


def side_signature(components: list[str]) -> tuple | None:
    """Element+H multiset and total formal charge for one reaction side."""
    elems: Counter = Counter()
    charge = 0
    for smi in components:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        for atom in m.GetAtoms():
            elems[atom.GetSymbol()] += 1
            elems["H"] += atom.GetTotalNumHs()
            charge += atom.GetFormalCharge()
    return (tuple(sorted(elems.items())), charge)


def step_delta(react: list[str], prod: list[str]) -> tuple | None:
    rs = side_signature(react)
    ps = side_signature(prod)
    if rs is None or ps is None:
        return None
    r_elems, r_chg = rs
    p_elems, p_chg = ps
    d: Counter = Counter(dict(p_elems))
    d.subtract(Counter(dict(r_elems)))
    return (tuple(sorted((k, v) for k, v in d.items() if v != 0)), p_chg - r_chg)


def load_art_case(case_dir: Path) -> dict | None:
    rxn_path = case_dir / "rxn.json"
    if not rxn_path.exists():
        return None
    data = json.loads(rxn_path.read_text(encoding="utf-8"))
    steps = []
    for entry in data.get("mechanism_rxn", []):
        parsed = parse_step(entry.get("rxn_smiles", ""))
        if parsed is None:
            return None
        react, prod = parsed
        steps.append({"reactants": ".".join(react), "products": ".".join(prod)})
    if not steps:
        return None
    conditions = []
    cond_path = case_dir / "conditions.txt"
    if cond_path.exists():
        for line in cond_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                conditions.append({"role": "reagent", "text": line})
    return {"case_id": case_dir.name, "steps": steps, "conditions": conditions}


def all_components(case: dict) -> list[str]:
    comps: list[str] = []
    for step in case["steps"]:
        comps.extend(step["reactants"].split("."))
        comps.extend(step["products"].split("."))
    return comps


def find_mcs_sites(case: dict, max_sites: int) -> tuple[Chem.Mol | None, list[int]]:
    """Compute the MCS over skeleton-bearing molecules and return aromatic-CH
    site indices (in the MCS query) usable for consistent decoration."""
    seen: dict[str, Chem.Mol] = {}
    for smi in all_components(case):
        m = mol_from_smiles(smi)
        if m is None or heavy_atoms(m) < 5:
            continue
        seen.setdefault(Chem.MolToSmiles(m), m)
    mols = list(seen.values())
    if len(mols) < 2:
        return None, []
    res = rdFMCS.FindMCS(
        mols,
        timeout=10,
        matchValences=True,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
    )
    if res.canceled or res.numAtoms < 5:
        return None, []
    query = Chem.MolFromSmarts(res.smartsString)
    if query is None:
        return None, []
    # An MCS-query atom is a usable site if it is an aromatic carbon that carries
    # an H in EVERY skeleton molecule (a genuine, decoratable spectator position).
    sites: list[int] = []
    for qatom in query.GetAtoms():
        qi = qatom.GetIdx()
        ok = True
        found_any = False
        for m in mols:
            match = m.GetSubstructMatch(query)
            if not match:
                ok = False
                break
            a = m.GetAtomWithIdx(match[qi])
            if a.GetAtomicNum() != 6 or a.GetTotalNumHs() < 1:
                ok = False
                break
            found_any = True
        if ok and found_any:
            sites.append(qi)
        if len(sites) >= max_sites:
            break
    return query, sites


def decorate_mol(mol: Chem.Mol, atom_idx: int, frag_smiles: str) -> Chem.Mol | None:
    frag = Chem.MolFromSmiles(frag_smiles)
    if frag is None:
        return None
    rw = Chem.RWMol(mol)
    amap = {}
    for a in frag.GetAtoms():
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(a.GetFormalCharge())
        na.SetNoImplicit(a.GetNoImplicit())
        amap[a.GetIdx()] = rw.AddAtom(na)
    for b in frag.GetBonds():
        rw.AddBond(amap[b.GetBeginAtomIdx()], amap[b.GetEndAtomIdx()], b.GetBondType())
    rw.AddBond(atom_idx, amap[0], Chem.BondType.SINGLE)
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m


def decorate_component(smi: str, query: Chem.Mol, site_assignment: dict[int, str]) -> str | None:
    """Decorate one component: for each (mcs_site -> substituent) that the
    component matches, add the substituent at the mapped atom. Components that
    do not contain the MCS are returned unchanged."""
    m = mol_from_smiles(smi)
    if m is None:
        return smi
    match = m.GetSubstructMatch(query)
    if not match:
        return smi  # spectator-free small reagent: leave as-is
    changed = False
    for site_qi, frag in site_assignment.items():
        atom_idx = match[site_qi]
        a = m.GetAtomWithIdx(atom_idx)
        if a.GetAtomicNum() != 6 or a.GetTotalNumHs() < 1:
            return None  # inconsistent placement -> reject whole variant
        new = decorate_mol(m, atom_idx, SUBSTITUENTS[frag])
        if new is None:
            return None
        m = new
        match = m.GetSubstructMatch(query)  # indices stable (appended atoms), refresh defensively
        if not match:
            return None
        changed = True
    if not changed:
        return smi
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def build_variant(case: dict, query: Chem.Mol, site_assignment: dict[int, str]) -> list[dict] | None:
    """Apply a site->substituent assignment to every step; validate balance."""
    new_steps = []
    for step in case["steps"]:
        react = step["reactants"].split(".")
        prod = step["products"].split(".")
        new_react, new_prod = [], []
        for smi in react:
            d = decorate_component(smi, query, site_assignment)
            if d is None:
                return None
            new_react.append(d)
        for smi in prod:
            d = decorate_component(smi, query, site_assignment)
            if d is None:
                return None
            new_prod.append(d)
        # balance must be identical to the original step
        if step_delta(new_react, new_prod) != step_delta(react, prod):
            return None
        new_steps.append({"reactants": ".".join(new_react), "products": ".".join(new_prod)})
    return new_steps


def traj_fingerprint(steps: list[dict]) -> str:
    blob = "|".join(f"{s['reactants']}>>{s['products']}" for s in steps)
    return hashlib.sha1(blob.encode()).hexdigest()


def load_benchmark_fps(repo_root: Path) -> tuple[list, set]:
    """Per-molecule Morgan fps and InChIKeys for every FukuyamaBench molecule."""
    fps, keys = [], set()
    seen_smiles = set()
    mech_dir = repo_root / "fukuyama_bench" / "mechanisms"
    for mech_path in sorted(mech_dir.glob("*/mechanism.json")):
        data = json.loads(mech_path.read_text(encoding="utf-8"))
        for step in data.get("mechanism", []):
            for role in ("reactants", "products"):
                for sp in step.get(role, []):
                    smi = sp.get("smiles", "") if isinstance(sp, dict) else str(sp)
                    for part in str(smi).split("."):
                        m = mol_from_smiles(part)
                        if m is None or heavy_atoms(m) < 3:
                            continue
                        cano = Chem.MolToSmiles(m)
                        if cano in seen_smiles:
                            continue
                        seen_smiles.add(cano)
                        fps.append(MORGAN.GetFingerprint(m))
                        try:
                            keys.add(Chem.MolToInchiKey(m))
                        except Exception:
                            pass
    return fps, keys


def variant_contaminated(steps: list[dict], bench_fps, bench_keys, thresh: float, cache: dict,
                         min_heavy: int = 3) -> bool:
    """Flag a trajectory as benchmark-contaminated if any *substantive* molecule
    (>= min_heavy heavy atoms) exactly matches (InChIKey) or is a near-duplicate
    (Tanimoto > thresh) of a benchmark molecule. Common small reagents/solvents
    are below min_heavy and ignored -- shared reagents are not contamination."""
    mols = set()
    for s in steps:
        mols.update(s["reactants"].split("."))
        mols.update(s["products"].split("."))
    for smi in mols:
        if smi in cache:
            if cache[smi]:
                return True
            continue
        m = mol_from_smiles(smi)
        bad = False
        if m is not None and heavy_atoms(m) >= min_heavy:
            try:
                if Chem.MolToInchiKey(m) in bench_keys:
                    bad = True
            except Exception:
                pass
            if not bad and bench_fps:
                sims = DataStructs.BulkTanimotoSimilarity(MORGAN.GetFingerprint(m), bench_fps)
                if sims and max(sims) > thresh:
                    bad = True
        cache[smi] = bad
        if bad:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=10000)
    ap.add_argument("--max-per-case", type=int, default=400)
    ap.add_argument("--max-sites", type=int, default=6)
    ap.add_argument("--decontam-thresh", type=float, default=0.7)
    ap.add_argument("--cases-limit", type=int, default=None, help="dry-run: only first N cases")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    art_dir = repo_root / "data_prep" / "sources" / "ART_mechanisms_gemini"
    out_dir = Path(args.out) if args.out else repo_root / "data" / "art_augmented_mechanisms"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark fingerprints for decontamination ...")
    bench_fps, bench_keys = load_benchmark_fps(repo_root)
    print(f"  benchmark molecules: {len(bench_fps)} fps, {len(bench_keys)} InChIKeys")

    case_dirs = sorted([d for d in art_dir.iterdir() if d.is_dir()])
    if args.cases_limit:
        case_dirs = case_dirs[: args.cases_limit]

    reactions = []
    meta_rows = []
    decontam_cache: dict[str, bool] = {}
    seen_traj: set[str] = set()
    rid = 0

    for case_dir in case_dirs:
        case = load_art_case(case_dir)
        if case is None:
            print(f"  [skip] {case_dir.name}: unparseable")
            continue
        # Emit the original (undecorated) ART mechanism first -- the most
        # genre-matched base -- after dedup + decontamination.
        orig_steps = case["steps"]
        th0 = traj_fingerprint(orig_steps)
        if th0 not in seen_traj and not variant_contaminated(
            orig_steps, bench_fps, bench_keys, args.decontam_thresh, decontam_cache
        ):
            seen_traj.add(th0)
            rid += 1
            reactions.append({
                "reaction_id": f"ART_{case['case_id']}_{rid}",
                "source": "ART_grossman_original",
                "source_case": case["case_id"],
                "augmentation": "original",
                "conditions": case["conditions"],
                "steps": orig_steps,
            })
            meta_rows.append({
                "reaction_id": f"ART_{case['case_id']}_{rid}",
                "source_case": case["case_id"],
                "n_steps": len(orig_steps),
                "augmentation": "original",
            })

        query, sites = find_mcs_sites(case, args.max_sites)
        kept = 0
        if query is not None and sites:
            # candidate assignments: single-site, then two-site combos
            assignments = []
            for site in sites:
                for sub in SUBSTITUENTS:
                    assignments.append({site: sub})
            for s1, s2 in itertools.combinations(sites, 2):
                for sub1 in SECOND_SITE_SUBS:
                    for sub2 in SECOND_SITE_SUBS:
                        assignments.append({s1: sub1, s2: sub2})
            for assignment in assignments:
                if kept >= args.max_per_case:
                    break
                steps = build_variant(case, query, assignment)
                if steps is None:
                    continue
                th = traj_fingerprint(steps)
                if th in seen_traj:
                    continue
                if variant_contaminated(steps, bench_fps, bench_keys, args.decontam_thresh, decontam_cache):
                    continue
                seen_traj.add(th)
                rid += 1
                label = "+".join(f"{k}:{v}" for k, v in sorted(assignment.items()))
                reactions.append({
                    "reaction_id": f"ART_{case['case_id']}_{rid}",
                    "source": "ART_grossman_augmented",
                    "source_case": case["case_id"],
                    "augmentation": label,
                    "conditions": case["conditions"],
                    "steps": steps,
                })
                meta_rows.append({
                    "reaction_id": f"ART_{case['case_id']}_{rid}",
                    "source_case": case["case_id"],
                    "n_steps": len(steps),
                    "augmentation": label,
                })
                kept += 1
        print(f"  {case_dir.name}: sites={len(sites)} kept={kept}")
        if len(reactions) >= args.target:
            print(f"Reached target {args.target}.")
            break

    out_json = out_dir / "reactions.json"
    out_json.write_text(json.dumps({"reactions": reactions}, indent=2), encoding="utf-8")
    with (out_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["reaction_id", "source_case", "n_steps", "augmentation"])
        w.writeheader()
        w.writerows(meta_rows)

    print(f"\nGenerated {len(reactions)} augmented mechanism trajectories")
    print(f"  -> {out_json}")
    print(f"  -> {out_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
