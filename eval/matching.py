#!/usr/bin/env python3
"""
Label-aware checkpoint matching for pathway mechanism evaluation.

Each ground-truth mechanism step records not just its product species but a
chemical label for each one (e.g. "intermediate", "product", "leaving",
"reagent", "catalyst", "solvent" - see fukuyama_bench/mechanisms/*/mechanism.json).
A prediction that only reproduces a byproduct (a leaving group, a spectator
reagent) must not count as having matched the step just because it is a
nonempty subset of the ground-truth products.

For a ground-truth step with product species G, define the required species
R (the chemically meaningful species the model must have predicted):

    1. All species labeled "intermediate", if any are present.
    2. Otherwise, all species labeled "product", if any are present.
    3. Otherwise, the single largest species by heavy-atom count (ties
       broken deterministically by canonical SMILES, then input order).

A prediction P (a set of predicted product SMILES) matches the step iff

    R subset-or-equal P subset-or-equal G

using RDKit-canonicalized molecular structures. Strict mode ignores labels
entirely and requires P == G exactly (full product-set equality).
"""

from typing import Dict, List, Optional, Sequence, Tuple
from rdkit import Chem


def canonicalize_smiles(smiles: Optional[str]) -> Optional[str]:
    """Canonicalize a single SMILES string using RDKit. Returns None if invalid."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def canonicalize_smiles_set(smiles_list: Sequence[str]) -> set:
    """Canonicalize a list of SMILES and return the set of valid canonical forms."""
    result = set()
    for smi in smiles_list:
        canon = canonicalize_smiles(smi)
        if canon:
            result.add(canon)
    return result


def heavy_atom_count(smiles: str) -> int:
    """Number of heavy (non-hydrogen) atoms in a SMILES string. 0 if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()


def select_required_species(gt_products: List[Dict]) -> List[str]:
    """
    Select the required species R from a step's labeled ground-truth
    products G.

    Args:
        gt_products: List of {"smiles": str, "label": Optional[str]}
                     describing every ground-truth product species for the
                     step.

    Returns:
        List of (uncanonicalized) SMILES strings that must all be present
        in a prediction for the step to match.
    """
    intermediates = [
        p["smiles"] for p in gt_products
        if p.get("label") == "intermediate" and p.get("smiles")
    ]
    if intermediates:
        return intermediates

    products = [
        p["smiles"] for p in gt_products
        if p.get("label") == "product" and p.get("smiles")
    ]
    if products:
        return products

    # Fallback: the single largest species by heavy-atom count. Ties are
    # broken deterministically by canonical SMILES (falling back to the raw
    # string if canonicalization fails) so the choice never depends on
    # dict/set iteration order.
    candidates = [p["smiles"] for p in gt_products if p.get("smiles")]
    if not candidates:
        return []

    def sort_key(smi: str):
        canon = canonicalize_smiles(smi) or smi
        return (-heavy_atom_count(smi), canon)

    return [sorted(candidates, key=sort_key)[0]]


def match_step(
    pred_products: Sequence[str],
    gt_products: List[Dict],
    strict: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Check whether predicted products satisfy a single ground-truth step.

    Args:
        pred_products: Predicted product SMILES for this step.
        gt_products: All ground-truth product species G for the step, as
                     {"smiles": str, "label": Optional[str]} dicts.
        strict: If True, require the predicted product set to equal G
                exactly (full product-set equality; labels are ignored).
                If False, require R subset-or-equal P subset-or-equal G,
                where R is selected by select_required_species(gt_products).

    Returns:
        (is_match, error_type). error_type is None on success.
    """
    gt_smiles = [p["smiles"] for p in gt_products if p.get("smiles")]
    pred_canon = canonicalize_smiles_set(pred_products)
    gt_canon = canonicalize_smiles_set(gt_smiles)

    if not pred_canon and not gt_canon:
        return False, "both_invalid"
    if not gt_canon:
        return False, "invalid_gt"
    if not pred_canon:
        return False, "invalid_pred"

    if pred_canon == gt_canon:
        return True, None

    if strict:
        return False, "strict_mismatch"

    if not pred_canon.issubset(gt_canon):
        if gt_canon.issubset(pred_canon):
            return False, "superset_match"
        if pred_canon & gt_canon:
            return False, "partial_match"
        return False, "mismatch"

    required = canonicalize_smiles_set(select_required_species(gt_products))
    if not required:
        return False, "no_required_species"

    if required.issubset(pred_canon):
        return True, "label_aware_match"

    return False, "missing_required_species"
