#!/usr/bin/env python3
"""
Regression tests for label-aware checkpoint matching (see matching.py and
the "Label-aware matching" section of eval_infer_pathway.py).

Background: the previous default evaluator accepted any nonempty subset of
a ground-truth step's product species as a match. That let a prediction
containing only a byproduct (e.g. a leaving group) satisfy a checkpoint -
for example, A020 incorrectly passed with a two-step prediction containing
only the leaving-group "O" followed by "[OH-]", neither of which is the
chemically meaningful intermediate for either step.

These tests exercise the fix against the real FukuyamaBench mechanism data
(fukuyama_bench/mechanisms/), not synthetic fixtures, so they catch
regressions in both the matching rule and its wiring into
eval_infer_pathway.py / run_infer_pathway.py.
"""

import os

import eval_infer_pathway as E
import matching as M
import run_infer_pathway as R
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

MECHANISMS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "fukuyama_bench", "mechanisms"
)

# The documented attack sequence: 13 unrelated small molecules, repeated
# three times (39 predicted steps), one molecule per predicted step.
ATTACK_13 = [
    "O", "[OH-]", "[H+]", "CO", "C[O-]", "[C-]#N", "O=S=O",
    "[Cl-]", "[Br-]", "[I-]", "[F-]", "O=C=O", "N",
]


def _all_case_ids():
    return sorted(
        d for d in os.listdir(MECHANISMS_DIR)
        if os.path.isdir(os.path.join(MECHANISMS_DIR, d))
    )


def _load_gt_pathway(case_id):
    """Mirror run_infer_pathway.load_pathway_data's ground_truth_pathway
    construction (step_id, products, product_labels) directly from
    mechanism.json."""
    import json
    mechanism_path = os.path.join(MECHANISMS_DIR, case_id, "mechanism.json")
    with open(mechanism_path, encoding="utf-8") as f:
        mechanism_data = json.load(f)
    gt_pathway = []
    for step in mechanism_data["mechanism"]:
        products = step.get("products", [])
        gt_pathway.append({
            "step_id": step.get("mechanism_id"),
            "products": [p.get("smiles") for p in products],
            "product_labels": [p.get("label") for p in products],
        })
    return gt_pathway


def _pred_pathway(*steps):
    """Build a minimal predicted pathway, one molecule (or list of
    molecules) per step."""
    return [{"products": [s] if isinstance(s, str) else list(s)} for s in steps]


def _required(gt_pathway, step_idx):
    """Required species R for gt_pathway[step_idx] (0-indexed)."""
    labeled = E.get_step_products_with_labels(gt_pathway[step_idx], None)
    return M.select_required_species(labeled)


def test_a020_leaving_group_attack_fails():
    """The bug report's worked example: a two-step prediction of only the
    leaving-group byproducts ("O" then "[OH-]") must not satisfy A020."""
    gt_pathway = _load_gt_pathway("A020")
    checkpoints = E.load_checkpoints(MECHANISMS_DIR, "A020")
    assert checkpoints == [[1], [2, 3, 4]]

    result = E.compare_pathways(
        _pred_pathway("O", "[OH-]"), gt_pathway, lenient=True, checkpoints=checkpoints
    )

    assert result["exact_match"] is False
    assert result["checkpoints_correct"] == 0


def test_attack_sequence_fails_on_all_319_cases():
    """The same fixed 39-step prediction (13 unrelated molecules x3) must
    not satisfy any of the 319 benchmark cases."""
    case_ids = _all_case_ids()
    assert len(case_ids) == 319
    pred_pathway = _pred_pathway(*(ATTACK_13 * 3))

    for case_id in case_ids:
        gt_pathway = _load_gt_pathway(case_id)
        checkpoints = E.load_checkpoints(MECHANISMS_DIR, case_id)
        result = E.compare_pathways(pred_pathway, gt_pathway, lenient=True, checkpoints=checkpoints)
        assert not result["exact_match"], f"attack sequence incorrectly passed {case_id}"


def test_required_species_govern_the_match():
    """Required species may not be omitted; optional byproducts may be;
    species outside the ground-truth set may not be added."""
    gt_pathway = _load_gt_pathway("A020")
    step1 = E.get_step_products_with_labels(gt_pathway[0], None)

    # Required intermediate alone is sufficient - the optional leaving-group
    # byproduct "O" may be omitted.
    assert M.match_step(["[CH2]=C([O-])c1ccccc1"], step1, strict=False)[0] is True
    # Omitting the required intermediate fails even with the byproduct present.
    assert M.match_step(["O"], step1, strict=False)[0] is False
    # Adding a species outside the ground-truth product set fails.
    assert M.match_step(["[CH2]=C([O-])c1ccccc1", "N"], step1, strict=False)[0] is False


def test_alternative_checkpoint_steps():
    """A checkpoint listing several step IDs represents alternatives: each
    is matched independently against its own required species and complete
    product set, never combined with another alternative's."""
    gt_pathway = _load_gt_pathway("A020")
    checkpoints = E.load_checkpoints(MECHANISMS_DIR, "A020")
    step1_required = _required(gt_pathway, 0)

    for alt_idx in (1, 2, 3):  # steps 2, 3, 4 (0-indexed)
        pred_pathway = _pred_pathway(step1_required, _required(gt_pathway, alt_idx))
        result = E.compare_pathways(pred_pathway, gt_pathway, lenient=True, checkpoints=checkpoints)
        assert result["exact_match"] is True, f"alternative step {alt_idx + 1} failed to match"

    # Mixing required species from two different alternatives into one
    # predicted step must not match any single alternative.
    mixed = _required(gt_pathway, 1) + _required(gt_pathway, 3)
    result = E.compare_pathways(
        _pred_pathway(step1_required, mixed), gt_pathway, lenient=True, checkpoints=checkpoints
    )
    assert result["checkpoints_correct"] < 2


def test_fallback_rules_for_required_species():
    """When no species is labeled "intermediate", fall back to "product";
    when neither is present, fall back to the largest species by
    heavy-atom count."""
    # Rule 2: A020's final step has only "product"/"leaving" labels.
    gt_pathway = _load_gt_pathway("A020")
    step5 = E.get_step_products_with_labels(gt_pathway[4], None)
    assert M.select_required_species(step5) == ["O=C(/C=C/c1ccccc1)c1ccccc1"]

    # Rule 3: A068 step 6 has only "reagent"/"leaving" labels - the larger
    # species (tert-butanol, 5 heavy atoms) must win over SeO2 (3 atoms).
    a068_pathway = _load_gt_pathway("A068")
    step6 = next(s for s in a068_pathway if s["step_id"] == "6")
    step6_labeled = E.get_step_products_with_labels(step6, None)
    assert M.select_required_species(step6_labeled) == ["CC(C)(C)O"]


def test_strict_mode_requires_full_product_set_equality():
    """--strict must remain full product-set equality, unaffected by labels."""
    gt_pathway = _load_gt_pathway("A020")
    step1 = E.get_step_products_with_labels(gt_pathway[0], None)

    assert M.match_step(["[CH2]=C([O-])c1ccccc1"], step1, strict=True)[0] is False
    assert M.match_step(gt_pathway[0]["products"], step1, strict=True)[0] is True


def test_labels_preserved_through_normal_entry_point():
    """run_infer_pathway.load_pathway_data must carry label metadata
    through to ground_truth_pathway."""
    samples = R.load_pathway_data(MECHANISMS_DIR, case_ids=["A020"])
    gt_pathway = samples[0]["ground_truth_pathway"]

    assert gt_pathway[0]["product_labels"] == ["intermediate", "leaving"]


def test_old_result_files_without_labels_still_work():
    """Older inference-result files have no "product_labels" field; the
    evaluator must fall back to mechanism.json via mechanisms_dir."""
    gt_pathway = _load_gt_pathway("A020")
    for step in gt_pathway:
        del step["product_labels"]  # simulate a pre-fix result file

    checkpoints = E.load_checkpoints(MECHANISMS_DIR, "A020")
    mechanism_labels = E.load_mechanism_labels(MECHANISMS_DIR, "A020")

    result = E.compare_pathways(
        _pred_pathway("O", "[OH-]"), gt_pathway, lenient=True,
        checkpoints=checkpoints, mechanism_labels=mechanism_labels,
    )

    assert result["exact_match"] is False
