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

These tests exercise the fix directly against the real FukuyamaBench
mechanism data (fukuyama_bench/mechanisms/), not synthetic fixtures, so
they catch regressions in both the matching rule and its wiring into
eval_infer_pathway.py / run_infer_pathway.py.
"""

import json
import os

import pytest
from rdkit import RDLogger

import eval_infer_pathway as E
import matching as M
import run_infer_pathway as R

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

# Known data limitation: these cases' mechanism.json contains SMILES that
# RDKit's default sanitizer cannot parse at all (dative-bonded organometallic
# complexes, or syntactically-invalid ionic aromatics like "[Br-]c1ccccc1").
# When every species in a step's required set is unparsable, the step can
# never be matched regardless of the matching rule. See the PR description
# for the specific SMILES.
KNOWN_INVALID_SMILES_CASES = {"B042", "B073", "B127", "C069", "C081", "C093"}


def _all_case_ids():
    return sorted(
        d for d in os.listdir(MECHANISMS_DIR)
        if os.path.isdir(os.path.join(MECHANISMS_DIR, d))
    )


def _load_gt_pathway(case_id):
    """Mirror run_infer_pathway.load_pathway_data's ground_truth_pathway
    construction (step_id, products, product_labels) directly from
    mechanism.json."""
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
    return [
        {"products": [s] if isinstance(s, str) else list(s)}
        for s in steps
    ]


def _required(gt_pathway, step_idx):
    """Required species R for gt_pathway[step_idx] (0-indexed)."""
    labeled = E.get_step_products_with_labels(gt_pathway[step_idx], None)
    return M.select_required_species(labeled)


class TestA020LeavingGroupAttack:
    """A020 is the bug report's worked example: step 1 produces the
    intermediate enolate plus leaving-group water; the "2,3,4" checkpoint's
    alternatives each produce an intermediate plus a leaving group too."""

    def test_leaving_group_only_fails_under_default_settings(self):
        gt_pathway = _load_gt_pathway("A020")
        checkpoints = E.load_checkpoints(MECHANISMS_DIR, "A020")
        assert checkpoints == [[1], [2, 3, 4]]

        # The exact attack from the bug report: only the leaving-group
        # byproducts, never the actual intermediates.
        pred_pathway = _pred_pathway("O", "[OH-]")

        result = E.compare_pathways(pred_pathway, gt_pathway, lenient=True, checkpoints=checkpoints)

        assert result["exact_match"] is False
        assert result["checkpoints_correct"] == 0

    def test_full_attack_sequence_fails_on_all_319_cases(self):
        """The documented attack sequence (13 unrelated molecules x3 = 39
        steps) must not satisfy any benchmark case."""
        case_ids = _all_case_ids()
        assert len(case_ids) == 319

        pred_pathway = _pred_pathway(*(ATTACK_13 * 3))

        passing = []
        for case_id in case_ids:
            gt_pathway = _load_gt_pathway(case_id)
            checkpoints = E.load_checkpoints(MECHANISMS_DIR, case_id)
            result = E.compare_pathways(pred_pathway, gt_pathway, lenient=True, checkpoints=checkpoints)
            if result["exact_match"]:
                passing.append(case_id)

        assert passing == []


class TestRequiredVsOptionalSpecies:
    """R (required species) must be necessary; everything else in G is
    optional and may be omitted, but nothing outside G may be added."""

    def test_required_intermediate_alone_is_sufficient(self):
        gt_pathway = _load_gt_pathway("A020")
        step1_labeled = E.get_step_products_with_labels(gt_pathway[0], None)

        # Omit the optional leaving-group byproduct "O" entirely.
        is_match, error = M.match_step(["[CH2]=C([O-])c1ccccc1"], step1_labeled, strict=False)

        assert is_match is True
        assert error == "label_aware_match"

    def test_omitting_the_required_intermediate_fails(self):
        gt_pathway = _load_gt_pathway("A020")
        step1_labeled = E.get_step_products_with_labels(gt_pathway[0], None)

        # Only the optional byproduct, no intermediate - must fail.
        is_match, _ = M.match_step(["O"], step1_labeled, strict=False)

        assert is_match is False

    def test_adding_an_unrelated_species_fails(self):
        gt_pathway = _load_gt_pathway("A020")
        step1_labeled = E.get_step_products_with_labels(gt_pathway[0], None)

        # Required intermediate present, but with an extra species that is
        # not in G at all.
        is_match, error = M.match_step(
            ["[CH2]=C([O-])c1ccccc1", "N"], step1_labeled, strict=False
        )

        assert is_match is False
        assert error in ("partial_match", "mismatch", "superset_match")


class TestAlternativeCheckpointSteps:
    """A checkpoint listing several step IDs represents alternatives: each
    is matched independently against its own required species and complete
    product set, never combined with another alternative's."""

    def test_each_alternative_matches_on_its_own(self):
        gt_pathway = _load_gt_pathway("A020")
        checkpoints = E.load_checkpoints(MECHANISMS_DIR, "A020")
        step1_required = _required(gt_pathway, 0)

        for alt_idx in (1, 2, 3):  # steps 2, 3, 4 (0-indexed 1, 2, 3)
            alt_required = _required(gt_pathway, alt_idx)
            pred_pathway = _pred_pathway(step1_required, alt_required)

            result = E.compare_pathways(pred_pathway, gt_pathway, lenient=True, checkpoints=checkpoints)

            assert result["exact_match"] is True, f"alternative step {alt_idx + 1} failed to match"
            assert result["checkpoints_correct"] == 2

    def test_alternatives_are_not_combined(self):
        """Mixing required species from two different alternatives into one
        predicted step must not match any single alternative's product set."""
        gt_pathway = _load_gt_pathway("A020")
        checkpoints = E.load_checkpoints(MECHANISMS_DIR, "A020")
        step1_required = _required(gt_pathway, 0)
        step2_required = _required(gt_pathway, 1)
        step4_required = _required(gt_pathway, 3)

        mixed = step2_required + step4_required
        pred_pathway = _pred_pathway(step1_required, mixed)

        result = E.compare_pathways(pred_pathway, gt_pathway, lenient=True, checkpoints=checkpoints)

        assert result["checkpoints_correct"] < 2


class TestFallbackRules:
    """When no species is labeled "intermediate", fall back to "product";
    when neither is present, fall back to the largest species by
    heavy-atom count."""

    def test_product_label_fallback(self):
        # A020 step 5 (the final step) has no "intermediate"-labeled
        # product, only "product" and "leaving" - rule 2 applies.
        gt_pathway = _load_gt_pathway("A020")
        step5_labeled = E.get_step_products_with_labels(gt_pathway[4], None)
        assert not any(p["label"] == "intermediate" for p in step5_labeled)
        assert any(p["label"] == "product" for p in step5_labeled)

        required = M.select_required_species(step5_labeled)

        assert required == ["O=C(/C=C/c1ccccc1)c1ccccc1"]

    def test_largest_heavy_atom_fallback(self):
        # A068 step 6 has neither "intermediate" nor "product" labels
        # (just "reagent" and "leaving") - rule 3 applies: the larger
        # species (tert-butanol, 5 heavy atoms) must be selected over the
        # smaller one (SeO2, 3 heavy atoms).
        gt_pathway = _load_gt_pathway("A068")
        step6 = next(s for s in gt_pathway if s["step_id"] == "6")
        step6_labeled = E.get_step_products_with_labels(step6, None)
        assert not any(p["label"] in ("intermediate", "product") for p in step6_labeled)

        required = M.select_required_species(step6_labeled)

        assert required == ["CC(C)(C)O"]
        assert M.heavy_atom_count("CC(C)(C)O") > M.heavy_atom_count("O=[Se]=O")

    def test_largest_heavy_atom_fallback_is_deterministic(self):
        gt_products = [
            {"smiles": "O=[Se]=O", "label": "reagent"},
            {"smiles": "CC(C)(C)O", "label": "leaving"},
        ]
        first = M.select_required_species(gt_products)
        second = M.select_required_species(list(reversed(gt_products)))

        assert first == second == ["CC(C)(C)O"]


class TestStrictModePreserved:
    """--strict must remain full product-set equality, unaffected by labels."""

    def test_strict_rejects_required_species_alone(self):
        gt_pathway = _load_gt_pathway("A020")
        step1_labeled = E.get_step_products_with_labels(gt_pathway[0], None)

        is_match, error = M.match_step(["[CH2]=C([O-])c1ccccc1"], step1_labeled, strict=True)

        assert is_match is False
        assert error == "strict_mismatch"

    def test_strict_accepts_full_product_set(self):
        gt_pathway = _load_gt_pathway("A020")
        step1_labeled = E.get_step_products_with_labels(gt_pathway[0], None)
        full_products = gt_pathway[0]["products"]

        is_match, error = M.match_step(full_products, step1_labeled, strict=True)

        assert is_match is True
        assert error is None


class TestOrderedCheckpointMatchingRetained:
    """Checkpoints must still be matched in order against the predicted
    steps, skipping over non-matching predicted steps as before."""

    def test_junk_steps_between_matches_are_skipped(self):
        gt_pathway = _load_gt_pathway("A020")
        checkpoints = E.load_checkpoints(MECHANISMS_DIR, "A020")
        step1_required = _required(gt_pathway, 0)
        step2_required = _required(gt_pathway, 1)

        pred_pathway = _pred_pathway("N", step1_required, "N", step2_required)

        result = E.compare_pathways(pred_pathway, gt_pathway, lenient=True, checkpoints=checkpoints)

        assert result["exact_match"] is True
        assert result["checkpoints_correct"] == 2


class TestLabelsPreservedThroughNormalEntryPoint:
    """run_infer_pathway.load_pathway_data must carry label metadata through
    to ground_truth_pathway so the evaluator does not need mechanism.json
    for newly-generated inference results."""

    def test_load_pathway_data_includes_product_labels(self):
        samples = R.load_pathway_data(MECHANISMS_DIR, case_ids=["A020"])
        assert len(samples) == 1
        gt_pathway = samples[0]["ground_truth_pathway"]

        assert gt_pathway[0]["product_labels"] == ["intermediate", "leaving"]
        assert len(gt_pathway[0]["product_labels"]) == len(gt_pathway[0]["products"])

    def test_evaluator_matches_label_aware_without_mechanisms_dir(self):
        """Given a result file built the normal way (with product_labels
        already attached), label-aware matching must work even without
        re-reading mechanism.json."""
        samples = R.load_pathway_data(MECHANISMS_DIR, case_ids=["A020"])
        gt_pathway = samples[0]["ground_truth_pathway"]
        checkpoints = E.load_checkpoints(MECHANISMS_DIR, "A020")

        pred_pathway = _pred_pathway("O", "[OH-]")  # the leaving-group attack
        result = E.compare_pathways(
            pred_pathway, gt_pathway, lenient=True, checkpoints=checkpoints, mechanism_labels=None
        )

        assert result["exact_match"] is False


class TestOldResultFilesWithoutLabels:
    """Older inference-result files have no "product_labels" field. The
    evaluator must fall back to mechanism.json via mechanisms_dir and still
    apply label-aware matching correctly."""

    def test_falls_back_to_mechanism_json_for_labels(self):
        gt_pathway = _load_gt_pathway("A020")
        for step in gt_pathway:
            del step["product_labels"]  # simulate an old result file

        checkpoints = E.load_checkpoints(MECHANISMS_DIR, "A020")
        mechanism_labels = E.load_mechanism_labels(MECHANISMS_DIR, "A020")

        pred_pathway = _pred_pathway("O", "[OH-]")
        result = E.compare_pathways(
            pred_pathway, gt_pathway, lenient=True, checkpoints=checkpoints, mechanism_labels=mechanism_labels
        )

        assert result["exact_match"] is False


class TestReferencePredictionCoverage:
    """An "intermediate-only" reference prediction - the required species R
    for every ground-truth step - should pass virtually every case under
    the new matching rule. The known exceptions are pinned explicitly so a
    regression here is caught, rather than silently accepted."""

    def test_reference_predictions_pass_except_known_invalid_smiles_cases(self):
        failing = []
        for case_id in _all_case_ids():
            gt_pathway = _load_gt_pathway(case_id)
            checkpoints = E.load_checkpoints(MECHANISMS_DIR, case_id)
            pred_pathway = [
                {"products": _required_from_step(step)}
                for step in gt_pathway
            ]
            result = E.compare_pathways(pred_pathway, gt_pathway, lenient=True, checkpoints=checkpoints)
            if not result["exact_match"]:
                failing.append(case_id)

        assert set(failing) == KNOWN_INVALID_SMILES_CASES

    def test_known_failures_are_due_to_unparsable_ground_truth_smiles(self):
        for case_id in KNOWN_INVALID_SMILES_CASES:
            gt_pathway = _load_gt_pathway(case_id)
            unparsable = [
                smi for step in gt_pathway for smi in step["products"]
                if M.canonicalize_smiles(smi) is None
            ]
            assert unparsable, f"{case_id} was expected to contain an unparsable ground-truth SMILES"


def _required_from_step(step):
    labeled = E.get_step_products_with_labels(step, None)
    return M.select_required_species(labeled)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
