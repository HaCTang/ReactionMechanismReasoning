#!/usr/bin/env python3
"""
Custom reward function for verl GSPO training on reaction mechanism prediction.

Reward design v5 (stronger format penalties):
  - Parse model output as JSON array of steps from ## Result section
  - Score each GT step independently:
    - Correct match (canonicalized SMILES equal): 1.0
    - Wrong but valid SMILES: 0.1
    - Invalid SMILES or missing step: 0.0
  - Final reward = mean(step_scores)

Extraction error penalties (much harsher than v3/v4):
  - No "## Result" section at all (reasoning truncation/loop): -1.0
  - "## Result" present but JSON unparseable (truncated JSON): -0.5

Changes from v4:
  - no_result: -0.3 → -1.0 (force model to always produce ## Result)
  - json_parse_fail: -0.2 → -0.5 (force valid JSON)
  - valid_wrong: 0.3 → 0.1 (back to v3, avoid lazy valid-SMILES strategy)

Examples (GT has 3 steps):
  3/3 correct, all valid → 1.0
  1/3 correct, 2 wrong but valid → (1.0 + 0.1 + 0.1) / 3 = 0.4
  0/3 correct, all valid → 0.1
  Predict only 1 step (correct) → (1.0 + 0 + 0) / 3 = 0.33
  No ## Result section → -1.0
  ## Result but bad JSON → -0.5
"""

import json
import re
from typing import Optional

from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid using RDKit."""
    if not smiles or not isinstance(smiles, str):
        return False
    try:
        for mol_smiles in smiles.split('.'):
            mol_smiles = mol_smiles.strip()
            if not mol_smiles:
                continue
            mol = Chem.MolFromSmiles(mol_smiles)
            if mol is None:
                return False
        return True
    except Exception:
        return False


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string, removing atom mapping and sorting molecules."""
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mols = []
        for mol_smiles in smiles.split('.'):
            mol_smiles = mol_smiles.strip()
            if not mol_smiles:
                continue
            mol = Chem.MolFromSmiles(mol_smiles)
            if mol is None:
                return None
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            mols.append(Chem.MolToSmiles(mol, canonical=True))
        return '.'.join(sorted(mols))
    except Exception:
        return None


def compare_smiles(pred_smiles: str, gt_smiles: str) -> bool:
    """Compare two SMILES strings after canonicalization."""
    pred_canon = canonicalize_smiles(pred_smiles)
    gt_canon = canonicalize_smiles(gt_smiles)
    if pred_canon is None or gt_canon is None:
        return False
    return pred_canon == gt_canon


def parse_model_output(output: str) -> tuple[list[dict], str]:
    """
    Parse model output to extract predicted steps.

    Returns:
        (steps, status) where status is one of:
        - "ok": successfully parsed steps from ## Result section
        - "no_result_section": response has no ## Result section (truncated reasoning)
        - "json_parse_fail": ## Result found but JSON is unparseable (truncated JSON)
    """
    # Check for ## Result section
    has_result_section = "## Result" in output

    if not has_result_section:
        return [], "no_result_section"

    # Extract content after ## Result
    result_content = output.split("## Result")[-1]

    # Try to find JSON block in ```json ... ``` or ``` ... ```
    json_block_patterns = [
        r'```json\s*([\[\{].*?[\]\}])\s*```',
        r'```\s*([\[\{].*?[\]\}])\s*```',
    ]

    raw_json = None
    for pattern in json_block_patterns:
        matches = re.findall(pattern, result_content, re.DOTALL)
        if matches:
            raw_json = matches[-1]
            break

    # Fallback: find bare JSON array or object in result section
    if raw_json is None:
        array_match = re.search(r'(\[\s*\{.*?\}\s*\])', result_content, re.DOTALL)
        if array_match:
            raw_json = array_match.group(1)

    if raw_json is None:
        return [], "json_parse_fail"

    steps = []
    try:
        data = json.loads(raw_json)
        # Handle both array and object with "steps" key
        if isinstance(data, list):
            step_list = data
        elif isinstance(data, dict) and "steps" in data:
            step_list = data["steps"]
        else:
            step_list = []

        for step in step_list:
            if isinstance(step, dict) and "product_smiles" in step:
                steps.append({
                    "step_id": step.get("step_id", len(steps) + 1),
                    "product_smiles": step["product_smiles"],
                })
        if steps:
            return steps, "ok"
        return [], "json_parse_fail"
    except (json.JSONDecodeError, TypeError):
        return [], "json_parse_fail"


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[str] = None) -> float:
    """
    Compute reward score for a single reaction mechanism prediction.

    Args:
        data_source: identifier for the data source (unused, required by verl)
        solution_str: model's generated output string
        ground_truth: JSON string of ground truth steps
            Format: [{"step_id": 0, "products": "SMILES..."}, ...]
        extra_info: optional JSON string with additional info (unused)

    Returns:
        float reward in [-1.0, 1.0]
    """
    # Parse ground truth
    try:
        gt_steps = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    if not gt_steps:
        return 0.0

    # Parse model output (no regex fallback — must have ## Result with valid JSON)
    pred_steps, parse_status = parse_model_output(solution_str)

    if parse_status == "no_result_section":
        # Reasoning truncation/loop: no ## Result section at all
        return -1.0

    if parse_status == "json_parse_fail":
        # ## Result present but JSON is truncated or malformed
        return -0.5

    # Per-step scoring: each GT step scored independently
    num_gt = len(gt_steps)
    step_scores = []
    for i, gt_step in enumerate(gt_steps):
        gt_product = gt_step.get("products", "")
        if i < len(pred_steps):
            pred_product = pred_steps[i].get("product_smiles", "")
            if compare_smiles(pred_product, gt_product):
                step_scores.append(1.0)   # correct match
            elif is_valid_smiles(pred_product):
                step_scores.append(0.1)   # wrong but valid SMILES
            else:
                step_scores.append(0.0)   # invalid SMILES
        else:
            step_scores.append(0.0)       # missing step

    return sum(step_scores) / num_gt
