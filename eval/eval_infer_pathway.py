#!/usr/bin/env python3
"""
Evaluation script for complete pathway mechanism prediction results (New Format).

This script evaluates the inference results from run_infer_pathway.py
by comparing predicted pathways with ground truth using RDKit canonicalization.
Supports top-K evaluation metrics.

New format differences:
- Model predictions use atom-mapped SMILES (e.g., [CH3:1][O:2][C:3]...)
- product_smiles is a single string, multiple molecules separated by '.'
- Ground truth products are a list of plain SMILES without atom mapping

Checkpoint-based evaluation:
- Uses ckpt.txt files from mechanisms folder to define checkpoints
- Each line in ckpt.txt is a checkpoint with comma-separated equivalent step IDs
- Model predictions are evaluated by matching checkpoints in order
- This handles cases where model predicts finer-grained steps than ground truth
"""

import os
import json
import re
import argparse
import glob
import math
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from rdkit import Chem


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric using the unbiased estimator from the Codex paper.

    pass@k = 1 - C(n-c, k) / C(n, k)

    This gives the probability that at least one of k randomly selected samples is correct.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to consider

    Returns:
        pass@k probability (between 0 and 1)
    """
    if n < k:
        # If we have fewer samples than k, return 1 if any correct, else 0
        return 1.0 if c > 0 else 0.0

    if c == 0:
        return 0.0

    if c >= n:
        return 1.0

    # Calculate 1 - C(n-c, k) / C(n, k)
    # Use log to avoid overflow for large numbers
    # C(n-c, k) / C(n, k) = product_{i=0}^{k-1} (n-c-i) / (n-i)

    if n - c < k:
        # Not enough incorrect samples to fill k slots, so at least one must be correct
        return 1.0

    # Calculate the product iteratively to avoid overflow
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)

    return 1.0 - result


def remove_atom_mapping(smiles: str) -> str:
    """
    Remove atom mapping numbers from SMILES.
    e.g., [CH3:1][O:2] -> [CH3][O] -> CO
    """
    if not smiles:
        return smiles
    
    # Pattern to match atom mapping numbers like :1, :2, :10, etc.
    # Handles both [C:1] and [CH3:1] formats
    pattern = r':\d+'
    return re.sub(pattern, '', smiles)


def load_checkpoints(mechanisms_dir: str, case_id: str) -> Optional[List[List[int]]]:
    """
    Load checkpoints from ckpt.txt file for a given case.
    
    Args:
        mechanisms_dir: Path to mechanisms directory
        case_id: Case ID (e.g., 'A074')
    
    Returns:
        List of checkpoints, where each checkpoint is a list of equivalent step IDs.
        Returns None if ckpt.txt not found (fallback to step-by-step evaluation).
    """
    ckpt_path = os.path.join(mechanisms_dir, case_id, "ckpt.txt")
    
    if not os.path.exists(ckpt_path):
        return None
    
    checkpoints = []
    with open(ckpt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse comma-separated step IDs
            step_ids = [int(x.strip()) for x in line.split(',') if x.strip()]
            if step_ids:
                checkpoints.append(step_ids)
    
    return checkpoints if checkpoints else None


def get_checkpoint_products(gt_pathway: List[Dict], step_ids: List[int]) -> List[List[str]]:
    """
    Get products from ground truth pathway for given step IDs.
    
    Args:
        gt_pathway: Ground truth pathway steps
        step_ids: List of step IDs (1-indexed) that are equivalent for this checkpoint
    
    Returns:
        List of product lists, one for each step ID
    """
    result = []
    for step_id in step_ids:
        idx = step_id - 1  # Convert to 0-indexed
        if 0 <= idx < len(gt_pathway):
            products = gt_pathway[idx].get("products", [])
            result.append(products)
    return result


def extract_pathway_from_response(text: str) -> Optional[List[Dict]]:
    """
    Extract pathway steps from model output.
    Expected format: JSON array with step objects containing product_smiles.
    """
    if not text:
        return None
    
    # Method 1: Find JSON array in ```json ... ``` code block
    json_blocks = re.findall(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
    for json_str in reversed(json_blocks):
        try:
            result = json.loads(json_str)
            if isinstance(result, list) and len(result) > 0:
                # Validate structure
                if any(isinstance(step, dict) and ("product_smiles" in step or "products" in step) for step in result):
                    return result
        except json.JSONDecodeError:
            continue
    
    # Method 2: Find any JSON array in the text (after ## Result section)
    result_section = re.search(r"##\s*Result\s*\n", text)
    search_text = text[result_section.end():] if result_section else text
    
    # Find array start positions
    start_positions = [m.start() for m in re.finditer(r'\[', search_text)]
    
    for start in start_positions:
        bracket_count = 0
        for i in range(start, len(search_text)):
            if search_text[i] == '[':
                bracket_count += 1
            elif search_text[i] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    try:
                        candidate = search_text[start:i+1]
                        result = json.loads(candidate)
                        if isinstance(result, list) and len(result) > 0:
                            if any(isinstance(step, dict) and ("product_smiles" in step or "products" in step) for step in result):
                                return result
                    except json.JSONDecodeError:
                        pass
                    break
    
    # Method 3: Try to parse individual step objects
    steps = []
    step_matches = re.finditer(r'\{[^{}]*"step_id"[^{}]*\}', text, re.DOTALL)
    for match in step_matches:
        try:
            step = json.loads(match.group())
            if "product_smiles" in step or "products" in step:
                steps.append(step)
        except json.JSONDecodeError:
            continue
    
    if steps:
        return steps
    
    return None


def get_product_smiles_from_step(step: Dict) -> List[str]:
    """
    Extract product SMILES from a step dictionary.
    Handles atom-mapped SMILES by removing mapping numbers.
    """
    if "product_smiles" in step:
        smiles = step["product_smiles"]
        if isinstance(smiles, str):
            # Remove atom mapping and split by '.' for multiple molecules
            unmapped = remove_atom_mapping(smiles)
            return [s.strip() for s in unmapped.split(".") if s.strip()]
        elif isinstance(smiles, list):
            # If already a list, remove mapping from each
            return [remove_atom_mapping(s) for s in smiles if s]
    elif "products" in step:
        products = step["products"]
        if isinstance(products, list):
            result = []
            for p in products:
                if isinstance(p, str):
                    result.append(remove_atom_mapping(p))
                elif isinstance(p, dict) and "smiles" in p:
                    result.append(remove_atom_mapping(p["smiles"]))
            return result
    return []


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES using RDKit."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def canonicalize_smiles_set(smiles_list: List[str]) -> set:
    """Canonicalize a list of SMILES and return as set."""
    result = set()
    for smi in smiles_list:
        canon = canonicalize_smiles(smi)
        if canon:
            result.add(canon)
    return result


def compare_step_products(
    pred_products: List[str], 
    gt_products: List[str],
    lenient: bool = True
) -> Tuple[bool, str]:
    """
    Compare predicted products with ground truth products.
    
    Args:
        pred_products: List of predicted SMILES
        gt_products: List of ground truth SMILES
        lenient: If True, prediction is correct if it's a subset of ground truth
                 (i.e., model predicted main products but not leaving groups)
    
    Returns:
        Tuple of (is_match, error_type)
    """
    pred_canon = canonicalize_smiles_set(pred_products)
    gt_canon = canonicalize_smiles_set(gt_products)
    
    if not pred_canon and not gt_canon:
        return False, 'both_invalid'
    if not pred_canon:
        return False, 'invalid_pred'
    if not gt_canon:
        return False, 'invalid_gt'
    
    # Check if predicted products match ground truth exactly
    if pred_canon == gt_canon:
        return True, None
    
    # Lenient mode: prediction is correct if it's a non-empty subset of ground truth
    # This handles cases where model predicts main products but not leaving groups
    if lenient and pred_canon.issubset(gt_canon) and len(pred_canon) > 0:
        return True, 'subset_match'
    
    # Check if prediction is a superset (model predicted extra products)
    if gt_canon.issubset(pred_canon):
        return False, 'superset_match'
    
    # Partial overlap
    if pred_canon & gt_canon:  # non-empty intersection
        return False, 'partial_match'
    
    return False, 'mismatch'


def compare_pathways(
    pred_pathway: List[Dict],
    gt_pathway: List[Dict],
    lenient: bool = True,
    checkpoints: Optional[List[List[int]]] = None
) -> Dict:
    """
    Compare predicted pathway with ground truth pathway.
    
    Args:
        pred_pathway: List of predicted step dictionaries
        gt_pathway: List of ground truth step dictionaries
        lenient: If True, allow subset matching for products
        checkpoints: Optional list of checkpoints from ckpt.txt.
                    Each checkpoint is a list of equivalent step IDs.
    
    Returns:
        Dictionary with comparison metrics
    """
    if not pred_pathway:
        num_checkpoints = len(checkpoints) if checkpoints else len(gt_pathway)
        return {
            "valid": False,
            "error": "empty_prediction",
            "step_matches": [],
            "checkpoint_matches": [],
            "exact_match": False,
            "steps_correct": 0,
            "checkpoints_correct": 0,
            "steps_total": len(gt_pathway),
            "checkpoints_total": num_checkpoints,
            "length_match": False,
            "use_checkpoints": checkpoints is not None
        }
    
    gt_length = len(gt_pathway)
    pred_length = len(pred_pathway)
    
    # Use checkpoint-based evaluation if checkpoints are provided
    if checkpoints:
        return compare_pathways_with_checkpoints(
            pred_pathway, gt_pathway, checkpoints, lenient
        )
    
    # Fallback to step-by-step evaluation
    length_match = gt_length == pred_length
    
    step_matches = []
    steps_correct = 0
    
    # Compare step by step
    for i in range(max(gt_length, pred_length)):
        if i >= gt_length:
            step_matches.append({
                "step": i + 1,
                "match": False,
                "error": "extra_step"
            })
        elif i >= pred_length:
            step_matches.append({
                "step": i + 1,
                "match": False,
                "error": "missing_step"
            })
        else:
            pred_products = get_product_smiles_from_step(pred_pathway[i])
            gt_products = gt_pathway[i].get("products", [])
            
            is_match, error_type = compare_step_products(pred_products, gt_products, lenient)
            
            step_matches.append({
                "step": i + 1,
                "match": is_match,
                "error": error_type,
                "pred_products": pred_products,
                "gt_products": gt_products
            })
            
            if is_match:
                steps_correct += 1
    
    exact_match = length_match and steps_correct == gt_length
    
    return {
        "valid": True,
        "exact_match": exact_match,
        "length_match": length_match,
        "steps_correct": steps_correct,
        "checkpoints_correct": steps_correct,  # Same as steps when no checkpoints
        "steps_total": gt_length,
        "checkpoints_total": gt_length,
        "pred_length": pred_length,
        "step_accuracy": steps_correct / gt_length if gt_length > 0 else 0,
        "step_matches": step_matches,
        "checkpoint_matches": step_matches,  # Same as step_matches when no checkpoints
        "use_checkpoints": False
    }


def compare_pathways_with_checkpoints(
    pred_pathway: List[Dict],
    gt_pathway: List[Dict],
    checkpoints: List[List[int]],
    lenient: bool = True
) -> Dict:
    """
    Compare predicted pathway with ground truth using checkpoint-based evaluation.
    
    This method allows model to predict finer-grained steps than ground truth.
    Each checkpoint can be satisfied by matching any of its equivalent step IDs.
    Checkpoints must be matched in order using the predicted steps in order.
    
    Args:
        pred_pathway: List of predicted step dictionaries
        gt_pathway: List of ground truth step dictionaries
        checkpoints: List of checkpoints, each containing equivalent step IDs
        lenient: If True, allow subset matching for products
    
    Returns:
        Dictionary with comparison metrics
    """
    num_checkpoints = len(checkpoints)
    pred_length = len(pred_pathway)
    
    checkpoint_matches = []
    checkpoints_correct = 0
    
    # Track which predicted step matched each checkpoint
    pred_idx = 0  # Current position in predicted pathway
    
    for ckpt_idx, ckpt_step_ids in enumerate(checkpoints):
        # Get all possible products for this checkpoint (from equivalent GT steps)
        possible_products_list = get_checkpoint_products(gt_pathway, ckpt_step_ids)
        
        ckpt_match = {
            "checkpoint": ckpt_idx + 1,
            "equivalent_steps": ckpt_step_ids,
            "match": False,
            "matched_pred_step": None,
            "matched_gt_step": None,
            "error": None
        }
        
        # Search through remaining predicted steps in order
        found = False
        while pred_idx < pred_length and not found:
            pred_products = get_product_smiles_from_step(pred_pathway[pred_idx])
            
            # Check if this predicted step matches any equivalent GT step
            for gt_step_id, gt_products in zip(ckpt_step_ids, possible_products_list):
                is_match, error_type = compare_step_products(pred_products, gt_products, lenient)
                
                if is_match:
                    ckpt_match["match"] = True
                    ckpt_match["matched_pred_step"] = pred_idx + 1
                    ckpt_match["matched_gt_step"] = gt_step_id
                    ckpt_match["pred_products"] = pred_products
                    ckpt_match["gt_products"] = gt_products
                    checkpoints_correct += 1
                    found = True
                    pred_idx += 1  # Move to next predicted step
                    break
            
            if not found:
                pred_idx += 1  # Try next predicted step
        
        if not found:
            ckpt_match["error"] = "checkpoint_not_found"
        
        checkpoint_matches.append(ckpt_match)
    
    # All checkpoints passed = exact match
    exact_match = checkpoints_correct == num_checkpoints
    
    return {
        "valid": True,
        "exact_match": exact_match,
        "length_match": pred_length == len(gt_pathway),
        "steps_correct": checkpoints_correct,  # For backward compatibility
        "checkpoints_correct": checkpoints_correct,
        "steps_total": len(gt_pathway),
        "checkpoints_total": num_checkpoints,
        "pred_length": pred_length,
        "step_accuracy": checkpoints_correct / num_checkpoints if num_checkpoints > 0 else 0,
        "checkpoint_accuracy": checkpoints_correct / num_checkpoints if num_checkpoints > 0 else 0,
        "step_matches": checkpoint_matches,  # For backward compatibility
        "checkpoint_matches": checkpoint_matches,
        "use_checkpoints": True
    }


def load_inference_results(
    results_dir: str,
    model_filter: Optional[str] = None,
    prefix_filter: Optional[str] = None
) -> List[Dict]:
    """Load inference results from JSON files."""
    result_files = []
    
    pattern = os.path.join(results_dir, "*.json")
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        
        if model_filter and model_filter.lower() not in filename.lower():
            continue
        
        if prefix_filter:
            prefix_upper = prefix_filter.upper()
            if f"_{prefix_upper}_" not in filename.upper():
                continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['source_file'] = filepath
        result_files.append(data)
    
    return result_files


def filter_results_by_prefix(result_data: Dict, prefix: str) -> Dict:
    """Filter results to only include cases with the specified prefix."""
    prefix_upper = prefix.upper()
    filtered_results = [
        r for r in result_data.get("results", [])
        if r.get("case_id", "").upper().startswith(prefix_upper)
    ]
    
    return {
        **result_data,
        "results": filtered_results,
        "total_samples": len(filtered_results),
        "filtered_prefix": prefix_upper
    }


def evaluate_results(
    result_data: Dict, 
    lenient: bool = True,
    mechanisms_dir: Optional[str] = None,
    top_k_values: List[int] = None
) -> Dict:
    """
    Evaluate inference results for a single model with top-K metrics.
    
    Args:
        result_data: Dictionary containing model results
        lenient: If True, allow subset matching for products
        mechanisms_dir: Path to mechanisms directory for loading checkpoints
        top_k_values: List of K values for top-K accuracy (default: [1, 3, 5, 8])
    
    Returns:
        Dictionary with evaluation metrics
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5, 8]
    
    results = result_data.get("results", [])
    model_name = result_data.get("model", "Unknown")
    sample_size = result_data.get("sample_size", result_data.get("top_k", 1))
    
    # Counters
    total_cases = 0
    cases_with_checkpoints = 0

    # Error categories
    extraction_errors = 0
    api_errors = 0
    
    # Per-case statistics
    case_evaluations = []
    
    for item in results:
        case_id = item.get("case_id")
        gt_pathway = item.get("ground_truth_pathway", [])
        predictions = item.get("predictions", [])
        
        total_cases += 1
        
        # Load checkpoints for this case
        checkpoints = None
        if mechanisms_dir:
            checkpoints = load_checkpoints(mechanisms_dir, case_id)
        
        num_checkpoints = len(checkpoints) if checkpoints else len(gt_pathway)
        if checkpoints:
            cases_with_checkpoints += 1
        
        case_eval = {
            "case_id": case_id,
            "ground_truth_steps": len(gt_pathway),
            "num_checkpoints": num_checkpoints,
            "has_checkpoints": checkpoints is not None,
            "k_evaluations": [],
            "best_exact_match": False,
            "best_checkpoints_correct": 0,
            "best_k": None,
            # For pass@k calculation
            "n_valid_samples": 0,  # Total valid samples (no API/extraction error)
            "n_exact_correct": 0,  # Samples with exact match
            "n_any_ckpt_correct": 0  # Samples with any checkpoint correct
        }

        best_checkpoints_correct = 0
        best_exact_match = False
        best_k = None

        # Counters for pass@k
        n_valid = 0
        n_exact_correct = 0
        n_any_ckpt_correct = 0

        for pred in predictions:
            k = pred.get("k", 1)
            error = pred.get("error")
            response = pred.get("model_response", "")

            k_eval = {
                "k": k,
                "error": None,
                "pathway_comparison": None,
                "is_exact_match": False,
                "is_any_ckpt_correct": False
            }

            if error:
                k_eval["error"] = "api_error"
                api_errors += 1
                case_eval["k_evaluations"].append(k_eval)
                continue

            # Extract pathway from response
            pred_pathway = extract_pathway_from_response(response)

            if not pred_pathway:
                k_eval["error"] = "extraction_error"
                extraction_errors += 1
                case_eval["k_evaluations"].append(k_eval)
                continue

            # This is a valid sample (no API/extraction error)
            n_valid += 1

            # Compare pathways with checkpoints
            comparison = compare_pathways(pred_pathway, gt_pathway, lenient, checkpoints)
            k_eval["pathway_comparison"] = comparison

            # Track exact match
            if comparison["exact_match"]:
                n_exact_correct += 1
                k_eval["is_exact_match"] = True
                best_exact_match = True
                if best_k is None:
                    best_k = k

            # Track any checkpoint correct
            ckpt_correct = comparison.get("checkpoints_correct", comparison.get("steps_correct", 0))
            if ckpt_correct > 0:
                n_any_ckpt_correct += 1
                k_eval["is_any_ckpt_correct"] = True

            # Update best results
            if ckpt_correct > best_checkpoints_correct:
                best_checkpoints_correct = ckpt_correct
                best_k = k

            case_eval["k_evaluations"].append(k_eval)

        case_eval["best_exact_match"] = best_exact_match
        case_eval["best_checkpoints_correct"] = best_checkpoints_correct
        case_eval["best_k"] = best_k
        case_eval["n_valid_samples"] = n_valid
        case_eval["n_exact_correct"] = n_exact_correct
        case_eval["n_any_ckpt_correct"] = n_any_ckpt_correct

        case_evaluations.append(case_eval)
    
    # Calculate final metrics
    metrics = {
        "total_cases": total_cases,
        "cases_with_checkpoints": cases_with_checkpoints,
        "sample_size": sample_size,
    }

    # Calculate pass@k metrics using unbiased estimator
    # For each case: pass@k = 1 - C(n-c, k) / C(n, k)
    # Then average across all cases
    pass_k_exact_match_sum = {k: 0.0 for k in top_k_values}
    pass_k_any_ckpt_correct_sum = {k: 0.0 for k in top_k_values}

    for case_eval in case_evaluations:
        n = case_eval["n_valid_samples"]
        c_exact = case_eval["n_exact_correct"]
        c_any_ckpt = case_eval["n_any_ckpt_correct"]

        for k in top_k_values:
            # pass@k for exact match
            pass_k_exact_match_sum[k] += pass_at_k(n, c_exact, k)
            # pass@k for any checkpoint correct
            pass_k_any_ckpt_correct_sum[k] += pass_at_k(n, c_any_ckpt, k)

    for k in top_k_values:
        metrics[f"pass@{k}_exact_match"] = pass_k_exact_match_sum[k] / total_cases if total_cases > 0 else 0
        metrics[f"pass@{k}_any_checkpoint_correct"] = pass_k_any_ckpt_correct_sum[k] / total_cases if total_cases > 0 else 0

    # Calculate average checkpoint accuracy across all cases
    total_checkpoint_accuracy = 0
    valid_cases = 0
    for case_eval in case_evaluations:
        num_ckpts = case_eval["num_checkpoints"]
        if num_ckpts > 0 and case_eval["best_checkpoints_correct"] > 0:
            total_checkpoint_accuracy += case_eval["best_checkpoints_correct"] / num_ckpts
            valid_cases += 1

    metrics["avg_best_checkpoint_accuracy"] = total_checkpoint_accuracy / valid_cases if valid_cases > 0 else 0
    
    return {
        "model": model_name,
        "source_file": result_data.get("source_file"),
        "metrics": metrics,
        "errors": {
            "api_errors": api_errors,
            "extraction_errors": extraction_errors
        },
        "case_evaluations": case_evaluations
    }


def generate_report(evaluations: List[Dict], output_path: str, lenient: bool = True, 
                    top_k_values: List[int] = None):
    """Generate evaluation report and save to file."""
    
    if top_k_values is None:
        top_k_values = [1, 3, 5, 8]
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    match_mode = "Lenient (subset match allowed)" if lenient else "Strict (exact match required)"
    
    report_lines = [
        "=" * 100,
        "Pathway Mechanism Prediction Evaluation Report (pass@k Analysis)",
        f"Generated: {timestamp}",
        f"Match Mode: {match_mode}",
        f"K Values: {top_k_values}",
        "=" * 100,
        "",
        "NOTE: pass@k = 1 - C(n-c,k)/C(n,k) where n=total samples, c=correct samples",
        "      This is the unbiased estimator from the Codex/HumanEval paper.",
        ""
    ]

    # Summary table with pass@k
    report_lines.append("EXACT MATCH pass@k SUMMARY")
    report_lines.append("-" * 100)

    # Build header dynamically based on top_k_values
    header_parts = [f"{'Model':<30}", f"{'Cases':<8}"]
    for k in top_k_values:
        header_parts.append(f"{'pass@'+str(k):<10}")
    report_lines.append(" ".join(header_parts))
    report_lines.append("-" * 100)

    for eval_data in evaluations:
        model = eval_data["model"][:28]
        metrics = eval_data["metrics"]
        total_cases = metrics.get("total_cases", 0)

        row_parts = [f"{model:<30}", f"{total_cases:<8}"]
        for k in top_k_values:
            rate = metrics.get(f"pass@{k}_exact_match", 0)
            row_parts.append(f"{rate:<10.1%}")
        report_lines.append(" ".join(row_parts))

    report_lines.append("-" * 100)
    report_lines.append("")

    # Any Checkpoint Correct Summary
    report_lines.append("ANY CHECKPOINT CORRECT pass@k SUMMARY")
    report_lines.append("-" * 100)
    header_parts = [f"{'Model':<30}", f"{'Cases':<8}"]
    for k in top_k_values:
        header_parts.append(f"{'pass@'+str(k):<10}")
    report_lines.append(" ".join(header_parts))
    report_lines.append("-" * 100)

    for eval_data in evaluations:
        model = eval_data["model"][:28]
        metrics = eval_data["metrics"]
        total_cases = metrics.get("total_cases", 0)

        row_parts = [f"{model:<30}", f"{total_cases:<8}"]
        for k in top_k_values:
            rate = metrics.get(f"pass@{k}_any_checkpoint_correct", 0)
            row_parts.append(f"{rate:<10.1%}")
        report_lines.append(" ".join(row_parts))

    report_lines.append("-" * 100)
    report_lines.append("")
    
    # Detailed results for each model
    for eval_data in evaluations:
        model = eval_data["model"]
        metrics = eval_data["metrics"]
        sample_size = metrics.get("sample_size", 1)
        
        report_lines.append("")
        report_lines.append("=" * 100)
        report_lines.append(f"MODEL: {model}")
        report_lines.append("=" * 100)
        report_lines.append(f"Source: {eval_data.get('source_file', 'N/A')}")
        report_lines.append(f"Sample Size: {sample_size}")
        report_lines.append(f"Total Cases: {metrics['total_cases']}")
        report_lines.append(f"Cases with Checkpoints: {metrics.get('cases_with_checkpoints', 0)}")
        report_lines.append("")
        
        # pass@k metrics table
        report_lines.append("pass@k METRICS:")
        header = f"{'K':<5} {'Exact Match':<15} {'Any Ckpt Correct':<18}"
        report_lines.append(header)
        report_lines.append("-" * 40)

        for k in top_k_values:
            exact_rate = metrics.get(f"pass@{k}_exact_match", 0)
            any_rate = metrics.get(f"pass@{k}_any_checkpoint_correct", 0)

            row = f"{k:<5} {exact_rate:<15.2%} {any_rate:<18.2%}"
            report_lines.append(row)
        
        report_lines.append("")
        avg_acc = metrics.get('avg_best_checkpoint_accuracy', 0)
        report_lines.append(f"Average Best Checkpoint Accuracy: {avg_acc:.2%}")
    
    report_lines.append("")
    report_lines.append("-" * 80)
    
    # Detailed results for each model
    for eval_data in evaluations:
        model = eval_data["model"]
        errors = eval_data["errors"]
        case_evals = eval_data["case_evaluations"]
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append(f"MODEL: {model}")
        report_lines.append("=" * 80)
        report_lines.append(f"Source: {eval_data.get('source_file', 'N/A')}")
        report_lines.append("")
        
        report_lines.append("ERROR BREAKDOWN:")
        report_lines.append(f"  API errors:         {errors['api_errors']}")
        report_lines.append(f"  Extraction errors:  {errors['extraction_errors']}")
        report_lines.append("")
        
        # Per-case breakdown with pass@k info
        report_lines.append("PER-CASE STATISTICS:")
        report_lines.append("  Format: Case ID: best_ckpts/total (acc%) | n=valid_samples, c_exact=exact_correct, c_any=any_ckpt_correct")
        report_lines.append("")

        for case_eval in sorted(case_evals, key=lambda x: x["case_id"]):
            case_id = case_eval["case_id"]
            num_ckpts = case_eval.get("num_checkpoints", case_eval.get("ground_truth_steps", 0))
            best_correct = case_eval.get("best_checkpoints_correct", 0)
            has_ckpt = "[ckpt]" if case_eval.get("has_checkpoints", False) else ""

            # pass@k info
            n_valid = case_eval.get("n_valid_samples", 0)
            n_exact = case_eval.get("n_exact_correct", 0)
            n_any = case_eval.get("n_any_ckpt_correct", 0)

            accuracy = best_correct / num_ckpts if num_ckpts > 0 else 0

            report_lines.append(
                f"  {case_id}: {best_correct}/{num_ckpts} ({accuracy:.0%}) | n={n_valid}, c_exact={n_exact}, c_any={n_any} {has_ckpt}"
            )
        
        report_lines.append("")
        
        # Sample errors
        report_lines.append("SAMPLE ERRORS (first 5 with errors):")
        error_count = 0
        for case_eval in case_evals:
            if error_count >= 5:
                break
            for k_eval in case_eval["k_evaluations"]:
                if k_eval.get("error") and error_count < 5:
                    report_lines.append(
                        f"  [{case_eval['case_id']}] k={k_eval['k']}: {k_eval['error']}"
                    )
                    error_count += 1
                    break
        
        if error_count == 0:
            report_lines.append("  (No extraction/API errors)")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pathway mechanism prediction results (Checkpoint-based)"
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default=None,
        help="Path to a specific result JSON file (overrides --results_dir and --model)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../test_result/pathway",
        help="Directory containing inference result JSON files"
    )
    parser.add_argument(
        "--mechanisms_dir",
        type=str,
        default="../fukuyama_bench/mechanisms",
        help="Directory containing mechanism data with ckpt.txt files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../test_result/eval_pathway_report.txt",
        help="Path to save evaluation report"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter results by model name (ignored if --result_file is provided)"
    )
    parser.add_argument(
        "--prefix", "-p",
        type=str,
        choices=["A", "B", "C", "a", "b", "c"],
        default=None,
        help="Filter results by case prefix (A, B, or C)"
    )
    parser.add_argument(
        "--save_detailed",
        action="store_true",
        help="Save detailed evaluation results as JSON"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict matching (require exact product match, not subset)"
    )
    parser.add_argument(
        "--no_checkpoints",
        action="store_true",
        help="Disable checkpoint-based evaluation, use step-by-step instead"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 8],
        help="K values for Top-K accuracy evaluation (default: 1 3 5 8)"
    )
    
    args = parser.parse_args()
    
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve paths relative to script directory
    if args.result_file and not os.path.isabs(args.result_file):
        args.result_file = os.path.join(script_dir, args.result_file)
    if not os.path.isabs(args.results_dir):
        args.results_dir = os.path.join(script_dir, args.results_dir)
    if not os.path.isabs(args.mechanisms_dir):
        args.mechanisms_dir = os.path.join(script_dir, args.mechanisms_dir)
    if not os.path.isabs(args.output_path):
        args.output_path = os.path.join(script_dir, args.output_path)

    prefix = args.prefix.upper() if args.prefix else None

    # Load result files
    if args.result_file:
        # Direct file path specified
        if not os.path.exists(args.result_file):
            print(f"Error: Result file not found: {args.result_file}")
            return
        print(f"Loading result file: {args.result_file}")
        with open(args.result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data['source_file'] = args.result_file
        result_files = [data]
    else:
        # Load from directory
        if not os.path.exists(args.results_dir):
            print(f"Error: Results directory not found: {args.results_dir}")
            print("Please run run_infer_pathway.py first to generate inference results.")
            return

        print(f"Loading inference results from: {args.results_dir}")
        if args.model:
            print(f"Filtering by model: {args.model}")
        result_files = load_inference_results(args.results_dir, args.model, prefix)

    if prefix:
        print(f"Filtering by prefix: {prefix}")

    if not result_files:
        print(f"No result files found")
        if not args.result_file:
            print(f"  Directory: {args.results_dir}")
            if args.model:
                print(f"  (filtered by model: {args.model})")
        if prefix:
            print(f"  (filtered by prefix: {prefix})")
        return
    
    print(f"Found {len(result_files)} result file(s)")
    
    # Determine matching mode
    lenient = not args.strict
    if lenient:
        print("Using lenient matching mode (subset of products is considered correct)")
    else:
        print("Using strict matching mode (exact product match required)")
    
    # Determine checkpoint mode
    mechanisms_dir = None
    if not args.no_checkpoints:
        if os.path.exists(args.mechanisms_dir):
            mechanisms_dir = args.mechanisms_dir
            print(f"Using checkpoint-based evaluation from: {args.mechanisms_dir}")
        else:
            print(f"Warning: Mechanisms directory not found: {args.mechanisms_dir}")
            print("Falling back to step-by-step evaluation")
    else:
        print("Checkpoint-based evaluation disabled, using step-by-step")
    
    # Evaluate each result file
    evaluations = []
    for result_data in result_files:
        if prefix and not result_data.get("prefix"):
            result_data = filter_results_by_prefix(result_data, prefix)
        
        model_name = result_data.get('model', 'Unknown')
        file_prefix = result_data.get('prefix') or result_data.get('filtered_prefix') or 'all'
        sample_size = result_data.get('sample_size', result_data.get('top_k', 1))
        print(f"\nEvaluating: {model_name} (prefix: {file_prefix}, sample_size: {sample_size})...")
        
        eval_result = evaluate_results(result_data, lenient, mechanisms_dir, args.top_k)
        evaluations.append(eval_result)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print("\nGenerating report...")
    generate_report(evaluations, args.output_path, lenient, args.top_k)
    print(f"\nReport saved to: {args.output_path}")
    
    if args.save_detailed:
        detailed_path = args.output_path.replace(".txt", "_detailed.json")
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(evaluations, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to: {detailed_path}")


if __name__ == "__main__":
    main()
