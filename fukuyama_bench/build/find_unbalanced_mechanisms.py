"""
Script to find unbalanced mechanism steps in mechanism.json files.
Checks if electron_number is conserved between reactants and products.
"""

import os
import re
import json
from datetime import datetime


def natural_sort_key(s):
    """
    Natural sorting key function.
    Handles folder names like A001, B027_a, C032_b correctly.
    
    Examples:
        A1 < A2 < A10
        B027_a < B027_b < B028
        C032_a < C032_b < C033
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def check_mechanism_balance(mechanism_file):
    """
    Check electron balance for all mechanism steps in a mechanism.json file.
    
    Args:
        mechanism_file: Path to mechanism.json file
        
    Returns:
        list: List of unbalanced steps with details
    """
    try:
        with open(mechanism_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return [{"error": str(e)}]
    
    unbalanced_steps = []
    
    if "mechanism" not in data:
        return []
    
    for step in data["mechanism"]:
        mechanism_id = step.get("mechanism_id", "?")
        reactants = step.get("reactants", [])
        products = step.get("products", [])
        
        # Calculate total electrons and charges for reactants
        reactant_electrons = 0
        reactant_charge = 0
        reactant_details = []
        
        for mol in reactants:
            e_num = mol.get("electron_number", 0)
            charge = mol.get("charge", 0)
            smiles = mol.get("smiles", "")
            
            if e_num is None:
                e_num = 0
            if charge is None:
                charge = 0
                
            reactant_electrons += e_num
            reactant_charge += charge
            reactant_details.append({
                "smiles": smiles,
                "electron_number": e_num,
                "charge": charge
            })
        
        # Calculate total electrons and charges for products
        product_electrons = 0
        product_charge = 0
        product_details = []
        
        for mol in products:
            e_num = mol.get("electron_number", 0)
            charge = mol.get("charge", 0)
            smiles = mol.get("smiles", "")
            
            if e_num is None:
                e_num = 0
            if charge is None:
                charge = 0
                
            product_electrons += e_num
            product_charge += charge
            product_details.append({
                "smiles": smiles,
                "electron_number": e_num,
                "charge": charge
            })
        
        # Check if electrons are balanced
        electron_diff = product_electrons - reactant_electrons
        charge_diff = product_charge - reactant_charge
        
        if electron_diff != 0 or charge_diff != 0:
            unbalanced_steps.append({
                "mechanism_id": mechanism_id,
                "reactant_electrons": reactant_electrons,
                "product_electrons": product_electrons,
                "electron_diff": electron_diff,
                "reactant_charge": reactant_charge,
                "product_charge": product_charge,
                "charge_diff": charge_diff,
                "reactants": reactant_details,
                "products": product_details
            })
    
    return unbalanced_steps


def batch_find_unbalanced(base_dir, output_file=None):
    """
    Batch check all mechanism.json files for unbalanced reactions.
    
    Args:
        base_dir: Base directory containing mechanism subdirectories
        output_file: Optional output file path for results
        
    Returns:
        dict: Results with all unbalanced mechanisms
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        "timestamp": timestamp,
        "total_folders": 0,
        "folders_with_issues": 0,
        "total_unbalanced_steps": 0,
        "unbalanced_mechanisms": []
    }
    
    print("=" * 80)
    print("Finding Unbalanced Mechanism Steps")
    print(f"Base directory: {base_dir}")
    print("=" * 80)
    
    # Get all subdirectories (supports names like A001, B027_a, C032_b)
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    subdirs.sort(key=natural_sort_key)
    
    print(f"Found {len(subdirs)} subdirectories\n")
    
    for subdir_name in subdirs:
        subdir_path = os.path.join(base_dir, subdir_name)
        mechanism_file = os.path.join(subdir_path, 'mechanism.json')
        
        results["total_folders"] += 1
        
        if not os.path.exists(mechanism_file):
            continue
        
        unbalanced = check_mechanism_balance(mechanism_file)
        
        if unbalanced:
            results["folders_with_issues"] += 1
            results["total_unbalanced_steps"] += len(unbalanced)
            
            folder_result = {
                "folder": subdir_name,
                "unbalanced_steps": unbalanced
            }
            results["unbalanced_mechanisms"].append(folder_result)
            
            # Print details
            print(f"\n[{subdir_name}] Found {len(unbalanced)} unbalanced step(s):")
            print("-" * 60)
            
            for step in unbalanced:
                if "error" in step:
                    print(f"  Error: {step['error']}")
                    continue
                    
                print(f"  Step {step['mechanism_id']}:")
                print(f"    Reactants: electrons={step['reactant_electrons']}, charge={step['reactant_charge']}")
                for r in step['reactants']:
                    print(f"      - {r['smiles'][:50]}... (e={r['electron_number']}, q={r['charge']})" 
                          if len(r['smiles']) > 50 else f"      - {r['smiles']} (e={r['electron_number']}, q={r['charge']})")
                
                print(f"    Products:  electrons={step['product_electrons']}, charge={step['product_charge']}")
                for p in step['products']:
                    print(f"      - {p['smiles'][:50]}... (e={p['electron_number']}, q={p['charge']})"
                          if len(p['smiles']) > 50 else f"      - {p['smiles']} (e={p['electron_number']}, q={p['charge']})")
                
                print(f"    Difference: electrons={step['electron_diff']:+d}, charge={step['charge_diff']:+d}")
                print()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total folders scanned: {results['total_folders']}")
    print(f"Folders with unbalanced steps: {results['folders_with_issues']}")
    print(f"Total unbalanced steps: {results['total_unbalanced_steps']}")
    
    # Save results to file
    if output_file is None:
        output_file = os.path.join(base_dir, f'unbalanced_mechanisms_{timestamp}.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Get the mechanisms folder path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mechanisms_dir = os.path.join(script_dir, "mechanisms")
    
    if not os.path.exists(mechanisms_dir):
        print(f"Error: mechanisms folder not found at {mechanisms_dir}")
        exit(1)
    
    # Run the check
    results = batch_find_unbalanced(mechanisms_dir)
    
    # Print folders with issues for quick reference
    if results["unbalanced_mechanisms"]:
        print("\nFolders with unbalanced mechanisms:")
        for item in results["unbalanced_mechanisms"]:
            folder = item["folder"]
            steps = [str(s.get("mechanism_id", "?")) for s in item["unbalanced_steps"] if "error" not in s]
            print(f"  - {folder}: steps {', '.join(steps)}")

