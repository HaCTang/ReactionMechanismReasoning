"""
Generate mapped reaction figures for each step in mechanism JSON files.
This script processes all mapped_rxn.json files in the mechanisms directory
and generates reaction images for each step.
"""

import json
from pathlib import Path
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions


def generate_reaction_image(mapped_rxn_smiles: str, output_path: str, 
                           sub_img_size: tuple = (600, 300)) -> bool:
    """
    Generate a reaction image from mapped reaction SMILES.
    
    Args:
        mapped_rxn_smiles: The mapped reaction SMILES string
        output_path: Path where the image will be saved
        sub_img_size: Size of sub-images for reactants/products
        
    Returns:
        True if successful, False otherwise
    """
    try:
        rxn = rdChemReactions.ReactionFromSmarts(mapped_rxn_smiles, useSmiles=True)
        if rxn is None:
            print(f"  Failed to parse reaction: {mapped_rxn_smiles[:50]}...")
            return False
        
        img = Draw.ReactionToImage(rxn, subImgSize=sub_img_size)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"  Error generating image: {e}")
        return False


def process_mechanism_folder(folder_path: Path) -> tuple:
    """
    Process a single mechanism folder, generating images for each step.
    
    Args:
        folder_path: Path to the mechanism folder
        
    Returns:
        Tuple of (success_count, total_steps)
    """
    mapped_rxn_file = folder_path / "mapped_rxn.json"
    
    if not mapped_rxn_file.exists():
        return (0, 0)
    
    # Read the mapped reaction JSON
    try:
        with open(mapped_rxn_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error reading {mapped_rxn_file}: {e}")
        return (0, 0)
    
    mechanism_rxn = data.get("mechanism_rxn", [])
    if not mechanism_rxn:
        print(f"  No mechanism_rxn found in {folder_path.name}")
        return (0, 0)
    
    # Create output directory
    output_dir = folder_path / "mapped_mechanism_steps"
    output_dir.mkdir(exist_ok=True)
    
    success_count = 0
    total_steps = len(mechanism_rxn)
    
    for step_data in mechanism_rxn:
        step_num = step_data.get("step", "unknown")
        mapped_rxn = step_data.get("mapped_rxn", "")
        
        if not mapped_rxn:
            print(f"  Step {step_num}: No mapped_rxn found, skipping")
            continue
        
        output_path = output_dir / f"mapped_step{step_num}.png"
        
        if generate_reaction_image(mapped_rxn, str(output_path)):
            success_count += 1
        else:
            print(f"  Step {step_num}: Failed to generate image")
    
    return (success_count, total_steps)


def main():
    """Main function to process all mechanism folders."""
    # Get the script directory and mechanisms folder
    script_dir = Path(__file__).parent
    mechanisms_dir = script_dir / "mechanisms"
    
    if not mechanisms_dir.exists():
        print(f"Mechanisms directory not found: {mechanisms_dir}")
        return
    
    # Get all subdirectories
    folders = sorted([f for f in mechanisms_dir.iterdir() if f.is_dir()])
    
    print(f"Found {len(folders)} mechanism folders")
    print("=" * 60)
    
    total_success = 0
    total_steps = 0
    processed_folders = 0
    
    for folder in folders:
        print(f"Processing {folder.name}...")
        success, steps = process_mechanism_folder(folder)
        
        if steps > 0:
            processed_folders += 1
            total_success += success
            total_steps += steps
            print(f"  Generated {success}/{steps} images")
        else:
            print("  Skipped (no mapped_rxn.json or empty)")
    
    print("=" * 60)
    print("Summary:")
    print(f"  Processed folders: {processed_folders}")
    print(f"  Total images generated: {total_success}/{total_steps}")
    print(f"  Success rate: {total_success/total_steps*100:.1f}%" if total_steps > 0 else "  No steps processed")


if __name__ == "__main__":
    main()

