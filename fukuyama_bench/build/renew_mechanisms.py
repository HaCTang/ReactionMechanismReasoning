import os
import re
import json
import logging
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors


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


# ============================================================================
# Electron Number and Charge Calculation
# ============================================================================

# Metal atoms that RDKit may not handle correctly for valence electron calculation
# These are transition metals and heavy metals commonly found in organometallic compounds
METAL_SYMBOLS = {
    # # Transition metals
    # 'Sc', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
    # 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ag', 'Cd',
    # 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    # # 'Pd', 'Sm', 'Ru', 'Ti', 'Zn', 'Sn', 'Rh'
    # # Lanthanides
    # 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    # # Actinides
    # 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
    # # Main group metals (that may cause issues)
    # 'Pb', 'Bi', 'Sb', 'Te', 'Po',
    # # Alkali and alkaline earth metals
    # 'Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba',
    # # Other metals
    # 'Al', 'Ga', 'In', 'Tl', 'Ge'
}


def contains_metal(smiles):
    """
    Check if a SMILES string contains metal atoms.
    
    Args:
        smiles: SMILES string
        
    Returns:
        bool: True if contains metal, False otherwise
    """
    for metal in METAL_SYMBOLS:
        # Check for metal in brackets like [Hg], [Pd], [Fe+2], etc.
        if f'[{metal}]' in smiles or f'[{metal}+' in smiles or f'[{metal}-' in smiles:
            return True
        # Check for metal with other patterns like [Hg+2], [Fe2+], etc.
        if f'[{metal}' in smiles:
            return True
    return False


def calculate_charge(smiles):
    """
    Calculate formal charge from SMILES using RDKit.
    This works reliably for all molecules including metal complexes.
    
    Args:
        smiles: SMILES string of the molecule
        
    Returns:
        int: Formal charge, or None if failed
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.GetFormalCharge(mol)
    except Exception:
        return None


def calculate_electron_number(smiles):
    """
    Calculate valence electron number from SMILES using RDKit.
    
    For molecules containing metal atoms, returns None to skip update
    because RDKit's NumValenceElectrons is unreliable for organometallic compounds.
    
    Args:
        smiles: SMILES string of the molecule
        
    Returns:
        int: Valence electron number, or None if failed or contains metal
    """
    # Skip molecules containing metals - RDKit's valence electron calculation is unreliable
    if contains_metal(smiles):
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Descriptors.NumValenceElectrons(mol)
    except Exception:
        return None


def update_molecule_in_list(molecule_list, updated_count):
    """
    Update electron_number and charge for all molecules in a list.
    
    - Charge is always updated (RDKit is reliable for charge calculation)
    - Electron number is only updated for non-metal molecules
    """
    for molecule in molecule_list:
        smiles = molecule.get("smiles")
        if not smiles:
            continue
        
        updated = False
        
        # Always try to update charge (works for all molecules)
        new_charge = calculate_charge(smiles)
        if new_charge is not None:
            old_charge = molecule.get("charge")
            if old_charge != new_charge:
                molecule["charge"] = new_charge
                updated = True
        
        # Only update electron number for non-metal molecules
        new_electron = calculate_electron_number(smiles)
        if new_electron is not None:
            old_electron = molecule.get("electron_number")
            if old_electron != new_electron:
                molecule["electron_number"] = new_electron
                updated = True
        
        if updated:
            updated_count[0] += 1


def update_mechanism_electron_charge(mechanism_file, logger):
    """
    Update electron_number and charge in mechanism.json file.
    
    Returns:
        int: Number of molecules updated
    """
    try:
        with open(mechanism_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"    Error reading mechanism.json: {e}")
        return 0
    
    updated_count = [0]
    
    # Process mechanism section
    if "mechanism" in data:
        for step in data["mechanism"]:
            if "reactants" in step:
                update_molecule_in_list(step["reactants"], updated_count)     
            if "products" in step:
                update_molecule_in_list(step["products"], updated_count)
    
    # Process reactions section
    if "reactions" in data:
        for reaction in data["reactions"]:
            if "reactants" in reaction:
                update_molecule_in_list(reaction["reactants"], updated_count)
            if "products" in reaction:
                update_molecule_in_list(reaction["products"], updated_count)
    
    if updated_count[0] > 0:
        with open(mechanism_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"    Updated {updated_count[0]} molecule(s) electron/charge")
    
    return updated_count[0]


# ============================================================================
# Reaction Image Generation
# ============================================================================

def generate_mechanism_images(rxn_json_path, output_dir, logger):
    """
    Generate mechanism step images from rxn.json file using RDKit.
    
    Args:
        rxn_json_path: Path to rxn.json file
        output_dir: Directory to save images
        logger: Logger instance
        
    Returns:
        int: Number of images generated
    """
    try:
        with open(rxn_json_path, 'r', encoding='utf-8') as f:
            rxn_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"    Error reading rxn.json: {e}")
        return 0
    
    mechanism_rxn = rxn_data.get("mechanism_rxn", [])
    if not mechanism_rxn:
        logger.warning("    No mechanism_rxn found in rxn.json")
        return 0
    
    images_generated = 0
    
    for rxn_step in mechanism_rxn:
        step_num = rxn_step.get("step", "")
        rxn_smiles = rxn_step.get("rxn_smiles", "")
        
        if not rxn_smiles:
            logger.warning(f"    Step {step_num}: No rxn_smiles found")
            continue
        
        try:
            # Parse reaction SMILES
            rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
            
            if rxn is None:
                logger.warning(f"    Step {step_num}: Could not parse reaction SMILES")
                continue
            
            # Draw reaction
            img = Draw.ReactionToImage(rxn, subImgSize=(400, 300))
            
            # Save image
            output_path = os.path.join(output_dir, f'mechanism_step{step_num}.png')
            img.save(output_path)
            images_generated += 1
            
        except Exception as e:
            logger.error(f"    Step {step_num}: Error generating image - {str(e)}")
            continue
    
    if images_generated > 0:
        logger.info(f"    Generated {images_generated} mechanism step image(s)")
    
    return images_generated


def remove_old_mechanism_images(subdir_path, logger):
    """Remove old mechanism_step*.png images from directory."""
    removed_count = 0
    for filename in os.listdir(subdir_path):
        if filename.startswith('mechanism_step') and filename.endswith('.png'):
            filepath = os.path.join(subdir_path, filename)
            try:
                os.remove(filepath)
                removed_count += 1
            except Exception as e:
                logger.warning(f"    Could not remove {filename}: {e}")
    
    if removed_count > 0:
        logger.info(f"    Removed {removed_count} old mechanism image(s)")
    
    return removed_count


# ============================================================================
# Batch Processing
# ============================================================================

def batch_process_mechanisms(base_dir, log_to_file=True):
    """
    Batch process all mechanism folders:
    1. Update electron_number and charge in mechanism.json
    2. Regenerate mechanism step images from rxn.json
    
    Args:
        base_dir: Base directory containing mechanism subdirectories
        log_to_file: Whether to save log to file
        
    Returns:
        dict: Processing statistics
    """
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = None
    
    if log_to_file:
        log_filename = os.path.join(base_dir, f'renew_ele_log_{timestamp}.txt')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()],
            force=True
        )
    
    logger = logging.getLogger(__name__)
    
    # Statistics
    stats = {
        'total_folders': 0,
        'successful_folders': 0,
        'failed_folders': 0,
        'total_molecules_updated': 0,
        'total_images_generated': 0,
        'details': []
    }
    
    logger.info("=" * 80)
    logger.info("Batch Processing: Update Electron/Charge & Regenerate Images")
    logger.info(f"Base directory: {base_dir}")
    logger.info("=" * 80)
    
    # Get all subdirectories (supports names like A001, B027_a, C032_b)
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    subdirs.sort(key=natural_sort_key)
    
    logger.info(f"Found {len(subdirs)} subdirectories")
    
    for subdir_name in subdirs:
        subdir_path = os.path.join(base_dir, subdir_name)
        mechanism_file = os.path.join(subdir_path, 'mechanism.json')
        rxn_file = os.path.join(subdir_path, 'rxn.json')
        
        stats['total_folders'] += 1
        folder_detail = {
            'folder': subdir_name,
            'status': 'success',
            'molecules_updated': 0,
            'images_generated': 0
        }
        
        logger.info("-" * 80)
        logger.info(f"[{stats['total_folders']}/{len(subdirs)}] Processing {subdir_name}")
        
        try:
            # Check if required files exist
            if not os.path.exists(mechanism_file):
                logger.warning("    mechanism.json not found, skipping electron/charge update")
            else:
                # Update electron_number and charge
                molecules_updated = update_mechanism_electron_charge(mechanism_file, logger)
                folder_detail['molecules_updated'] = molecules_updated
                stats['total_molecules_updated'] += molecules_updated
            
            if not os.path.exists(rxn_file):
                logger.warning("    rxn.json not found, skipping image generation")
            else:
                # Remove old images
                remove_old_mechanism_images(subdir_path, logger)
                
                # Generate new images
                images_generated = generate_mechanism_images(rxn_file, subdir_path, logger)
                folder_detail['images_generated'] = images_generated
                stats['total_images_generated'] += images_generated
            
            stats['successful_folders'] += 1
            logger.info("    Completed successfully")
            
        except Exception as e:
            folder_detail['status'] = 'failed'
            folder_detail['reason'] = str(e)
            stats['failed_folders'] += 1
            logger.error(f"    Error: {str(e)}")
        
        stats['details'].append(folder_detail)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total folders processed: {stats['total_folders']}")
    logger.info(f"Successful: {stats['successful_folders']}")
    logger.info(f"Failed: {stats['failed_folders']}")
    logger.info(f"Total molecules updated: {stats['total_molecules_updated']}")
    logger.info(f"Total images generated: {stats['total_images_generated']}")
    
    if log_to_file:
        logger.info(f"Log file saved to: {log_filename}")
    
    # Log failed folders
    failed_details = [d for d in stats['details'] if d['status'] == 'failed']
    if failed_details:
        logger.info("\n" + "-" * 80)
        logger.info(f"FAILED FOLDERS ({len(failed_details)}):")
        logger.info("-" * 80)
        for item in failed_details:
            logger.info(f"  - {item['folder']}: {item.get('reason', 'Unknown error')}")
    
    logger.info("=" * 80)
    
    # Save summary to JSON
    if log_to_file:
        summary_file = os.path.join(base_dir, f'renew_ele_summary_{timestamp}.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                **stats
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to: {summary_file}")
    
    stats['log_file'] = log_filename
    return stats


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Get the mechanisms folder path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mechanisms_dir = os.path.join(script_dir, "mechanisms")
    
    if not os.path.exists(mechanisms_dir):
        print(f"Error: mechanisms folder not found at {mechanisms_dir}")
        exit(1)
    
    # Run batch processing
    result = batch_process_mechanisms(mechanisms_dir, log_to_file=True)
    
    print(f"\n{'=' * 80}")
    print("All processing complete!")
    print(f"Successful folders: {result['successful_folders']}/{result['total_folders']}")
    print(f"Molecules updated: {result['total_molecules_updated']}")
    print(f"Images generated: {result['total_images_generated']}")
    if result['log_file']:
        print(f"Check log file: {result['log_file']}")
    print(f"{'=' * 80}")
