from rdkit.Chem import AllChem, Draw
import json
import os
import ast
import logging
from datetime import datetime


def generate_reaction_images(smiles_json_path, output_dir=None):
    """
    Generate reaction images from SMILES JSON file using RDKit.
    
    Parameters:
    -----------
    smiles_json_path : str
        Path to the smiles.json file containing reaction SMILES strings
    output_dir : str, optional
        Directory to save reaction images. If None, saves in the same directory as smiles.json
    
    Returns:
    --------
    list
        List of paths to the generated reaction image files
    """
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(smiles_json_path)
    
    # Read SMILES JSON file
    with open(smiles_json_path, 'r', encoding='utf-8') as f:
        smiles_data = json.load(f)
    
    # Parse the SMILES list (it's stored as a string representation of a list)
    if isinstance(smiles_data, str):
        try:
            smiles_list = ast.literal_eval(smiles_data)
        except (ValueError, SyntaxError):
            smiles_list = [smiles_data]
    else:
        smiles_list = smiles_data
    
    saved_images = []
    
    # Generate image for each reaction SMILES
    for idx, rxn_smiles in enumerate(smiles_list, start=1):
        try:
            # Parse reaction SMILES
            rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
            
            if rxn is None:
                print(f"  Warning: Could not parse reaction {idx}: {rxn_smiles}")
                continue
            
            # Draw reaction
            img = Draw.ReactionToImage(rxn, subImgSize=(400, 300))
            
            # Save image
            output_path = os.path.join(output_dir, f'reaction{idx}.png')
            img.save(output_path)
            saved_images.append(output_path)
            
            print(f"  ✓ Generated reaction{idx}.png")
            
        except Exception as e:
            print(f"  ✗ Error generating reaction {idx}: {str(e)}")
            continue
    
    return saved_images


def batch_regenerate_reaction_images(base_dir, log_to_file=True):
    """
    Batch regenerate reaction images for all subdirectories containing smiles.json.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing subdirectories with smiles.json files
    log_to_file : bool
        Whether to save log to file
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'total_folders': Total number of folders processed
        - 'successful_folders': Number of successfully processed folders
        - 'failed_folders': Number of failed folders
        - 'total_reactions_generated': Total number of reaction images generated
        - 'details': List of processing details for each folder
    """
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = None
    
    if log_to_file:
        log_filename = os.path.join(base_dir, f'regeneration_log_{timestamp}.txt')
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
    total_folders = 0
    successful_folders = 0
    failed_folders = 0
    total_reactions_generated = 0
    details = []
    
    logger.info("=" * 80)
    logger.info("Starting batch regeneration of reaction images")
    logger.info(f"Base directory: {base_dir}")
    logger.info("=" * 80)
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    subdirs.sort()
    
    logger.info(f"Found {len(subdirs)} subdirectories")
    
    for subdir_name in subdirs:
        subdir_path = os.path.join(base_dir, subdir_name)
        smiles_json_path = os.path.join(subdir_path, 'smiles.json')
        
        total_folders += 1
        
        # Check if smiles.json exists
        if not os.path.exists(smiles_json_path):
            logger.warning(f"[{total_folders}/{len(subdirs)}] ⊗ Skipping {subdir_name} - smiles.json not found")
            failed_folders += 1
            details.append({
                'folder': subdir_name,
                'status': 'failed',
                'reason': 'smiles.json not found',
                'reactions_generated': 0
            })
            continue
        
        try:
            logger.info("=" * 80)
            logger.info(f"[{total_folders}/{len(subdirs)}] Processing {subdir_name}")
            logger.info("=" * 80)
            
            # Remove old reaction images
            old_reaction_images = [f for f in os.listdir(subdir_path) if f.startswith('reaction') and f.endswith('.png')]
            for old_img in old_reaction_images:
                old_img_path = os.path.join(subdir_path, old_img)
                os.remove(old_img_path)
                logger.info(f"  Removed old image: {old_img}")
            
            # Generate new reaction images
            reaction_images = generate_reaction_images(smiles_json_path, subdir_path)
            
            successful_folders += 1
            total_reactions_generated += len(reaction_images)
            
            logger.info(f"✓ Successfully processed {subdir_name}")
            logger.info(f"  - Reactions generated: {len(reaction_images)}")
            
            details.append({
                'folder': subdir_name,
                'status': 'success',
                'reactions_generated': len(reaction_images)
            })
            
        except Exception as e:
            error_msg = f"✗ Error processing {subdir_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"  - Error type: {type(e).__name__}")
            failed_folders += 1
            details.append({
                'folder': subdir_name,
                'status': 'failed',
                'reason': str(e),
                'error_type': type(e).__name__,
                'reactions_generated': 0
            })
            continue
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("BATCH REGENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total folders: {total_folders}")
    logger.info(f"Successful: {successful_folders}")
    logger.info(f"Failed: {failed_folders}")
    logger.info(f"Total reaction images generated: {total_reactions_generated}")
    
    if log_to_file:
        logger.info(f"Log file saved to: {log_filename}")
    
    # Log failed folders
    failed_details = [d for d in details if d['status'] == 'failed']
    if failed_details:
        logger.info("\n" + "-" * 80)
        logger.info(f"FAILED FOLDERS ({len(failed_details)}):")
        logger.info("-" * 80)
        for item in failed_details:
            logger.info(f"  - {item['folder']}")
            if 'reason' in item:
                logger.info(f"    Reason: {item['reason']}")
    
    logger.info("=" * 80)
    
    # Save summary to JSON
    if log_to_file:
        summary_file = os.path.join(base_dir, f'regeneration_summary_{timestamp}.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'total_folders': total_folders,
                'successful_folders': successful_folders,
                'failed_folders': failed_folders,
                'total_reactions_generated': total_reactions_generated,
                'details': details
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to: {summary_file}")
    
    return {
        'total_folders': total_folders,
        'successful_folders': successful_folders,
        'failed_folders': failed_folders,
        'total_reactions_generated': total_reactions_generated,
        'details': details,
        'log_file': log_filename
    }


if __name__ == "__main__":
    # Configuration
    base_dir = r'C:\Users\23163\Desktop\LLM4chem\fukuyamamechanism\RxnIM_Reactions'
    
    # Batch regenerate all reaction images
    result = batch_regenerate_reaction_images(base_dir, log_to_file=True)
    
    print(f"\n{'=' * 80}")
    print("All regeneration complete!")
    print(f"Successful folders: {result['successful_folders']}/{result['total_folders']}")
    print(f"Total reactions generated: {result['total_reactions_generated']}")
    if result['log_file']:
        print(f"Check log file: {result['log_file']}")
    print(f"{'=' * 80}")

