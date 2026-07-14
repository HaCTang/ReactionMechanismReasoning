from gradio_client import Client, handle_file
import json
import os
import shutil
from pathlib import Path
import ast
from rdkit.Chem import AllChem, Draw
import logging
from datetime import datetime



def process_reaction_image(input_image_path, output_base_dir):
    """
    Process reaction image using RxnIM API and save results to specified directory.
    
    Parameters:
    -----------
    input_image_path : str
        Path to the input reaction image file
    output_base_dir : str
        Base directory where results will be saved. A subfolder with the image name will be created.
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'output_dir': Path to the output directory
        - 'html': HTML formatted reaction information
        - 'smiles': List of SMILES strings
        - 'files': Dictionary of saved file paths
    """
    # Extract filename without extension
    image_name = Path(input_image_path).stem
    
    # Create output directory
    output_dir = os.path.join(output_base_dir, image_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing image: {input_image_path}")
    print(f"Output directory: {output_dir}")
    
    # Call API
    client = Client("CYF200127/RxnIM")
    result = client.predict(
        image=handle_file(input_image_path),
        selected_task="Reaction Image Parsing Workflow",
        api_name="/process_chem_image"
    )
    
    # Save results
    # result[0]: HTML text
    # result[1]: SMILES list
    # result[2]: combined_output.png path
    # result[3]: exp.png path
    # result[4]: output.json path
    
    saved_files = {}
    
    # Save HTML text
    html_output_path = os.path.join(output_dir, 'reaction_info.html')
    with open(html_output_path, 'w', encoding='utf-8') as f:
        f.write(result[0])
    saved_files['html'] = html_output_path
    
    # Save SMILES list
    smiles_output_path = os.path.join(output_dir, 'smiles.json')
    with open(smiles_output_path, 'w', encoding='utf-8') as f:
        json.dump(result[1], f, indent=2)
    saved_files['smiles'] = smiles_output_path
    
    # Copy image files
    if result[2] and os.path.exists(result[2]):
        combined_output_path = os.path.join(output_dir, 'combined_output.png')
        shutil.copy2(result[2], combined_output_path)
        saved_files['combined_output'] = combined_output_path
        
    if result[3] and os.path.exists(result[3]):
        exp_output_path = os.path.join(output_dir, 'exp.png')
        shutil.copy2(result[3], exp_output_path)
        saved_files['exp'] = exp_output_path
    
    # Copy JSON file
    if result[4] and os.path.exists(result[4]):
        json_output_path = os.path.join(output_dir, 'output.json')
        shutil.copy2(result[4], json_output_path)
        saved_files['output_json'] = json_output_path
    
    # Save complete result
    complete_result_path = os.path.join(output_dir, 'complete_result.json')
    with open(complete_result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'html': result[0],
            'smiles': result[1],
            'combined_output_path': result[2],
            'exp_path': result[3],
            'output_json_path': result[4]
        }, f, indent=2, ensure_ascii=False)
    saved_files['complete_result'] = complete_result_path
    
    print(f"\nResults saved successfully!")
    print(f"Files created:")
    for file_type, file_path in saved_files.items():
        print(f"  - {Path(file_path).name}")
    
    return {
        'output_dir': output_dir,
        'html': result[0],
        'smiles': result[1],
        'files': saved_files
    }


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
        except:
            smiles_list = [smiles_data]
    else:
        smiles_list = smiles_data
    
    saved_images = []
    
    print(f"\nGenerating reaction images from: {smiles_json_path}")
    
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
    
    print(f"\nSuccessfully generated {len(saved_images)} reaction images")
    return saved_images


def batch_process_with_logging(start_prefix, start_num, end_num, base_image_dir, output_base_dir):
    """
    Batch process reaction images with comprehensive logging.
    
    Parameters:
    -----------
    start_prefix : str
        Prefix for image files (e.g., 'A', 'B', 'C')
    start_num : int
        Starting number (e.g., 1 for A001q)
    end_num : int
        Ending number (e.g., 78 for A078q)
    base_image_dir : str
        Directory containing input images
    output_base_dir : str
        Base directory for output files
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'total_processed': Number of successfully processed images
        - 'total_failed': Number of failed images
        - 'total_reactions_generated': Total number of reaction images generated
        - 'failed_files': List of failed files with error details
        - 'missing_files': List of missing files
        - 'log_file': Path to log file
        - 'error_summary_file': Path to error summary file (if errors exist)
    """
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(output_base_dir, f'processing_log_{start_prefix}{start_num:03d}-{start_prefix}{end_num:03d}_{timestamp}.txt')
    
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True  # Force reconfiguration
    )
    
    logger = logging.getLogger(__name__)
    
    # Statistics
    total_processed = 0
    total_failed = 0
    total_reactions_generated = 0
    failed_files = []
    missing_files = []
    
    total_count = end_num - start_num + 1
    
    logger.info("=" * 80)
    logger.info(f"Starting batch processing: {start_prefix}{start_num:03d}q to {start_prefix}{end_num:03d}q")
    logger.info(f"Total files to process: {total_count}")
    logger.info("=" * 80)
    
    for i in range(start_num, end_num + 1):
        # Format image name with leading zeros
        image_name = f"{start_prefix}{i:03d}q.png"
        input_image_path = os.path.join(base_image_dir, image_name)
        
        # Check if file exists
        if not os.path.exists(input_image_path):
            logger.warning(f"[{i-start_num+1}/{total_count}] ⊗ Skipping {image_name} - file not found")
            missing_files.append(image_name)
            total_failed += 1
            continue
        
        try:
            logger.info("=" * 80)
            logger.info(f"[{i-start_num+1}/{total_count}] Processing {image_name}")
            logger.info("=" * 80)
            
            # Process reaction image
            result = process_reaction_image(input_image_path, output_base_dir)
            
            # Generate reaction images from SMILES
            smiles_json_path = result['files']['smiles']
            reaction_images = generate_reaction_images(smiles_json_path)
            
            total_processed += 1
            total_reactions_generated += len(reaction_images)
            
            logger.info(f"✓ Successfully processed {image_name}")
            logger.info(f"  - Output: {result['output_dir']}")
            logger.info(f"  - Reactions generated: {len(reaction_images)}")
            
        except Exception as e:
            error_msg = f"✗ Error processing {image_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"  - Error type: {type(e).__name__}")
            failed_files.append({
                'file': image_name,
                'error': str(e),
                'error_type': type(e).__name__
            })
            total_failed += 1
            continue
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total images processed: {total_processed}/{total_count}")
    logger.info(f"Total failures: {total_failed}")
    logger.info(f"Total reaction images generated: {total_reactions_generated}")
    logger.info(f"Results saved to: {output_base_dir}")
    logger.info(f"Log file saved to: {log_filename}")
    
    # Log missing files
    if missing_files:
        logger.info("\n" + "-" * 80)
        logger.info(f"MISSING FILES ({len(missing_files)}):")
        logger.info("-" * 80)
        for file in missing_files:
            logger.info(f"  - {file}")
    
    # Log failed files with error details
    if failed_files:
        logger.info("\n" + "-" * 80)
        logger.info(f"FAILED FILES ({len(failed_files)}):")
        logger.info("-" * 80)
        for item in failed_files:
            logger.info(f"  - {item['file']}")
            logger.info(f"    Error Type: {item['error_type']}")
            logger.info(f"    Error Message: {item['error']}")
    
    logger.info("=" * 80)
    
    # Save error summary to separate file
    error_summary_file = None
    if failed_files or missing_files:
        error_summary_file = os.path.join(output_base_dir, f'error_summary_{start_prefix}{start_num:03d}-{start_prefix}{end_num:03d}_{timestamp}.json')
        with open(error_summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'range': f'{start_prefix}{start_num:03d}q to {start_prefix}{end_num:03d}q',
                'total_count': total_count,
                'total_processed': total_processed,
                'total_failed': total_failed,
                'missing_files': missing_files,
                'failed_files': failed_files
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Error summary saved to: {error_summary_file}")
    
    return {
        'total_processed': total_processed,
        'total_failed': total_failed,
        'total_reactions_generated': total_reactions_generated,
        'failed_files': failed_files,
        'missing_files': missing_files,
        'log_file': log_filename,
        'error_summary_file': error_summary_file
    }


if __name__ == "__main__":
    # Configuration
    base_image_dir = r'C:\Users\23163\Desktop\LLM4chem\fukuyamamechanism\images'
    output_dir = r'C:\Users\23163\Desktop\LLM4chem\fukuyamamechanism\RxnIM_Reactions_test'
    
    # Batch process B001q to B128q
    result = batch_process_with_logging(
        start_prefix='C',
        start_num=16,
        end_num=16,
        base_image_dir=base_image_dir,
        output_base_dir=output_dir
    )
    
    print(f"\n{'=' * 80}")
    print("All processing complete!")
    print(f"Check log file: {result['log_file']}")
    if result['error_summary_file']:
        print(f"Check error summary: {result['error_summary_file']}")
    print(f"{'=' * 80}")

