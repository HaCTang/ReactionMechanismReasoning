# Gemini Checkpoint Generator
# This script analyzes mechanism.json files and generates checkpoint files (ckpt.txt)
# for key reaction intermediates based on organic chemistry principles
#
# Checkpoint Selection Criteria:
# 1. Bond Topology Change: Any step involving C-C, C-N, C-O bond formation or cleavage
# 2. Functional Group Transformation: When molecule type changes (e.g., aldehyde -> imine -> nitrile -> amide)
# 3. Collapse Trivial Proton Transfers: Merge consecutive proton gain/loss steps into the same checkpoint
#
# Output format (ckpt.txt):
# Each line represents a checkpoint (any match counts as success)
# checkpoint_id
# checkpoint_id0,checkpoint_id1,checkpoint_id2,...

import os
import json
import re
from pathlib import Path
from google.genai import Client
from google.genai.types import Part, Content

# ============================================
# Configuration
# ============================================
def _load_google_api_key() -> str:
    env = os.environ.get("GOOGLE_API_KEY", "")
    if env:
        return env
    key_file = Path(__file__).resolve().parents[2] / "gemini.key"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    return ""


GOOGLE_API_KEY = _load_google_api_key()

# Model options:
# - "gemini-2.5-pro" (recommended for complex mechanism analysis)
# - "gemini-2.0-flash-exp" (latest experimental model)
# - "gemini-1.5-pro" (recommended for complex tasks)
# - "gemini-1.5-flash" (faster and cheaper)
MODEL_NAME = "gemini-3.1-pro-preview"

# Initialize Gemini client
client = Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None


def load_mechanism_data(mechanism_json_path):
    """
    Load mechanism data from mechanism.json file
    
    Args:
        mechanism_json_path: path to the mechanism.json file
    
    Returns:
        dict with mechanism data
    """
    try:
        with open(mechanism_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading mechanism data: {e}")
        return None


def analyze_checkpoints(mechanism_data, reaction_label):
    """
    Analyze mechanism data and identify key checkpoints using Gemini
    
    Args:
        mechanism_data: dictionary with mechanism information
        reaction_label: label of the reaction (e.g., "A011")
    
    Returns:
        list of checkpoint lines (strings)
    """
    print(f"Analyzing checkpoints for reaction: {reaction_label}")
    
    try:
        # Format mechanism data for the prompt
        mechanism_info = json.dumps(mechanism_data, indent=2, ensure_ascii=False)
        
        # Create the prompt for checkpoint analysis
        prompt = f"""You are an expert in organic chemistry reaction mechanisms.

I will provide you with a detailed mechanism JSON file containing elementary reaction steps.
Your task is to identify KEY CHECKPOINTS - the critical intermediates that represent major transformations.

MECHANISM DATA:
{mechanism_info}

CHECKPOINT SELECTION CRITERIA:

1. BOND TOPOLOGY CHANGE (Most Critical):
   - Any step where C-C, C-N, or C-O bonds are FORMED or BROKEN
   - Examples: nucleophilic addition, elimination, substitution, cyclization, ring-opening
   - This is the PRIMARY criterion for checkpoint selection

2. FUNCTIONAL GROUP TRANSFORMATION:
   - When the molecule converts from one compound class to another
   - Examples: aldehyde -> imine, imine -> nitrile, nitrile -> amide, alcohol -> ether
   - Record the representative structure at each transformation point

3. COLLAPSE TRIVIAL PROTON TRANSFERS:
   - Simple protonation/deprotonation steps that don't change the carbon skeleton
   - Consecutive proton gain/loss steps should be merged into the SAME checkpoint
   - Only the FINAL intermediate after proton equilibration should be the checkpoint
   - Exception: If protonation is essential for activating a leaving group, it should be included

RULES FOR CHECKPOINT GROUPING:
- If multiple mechanism_ids represent the SAME key intermediate (just different protonation states), 
  group them on the SAME line separated by commas
- For example, if step 5, 6, 7 all represent variants of an imine intermediate with different protonation,
  output: "5,6,7" on one line (any of these appearing counts as correct)
- Each line should represent ONE conceptual checkpoint

ANALYSIS PROCESS:
1. Identify all steps with C-C, C-N, C-O bond changes
2. Identify functional group transformations
3. Group consecutive proton transfer steps
4. Determine which mechanism_ids should be checkpoints

OUTPUT FORMAT:
Return ONLY a simple text format with one checkpoint per line:
- Each line contains mechanism_id(s) that represent that checkpoint
- If multiple mechanism_ids are equivalent, separate them with commas (no spaces)
- Order the lines from first checkpoint to last
- Do not include the starting material or final product (only intermediates)

EXAMPLE OUTPUT:
1
5,6,7
8
12

This means:
- Step 1 is the first checkpoint
- Steps 5, 6, or 7 are equivalent representations of the second checkpoint
- Step 8 is the third checkpoint
- Step 12 is the fourth checkpoint

Return ONLY the checkpoint list, nothing else. No explanations, no formatting, just the lines."""

        # Generate response using Gemini
        text_part = Part(text=prompt)
        content = Content(parts=[text_part])
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=content
        )
        
        # Extract text from response
        response_text = response.text.strip()
        
        # Parse the checkpoint lines
        checkpoint_lines = []
        for line in response_text.split('\n'):
            line = line.strip()
            # Validate line format (should be numbers separated by commas)
            if line and re.match(r'^[\d,]+$', line):
                checkpoint_lines.append(line)
        
        if checkpoint_lines:
            print(f"  Found {len(checkpoint_lines)} checkpoints")
            return checkpoint_lines
        else:
            print("  Warning: No valid checkpoints found in response")
            print(f"  Response: {response_text}")
            return None
    
    except Exception as e:
        print(f"Error analyzing checkpoints: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_checkpoints(checkpoint_lines, output_path):
    """
    Save checkpoint data to ckpt.txt file
    
    Args:
        checkpoint_lines: list of checkpoint strings
        output_path: path to save ckpt.txt
    """
    if not checkpoint_lines:
        print("No checkpoint data to save")
        return
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for line in checkpoint_lines:
                f.write(line + "\n")
        print(f"  Saved checkpoints to {output_path}")
    except Exception as e:
        print(f"Error saving checkpoints: {e}")


def process_single_mechanism(mechanism_folder, overwrite=False):
    """
    Process a single mechanism folder and generate ckpt.txt
    
    Args:
        mechanism_folder: path to the mechanism folder (e.g., mechanisms/A011)
        overwrite: whether to overwrite existing ckpt.txt files
    
    Returns:
        True if successful, False otherwise
    """
    mechanism_json_path = os.path.join(mechanism_folder, "mechanism.json")
    ckpt_output_path = os.path.join(mechanism_folder, "ckpt.txt")
    
    # Check if mechanism.json exists
    if not os.path.exists(mechanism_json_path):
        print(f"  Skipping: mechanism.json not found")
        return False
    
    # Check if ckpt.txt already exists and we're not overwriting
    if os.path.exists(ckpt_output_path) and not overwrite:
        print(f"  Skipping: ckpt.txt already exists (use overwrite=True to regenerate)")
        return True
    
    # Get reaction label from folder name
    reaction_label = os.path.basename(mechanism_folder)
    
    # Load mechanism data
    mechanism_data = load_mechanism_data(mechanism_json_path)
    if not mechanism_data:
        print(f"  Failed to load mechanism data")
        return False
    
    # Analyze and identify checkpoints
    checkpoint_lines = analyze_checkpoints(mechanism_data, reaction_label)
    
    if checkpoint_lines:
        save_checkpoints(checkpoint_lines, ckpt_output_path)
        return True
    else:
        print(f"  Failed to generate checkpoints")
        return False


def batch_process_checkpoints(mechanisms_folder, prefix=None, start_num=None, end_num=None, overwrite=False):
    """
    Process mechanism folders in batch and generate ckpt.txt files
    
    Args:
        mechanisms_folder: base folder containing mechanism subfolders
        prefix: reaction prefix like 'A', 'B', 'C' (default: None, process all)
        start_num: starting number (e.g., 1 for A001)
        end_num: ending number (e.g., 78 for A078)
        overwrite: whether to overwrite existing ckpt.txt files
    
    Returns:
        tuple (successful_count, failed_count)
    """
    successful = 0
    failed = 0
    
    if not os.path.exists(mechanisms_folder):
        print(f"Error: Mechanisms folder not found: {mechanisms_folder}")
        return 0, 0
    
    # If prefix and number range are specified, generate the list
    if prefix is not None and start_num is not None and end_num is not None:
        labels_to_process = []
        for i in range(start_num, end_num + 1):
            # Handle special cases like C032_a, C032_b
            label = f"{prefix}{i:03d}"
            labels_to_process.append(label)
            # Also check for _a, _b suffixes
            labels_to_process.append(f"{label}_a")
            labels_to_process.append(f"{label}_b")
        
        print(f"Processing {prefix}{start_num:03d} to {prefix}{end_num:03d}")
        
        for label in labels_to_process:
            mechanism_folder = os.path.join(mechanisms_folder, label)
            if os.path.isdir(mechanism_folder):
                print(f"\nProcessing {label}...")
                if process_single_mechanism(mechanism_folder, overwrite):
                    successful += 1
                else:
                    failed += 1
    else:
        # Process all folders in mechanisms_folder
        mechanism_dirs = sorted([d for d in os.listdir(mechanisms_folder) 
                                 if os.path.isdir(os.path.join(mechanisms_folder, d))])
        
        # Filter by prefix if specified
        if prefix is not None:
            mechanism_dirs = [d for d in mechanism_dirs if d.startswith(prefix)]
        
        total_count = len(mechanism_dirs)
        print(f"Found {total_count} mechanism folders")
        print("=" * 50)
        
        for i, dir_name in enumerate(mechanism_dirs, 1):
            mechanism_folder = os.path.join(mechanisms_folder, dir_name)
            print(f"\n[{i}/{total_count}] Processing {dir_name}...")
            
            if process_single_mechanism(mechanism_folder, overwrite):
                successful += 1
            else:
                failed += 1
    
    print("\n" + "=" * 50)
    print("Batch processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("=" * 50)
    
    return successful, failed


def print_mechanism_summary(mechanism_data):
    """
    Print a summary of the mechanism steps
    
    Args:
        mechanism_data: dictionary with mechanism information
    """
    mechanism = mechanism_data.get('mechanism', [])
    print(f"\nMechanism has {len(mechanism)} steps:")
    
    for step in mechanism:
        step_id = step.get('mechanism_id', '?')
        reactants = step.get('reactants', [])
        products = step.get('products', [])
        
        reactant_info = []
        for r in reactants:
            label = r.get('label', 'N/A')
            smiles = r.get('smiles', 'N/A')[:30]  # Truncate long SMILES
            reactant_info.append(f"{label}")
        
        product_info = []
        for p in products:
            label = p.get('label', 'N/A')
            smiles = p.get('smiles', 'N/A')[:30]
            product_info.append(f"{label}")
        
        print(f"  Step {step_id}: {', '.join(reactant_info)} -> {', '.join(product_info)}")


def compare_checkpoints(mechanisms_folder, verbose=False):
    """
    Compare generated checkpoints with existing ckpt.txt files (if any)
    Useful for validation
    
    Args:
        mechanisms_folder: base folder containing mechanism subfolders
        verbose: whether to print detailed comparison
    
    Returns:
        dict with comparison results
    """
    results = {}
    
    mechanism_dirs = sorted([d for d in os.listdir(mechanisms_folder) 
                             if os.path.isdir(os.path.join(mechanisms_folder, d))])
    
    for dir_name in mechanism_dirs:
        ckpt_path = os.path.join(mechanisms_folder, dir_name, "ckpt.txt")
        
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            results[dir_name] = lines
            
            if verbose:
                print(f"{dir_name}: {lines}")
    
    return results


if __name__ == "__main__":
    # Example 1: Process a single mechanism folder
    # process_single_mechanism("mechanisms/A011", overwrite=True)
    
    # Example 2: Batch process with prefix and number range
    # batch_process_checkpoints(
    #     mechanisms_folder="mechanisms",
    #     prefix="C",
    #     start_num=11,
    #     end_num=109,
    #     overwrite=False
    # )
    
    # Example 3: Batch process all mechanisms with a specific prefix
    batch_process_checkpoints(
        mechanisms_folder="mechanisms",
        prefix="C",
        overwrite=True  # Set to True to regenerate all ckpt.txt files
    )
    
    # Example 4: Batch process all mechanisms (all prefixes)
    # batch_process_checkpoints(
    #     mechanisms_folder="mechanisms",
    #     overwrite=False
    # )
    
    # Example 5: Compare/view existing checkpoints
    # compare_checkpoints("mechanisms", verbose=True)

