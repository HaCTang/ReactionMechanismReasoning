# Gemini Mechanism Analysis Tool
# This script analyzes reaction mechanism images and generates detailed mechanism JSON files
# Input: Reaction description (output.json) + Mechanism image (e.g., A001a.png)
# Output: Detailed mechanism JSON file (mechanism.json)

import os
import json
import re
import base64
from pathlib import Path
from google.genai import Client
from google.genai.types import Part, Content
from rdkit.Chem import AllChem, Draw

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
MODEL_NAME = "gemini-3-pro-preview"

# Initialize Gemini client
client = Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None


def extract_label_from_filename(image_path):
    """
    Extract reaction label from filename
    e.g., "A001a.png" -> "A001", "B001a.png" -> "B001"
    """
    filename = Path(image_path).stem  # Get filename without extension
    # Try to match pattern like A007, B001, C094, etc.
    match = re.match(r'^([A-Z]\d{3})', filename)
    if match:
        return match.group(1)
    # If no match, use the whole filename
    return filename


def load_reaction_data(json_path):
    """
    Load reaction data from output.json file
    
    Args:
        json_path: path to the output.json file
    
    Returns:
        dict with reaction data
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading reaction data: {e}")
        return None


def analyze_mechanism_image(image_path, reaction_data):
    """
    Analyze a mechanism image and extract detailed mechanism data using Gemini
    
    Args:
        image_path: path to the mechanism image
        reaction_data: dictionary with reaction information from output.json
    
    Returns:
        dict with mechanism data
    """
    label = extract_label_from_filename(image_path)
    
    print(f"Analyzing mechanism for reaction: {label}")
    
    try:
        # Read and encode the image
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Base64 encode the image
        encoded_image = base64.b64encode(image_data).decode("utf-8")
        
        # Determine MIME type based on file extension
        file_ext = Path(image_path).suffix.lower()
        mime_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_type_map.get(file_ext, 'image/png')
        
        # Format reaction data for the prompt
        reaction_info = json.dumps(reaction_data, indent=2, ensure_ascii=False)
        
        # Create the prompt
        prompt = f"""You are an expert in organic chemistry reaction mechanisms.

I will provide you with:
1. A reaction description in JSON format
2. A mechanism image showing the step-by-step reaction mechanism

REACTION DATA:
{reaction_info}

TASK: Analyze the mechanism image and generate a detailed mechanism JSON with each elementary step.

CRITICAL RULES:
1. CHARGE CONSERVATION: The total charge of reactants MUST equal the total charge of products in EACH mechanism step.
   - Count charges carefully: [OH-] has charge -1, [H+] has charge +1, neutral molecules have charge 0
   - For each step, verify: sum(reactant charges) = sum(product charges)

2. LABEL CLASSIFICATION:
   - "substrate": The main organic compound at the start of the reaction
   - "reagent": External reagents that participate in the reaction (like [OH-], [H+], etc.)
   - "intermediate": Products of one step that become reactants in the next step
   - "leaving": Molecules that leave the intermediate products (like [CH3O-], H2O, etc.)
   - "product": The final product of the entire mechanism

3. INTERMEDIATE INHERITANCE:
   - The intermediate product of step N becomes the intermediate reactant of step N+1
   - Track the flow of the main organic molecule through all steps
   - The SMILES of an intermediate product must exactly match the intermediate reactant in the next step

4. ELECTRON NUMBER: Calculate total valence electrons for each molecule
   - H: 1, C: 4, N: 5, O: 6, F: 7, Cl: 7, Br: 7, I: 7, S: 6, P: 5
   - Add electrons for negative charges, subtract for positive charges

5. SMILES ACCURACY:
   - Convert all structures to accurate SMILES format
   - Include stereochemistry where applicable
   - Verify that SMILES correctly represents the structure in the image
   - Pay attention to charged species: use [OH-], [H+], [O-], etc.
   - IMPORTANT: For protonated species, put the H inside the bracket with the charge:
     * CORRECT: [OH+] (protonated oxygen/oxonium)
     * WRONG: [O+](H), [O+]H or [O+H]
     * CORRECT: [OH2] (water)
     * WRONG: [H2O]     
     * CORRECT: [NH4+] (ammonium)
     * WRONG: [N+](H)(H)(H)H
     * CORRECT: [OH2+] (hydronium-like)
     * WRONG: [O+](H)H
     * CORRECT: [nH+]1ccccc1 (pyridine-like)
     * WRONG: [nH]1ccccc1+
     * CORRECT: [NH3] (ammonia)
     * WRONG: NH3     
     * CORRECT: [H2O] (water)
     * WRONG: H2O
   - IMPORTANT: For Phosphorus compounds, use the correct SMILES format:
     * CORRECT: ClP(Cl)Cl
     * WRONG: PCl3
   - IMPORTANT: For hydride ions like [H-], use the correct SMILES format:
     * CORRECT: [HgH]
     * WRONG: [Hg]H
   - IMPORTANT: For metal-monoolefin ternary cyclic complex compounds, use the correct SMILES format:
     * CORRECT: C1=C[Hg+]1
     * WRONG: C1=[C+]1[Hg]
   - IMPORTANT: For carbanions ([CH-], [C-]), write them INLINE in the chain, NOT as a branch:
     * CORRECT: COC(=O)C[CH-]C(=O)OC (carbanion is part of the main chain)
     * WRONG: COC(=O)CC([CH-])C(=O)OC (carbanion as branch adds extra carbon!)
     * CORRECT: CC[CH-]CC (carbanion inline)
     * WRONG: CCC([CH-])C (carbanion as branch - this adds an extra carbon atom)
     * The [CH-] or [C-] must replace a carbon in the chain, not be attached as a substituent
   - IMPORTANT: For alpha-carbonyl anions, prefer ENOLATE form over ketone carbanion form:
     * CORRECT: CC(=C[O-])OC (enolate anion - negative charge on oxygen)
     * WRONG: CC(=O)[CH-]OC (carbanion form - negative charge on carbon)
     * CORRECT: C/C([O-])=C/C (enolate with defined stereochemistry)
     * WRONG: CC(=O)[CH-]C (ketone carbanion)
     * Enolates are more stable due to resonance delocalization of the negative charge
     * When deprotonation occurs alpha to a carbonyl, draw the enolate tautomer

6. REACTION_STEP NUMBERING: Assign step numbers based on reagent addition batches
   - Conditions added in the SAME batch (same time/same arrow in the mechanism) get the SAME step number
   - The conditions in the input JSON are already ordered from first to last batch
   - Look at the mechanism image to identify separate addition steps (usually separated by arrows)
   - Example: If "H2SO4", "EtOH", "reflux" are added together first → all get reaction_step="1"
   - Example: If "NaOH", "heat" are added in a second step → both get reaction_step="2"
   - The step numbers should be "1", "2", "3"... as strings, corresponding to each batch of additions

OUTPUT FORMAT - Return a JSON object with this exact structure:
{{
    "reactions": [
        {{
            "reaction_id": "1",
            "reactants": [
                {{
                    "smiles": "REACTANT_SMILES",
                    "label": "None"
                }}
            ],
            "conditions": [
                {{
                    "role": "reagent|solvent|temperature|catalyst",
                    "text": "condition_text",
                    "reaction_step": "step_number"
                }}
            ],
            "products": [
                {{
                    "smiles": "PRODUCT_SMILES",
                    "label": "None"
                }}
            ]
        }}
    ],
    "mechanism": [
        {{
            "mechanism_id": "1",
            "reactants": [
                {{
                    "smiles": "SMILES",
                    "label": "substrate|reagent|intermediate",
                    "electron_number": NUMBER,
                    "charge": NUMBER
                }}
            ],
            "products": [
                {{
                    "smiles": "SMILES",
                    "label": "intermediate|leaving|product",
                    "electron_number": NUMBER,
                    "charge": NUMBER
                }}
            ]
        }},
        {{
            "mechanism_id": "2",
            "reactants": [
                {{
                    "smiles": "INTERMEDIATE_FROM_STEP_1",
                    "label": "intermediate",
                    "electron_number": NUMBER,
                    "charge": NUMBER
                }}
            ],
            "products": [
                {{
                    "smiles": "SMILES",
                    "label": "intermediate|leaving|product",
                    "electron_number": NUMBER,
                    "charge": NUMBER
                }}
            ]
        }}
    ]
}}

VALIDATION CHECKLIST (verify before returning):
1. [ ] Each mechanism step has balanced charges (sum of reactant charges = sum of product charges)
2. [ ] Intermediate products correctly become intermediate reactants in the next step
3. [ ] All SMILES are valid and represent the correct structures
4. [ ] Electron numbers are correctly calculated
5. [ ] Labels are correctly assigned (substrate, reagent, intermediate, leaving, product)
6. [ ] The first mechanism step starts with the substrate
7. [ ] The last mechanism step produces the final product
8. [ ] Conditions are properly assigned to reaction steps

Return ONLY the JSON object, nothing else."""

        # Generate response using the correct API format with proper types
        text_part = Part(text=prompt)
        image_part = Part(inline_data={'mime_type': mime_type, 'data': encoded_image})
        
        # Create content with both parts
        content = Content(parts=[text_part, image_part])
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=content
        )
        
        # Extract text from response
        response_text = response.text
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            mechanism_data = json.loads(json_match.group())
            mechanism_data['label'] = label
            
            # Validate charge conservation
            validate_charge_conservation(mechanism_data)
            
            return mechanism_data
        else:
            print("Warning: Could not find JSON in response")
            print(f"Response: {response_text}")
            return None
    
    except Exception as e:
        print(f"Error analyzing mechanism image: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_charge_conservation(mechanism_data):
    """
    Validate that charge is conserved in each mechanism step
    
    Args:
        mechanism_data: dictionary with mechanism information
    """
    mechanism = mechanism_data.get('mechanism', [])
    
    for step in mechanism:
        step_id = step.get('mechanism_id', '?')
        
        # Sum reactant charges
        reactant_charges = sum(r.get('charge', 0) for r in step.get('reactants', []))
        
        # Sum product charges
        product_charges = sum(p.get('charge', 0) for p in step.get('products', []))
        
        if reactant_charges != product_charges:
            print(f"WARNING: Charge imbalance in step {step_id}!")
            print(f"  Reactant total charge: {reactant_charges}")
            print(f"  Product total charge: {product_charges}")
        else:
            print(f"Step {step_id}: Charge balanced (total: {reactant_charges})")


def save_mechanism_data(mechanism_data, output_folder="mechanisms"):
    """
    Save mechanism data to JSON file
    
    Args:
        mechanism_data: dictionary with mechanism information
        output_folder: base folder for saving files
    """
    if not mechanism_data:
        print("No mechanism data to save")
        return
    
    label = mechanism_data.get('label', 'UNKNOWN')
    mechanism_folder = os.path.join(output_folder, label)
    
    if not os.path.exists(mechanism_folder):
        os.makedirs(mechanism_folder)
    
    # Remove the 'label' field before saving (it was added for tracking)
    save_data = {k: v for k, v in mechanism_data.items() if k != 'label'}
    
    # Save as JSON file
    json_output_path = os.path.join(mechanism_folder, "mechanism.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)
    
    print(f"Saved mechanism data to {json_output_path}")
    
    # Also print a summary
    print_mechanism_summary(mechanism_data)


def print_mechanism_summary(mechanism_data):
    """
    Print a summary of the mechanism
    
    Args:
        mechanism_data: dictionary with mechanism information
    """
    print("\n" + "=" * 60)
    print("MECHANISM SUMMARY")
    print("=" * 60)
    
    mechanism = mechanism_data.get('mechanism', [])
    print(f"Total mechanism steps: {len(mechanism)}")
    print()
    
    for step in mechanism:
        step_id = step.get('mechanism_id', '?')
        reactants = step.get('reactants', [])
        products = step.get('products', [])
        
        print(f"--- Step {step_id} ---")
        
        # Reactants
        reactant_smiles = []
        for r in reactants:
            smiles = r.get('smiles', 'N/A')
            label = r.get('label', 'N/A')
            charge = r.get('charge', 0)
            reactant_smiles.append(f"{smiles} ({label}, charge={charge})")
        print(f"Reactants: {' + '.join(reactant_smiles)}")
        
        # Products
        product_smiles = []
        for p in products:
            smiles = p.get('smiles', 'N/A')
            label = p.get('label', 'N/A')
            charge = p.get('charge', 0)
            product_smiles.append(f"{smiles} ({label}, charge={charge})")
        print(f"Products: {' + '.join(product_smiles)}")
        print()
    
    print("=" * 60)


def process_mechanism(image_path, reaction_json_path, output_folder="mechanisms"):
    """
    Process a mechanism image with its reaction data
    
    Args:
        image_path: path to the mechanism image (e.g., A001a.png)
        reaction_json_path: path to the reaction output.json file
        output_folder: folder to save results
    
    Returns:
        mechanism data dictionary
    """
    if not os.path.exists(image_path):
        print(f"Error: Mechanism image not found: {image_path}")
        return None
    
    if not os.path.exists(reaction_json_path):
        print(f"Error: Reaction JSON not found: {reaction_json_path}")
        return None
    
    print("Processing mechanism:")
    print(f"  Image: {image_path}")
    print(f"  Reaction data: {reaction_json_path}")
    print("-" * 50)
    
    # Load reaction data
    reaction_data = load_reaction_data(reaction_json_path)
    if not reaction_data:
        print("Failed to load reaction data")
        return None
    
    # Analyze the mechanism image
    mechanism_data = analyze_mechanism_image(image_path, reaction_data)
    
    if mechanism_data:
        # Save the results
        save_mechanism_data(mechanism_data, output_folder)
        print("-" * 50)
        print("Processing complete!")
        return mechanism_data
    else:
        print("Failed to extract mechanism data")
        return None


def batch_process_mechanisms(reactions_folder, images_folder, output_folder="mechanisms", 
                             image_suffix="a", prefix=None, start_num=None, end_num=None,
                             generate_images=True):
    """
    Process mechanism images in batch with complete workflow
    
    Args:
        reactions_folder: folder containing reaction subfolders with output.json files
        images_folder: folder containing mechanism images
        output_folder: folder to save results
        image_suffix: suffix for mechanism images (default: 'a', e.g., A001a.png)
        prefix: reaction prefix like 'A', 'B', 'C' (default: None, process all)
        start_num: starting number (e.g., 1 for A001)
        end_num: ending number (e.g., 78 for A078)
        generate_images: whether to generate rxn.json and mechanism step images (default: True)
    
    Returns:
        list of result dictionaries containing mechanism_data, rxn_data, and images
    """
    all_results = []
    total_images_generated = 0
    
    # Find all reaction folders
    if not os.path.exists(reactions_folder):
        print(f"Error: Reactions folder not found: {reactions_folder}")
        return []
    
    # If prefix and number range are specified, generate the list
    if prefix is not None and start_num is not None and end_num is not None:
        # Generate specific labels to process
        labels_to_process = []
        for i in range(start_num, end_num + 1):
            label = f"{prefix}{i:03d}"
            labels_to_process.append(label)
        
        total_count = len(labels_to_process)
        print(f"Processing {prefix}{start_num:03d} to {prefix}{end_num:03d}")
        print(f"Total reactions to process: {total_count}")
        print("=" * 50)
        
        for i, label in enumerate(labels_to_process, 1):
            # Try both with and without 'q' suffix for reaction folder
            reaction_dir_q = f"{label}q"
            reaction_dir = label
            
            # Check which folder exists
            if os.path.isdir(os.path.join(reactions_folder, reaction_dir_q)):
                reaction_json_path = os.path.join(reactions_folder, reaction_dir_q, "output.json")
            elif os.path.isdir(os.path.join(reactions_folder, reaction_dir)):
                reaction_json_path = os.path.join(reactions_folder, reaction_dir, "output.json")
            else:
                print(f"\n[{i}/{total_count}] Skipping {label}: folder not found")
                continue
            
            image_path = os.path.join(images_folder, f"{label}{image_suffix}.png")
            
            print(f"\n[{i}/{total_count}] Processing {label}")
            
            if not os.path.exists(reaction_json_path):
                print("  Skipping: output.json not found")
                continue
            
            if not os.path.exists(image_path):
                print(f"  Skipping: mechanism image not found ({image_path})")
                continue
            
            try:
                if generate_images:
                    # Complete workflow with rxn.json and images
                    result = process_mechanism_complete(image_path, reaction_json_path, output_folder)
                    if result:
                        all_results.append(result)
                        total_images_generated += len(result.get('images', []))
                else:
                    # Only process mechanism without generating images
                    mechanism_data = process_mechanism(image_path, reaction_json_path, output_folder)
                    if mechanism_data:
                        all_results.append({'mechanism_data': mechanism_data, 'rxn_data': None, 'images': []})
            except Exception as e:
                print(f"Error processing {label}: {e}")
                continue
    else:
        # Process all folders in reactions_folder
        reaction_dirs = [d for d in os.listdir(reactions_folder) 
                         if os.path.isdir(os.path.join(reactions_folder, d))]
        
        # Filter by prefix if specified
        if prefix is not None:
            reaction_dirs = [d for d in reaction_dirs if d.startswith(prefix)]
        
        total_count = len(reaction_dirs)
        print(f"Found {total_count} reaction folders")
        print("=" * 50)
        
        for i, reaction_dir in enumerate(reaction_dirs, 1):
            label = reaction_dir.rstrip('q')  # Remove 'q' suffix if present (e.g., A001q -> A001)
            
            # Construct paths
            reaction_json_path = os.path.join(reactions_folder, reaction_dir, "output.json")
            image_path = os.path.join(images_folder, f"{label}{image_suffix}.png")
            
            print(f"\n[{i}/{total_count}] Processing {label}")
            
            if not os.path.exists(reaction_json_path):
                print("  Skipping: output.json not found")
                continue
            
            if not os.path.exists(image_path):
                print(f"  Skipping: mechanism image not found ({image_path})")
                continue
            
            try:
                if generate_images:
                    # Complete workflow with rxn.json and images
                    result = process_mechanism_complete(image_path, reaction_json_path, output_folder)
                    if result:
                        all_results.append(result)
                        total_images_generated += len(result.get('images', []))
                else:
                    # Only process mechanism without generating images
                    mechanism_data = process_mechanism(image_path, reaction_json_path, output_folder)
                    if mechanism_data:
                        all_results.append({'mechanism_data': mechanism_data, 'rxn_data': None, 'images': []})
            except Exception as e:
                print(f"Error processing {label}: {e}")
                continue
    
    print("\n" + "=" * 50)
    print("Batch processing complete!")
    print(f"Successfully processed: {len(all_results)} reactions")
    if generate_images:
        print(f"Total mechanism step images generated: {total_images_generated}")
    print("=" * 50)
    
    return all_results


def generate_rxn_json(mechanism_json_path, output_path=None):
    """
    Generate rxn.json file from mechanism.json containing rxn SMILES for each step
    
    Args:
        mechanism_json_path: path to the mechanism.json file
        output_path: path to save rxn.json (default: same directory as mechanism.json)
    
    Returns:
        dict with rxn data, or None if failed
    """
    if not os.path.exists(mechanism_json_path):
        print(f"Error: Mechanism JSON not found: {mechanism_json_path}")
        return None
    
    # Determine output path
    if output_path is None:
        output_dir = os.path.dirname(mechanism_json_path)
        output_path = os.path.join(output_dir, "rxn.json")
    
    try:
        # Load mechanism data
        with open(mechanism_json_path, "r", encoding="utf-8") as f:
            mechanism_data = json.load(f)
        
        mechanism = mechanism_data.get('mechanism', [])
        
        if not mechanism:
            print("Warning: No mechanism steps found in the file")
            return None
        
        rxn_list = []
        
        for step in mechanism:
            step_id = step.get('mechanism_id', '?')
            reactants = step.get('reactants', [])
            products = step.get('products', [])
            
            # Build rxn SMILES: reactant1.reactant2>>product1.product2
            reactant_smiles_list = [r.get('smiles', '') for r in reactants if r.get('smiles')]
            product_smiles_list = [p.get('smiles', '') for p in products if p.get('smiles')]
            
            if reactant_smiles_list and product_smiles_list:
                reactants_str = '.'.join(reactant_smiles_list)
                products_str = '.'.join(product_smiles_list)
                rxn_smiles = f"{reactants_str}>>{products_str}"
                
                rxn_list.append({
                    "step": str(step_id),
                    "rxn_smiles": rxn_smiles
                })
        
        # Create rxn data structure
        rxn_data = {
            "mechanism_rxn": rxn_list
        }
        
        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rxn_data, f, ensure_ascii=False, indent=4)
        
        print(f"Generated rxn.json with {len(rxn_list)} reaction steps")
        print(f"Saved to: {output_path}")
        
        return rxn_data
    
    except Exception as e:
        print(f"Error generating rxn.json: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_mechanism_images(rxn_json_path, output_dir=None):
    """
    Generate reaction images from rxn.json file using RDKit
    
    Args:
        rxn_json_path: path to the rxn.json file
        output_dir: directory to save reaction images (default: same directory as rxn.json)
    
    Returns:
        list of paths to generated image files
    """
    if not os.path.exists(rxn_json_path):
        print(f"Error: rxn.json not found: {rxn_json_path}")
        return []
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(rxn_json_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Load rxn data
        with open(rxn_json_path, "r", encoding="utf-8") as f:
            rxn_data = json.load(f)
        
        rxn_list = rxn_data.get('mechanism_rxn', [])
        
        if not rxn_list:
            print("Warning: No reaction steps found in rxn.json")
            return []
        
        saved_images = []
        
        print(f"\nGenerating mechanism images from: {rxn_json_path}")
        
        for rxn_item in rxn_list:
            step = rxn_item.get('step', '?')
            rxn_smiles = rxn_item.get('rxn_smiles', '')
            
            if not rxn_smiles:
                print(f"  Warning: Empty rxn_smiles for step {step}")
                continue
            
            try:
                # Parse reaction SMILES
                rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
                
                if rxn is None:
                    print(f"  Warning: Could not parse reaction step {step}: {rxn_smiles}")
                    continue
                
                # Draw reaction
                img = Draw.ReactionToImage(rxn, subImgSize=(400, 300))
                
                # Save image
                output_path = os.path.join(output_dir, f'mechanism_step{step}.png')
                img.save(output_path)
                saved_images.append(output_path)
                
                print(f"  ✓ Generated mechanism_step{step}.png")
                
            except Exception as e:
                print(f"  ✗ Error generating image for step {step}: {str(e)}")
                continue
        
        print(f"\nSuccessfully generated {len(saved_images)} mechanism images")
        return saved_images
    
    except Exception as e:
        print(f"Error generating mechanism images: {e}")
        import traceback
        traceback.print_exc()
        return []


def process_mechanism_complete(image_path, reaction_json_path, output_folder="mechanisms"):
    """
    Complete workflow: process mechanism image, generate rxn.json, and create reaction images
    
    Args:
        image_path: path to the mechanism image
        reaction_json_path: path to the reaction output.json file
        output_folder: folder to save results
    
    Returns:
        dict with all generated data and file paths
    """
    # Step 1: Process the mechanism image
    mechanism_data = process_mechanism(image_path, reaction_json_path, output_folder)
    
    if not mechanism_data:
        return None
    
    label = mechanism_data.get('label', 'UNKNOWN')
    mechanism_folder = os.path.join(output_folder, label)
    mechanism_json_path = os.path.join(mechanism_folder, "mechanism.json")
    
    # Step 2: Generate rxn.json from mechanism.json
    rxn_data = generate_rxn_json(mechanism_json_path)
    
    if not rxn_data:
        print("Warning: Could not generate rxn.json")
        return {
            'mechanism_data': mechanism_data,
            'rxn_data': None,
            'images': []
        }
    
    # Step 3: Generate mechanism images from rxn.json
    rxn_json_path = os.path.join(mechanism_folder, "rxn.json")
    images = generate_mechanism_images(rxn_json_path)
    
    print("\n" + "=" * 60)
    print("COMPLETE WORKFLOW FINISHED")
    print("=" * 60)
    print(f"Label: {label}")
    print(f"Mechanism steps: {len(mechanism_data.get('mechanism', []))}")
    print("Generated files:")
    print("  - mechanism.json")
    print("  - rxn.json")
    print(f"  - {len(images)} mechanism step images")
    print("=" * 60)
    
    return {
        'mechanism_data': mechanism_data,
        'rxn_data': rxn_data,
        'images': images
    }


if __name__ == "__main__":
    # Example 1: Complete workflow - process mechanism, generate rxn.json and images
    # Using paths based on the project structure
    
    # # Mechanism image path (e.g., A001a.png for answer/mechanism image)
    # image_path = "images/A002a.png"
    
    # # Reaction data path (output.json from RxnIM)
    # reaction_json_path = "RxnIM_Reactions/A002q/output.json"
    
    # # Complete workflow: mechanism analysis + rxn.json + images
    # result = process_mechanism_complete(image_path, reaction_json_path)
    
    # if result:
    #     print(f"\nExtracted mechanism data for: {result['mechanism_data'].get('label', 'UNKNOWN')}")
    #     print(f"Number of mechanism steps: {len(result['mechanism_data'].get('mechanism', []))}")
    #     print(f"Generated {len(result['images'])} mechanism images")
    
    # Example 2: Generate rxn.json and images from existing mechanism.json
    # mechanism_json_path = "mechanisms/A001/mechanism.json"
    # rxn_data = generate_rxn_json(mechanism_json_path)
    # if rxn_data:
    #     rxn_json_path = "mechanisms/A001/rxn.json"
    #     images = generate_mechanism_images(rxn_json_path)
    
    # Example 3: Batch process with prefix and number range (e.g., A001 to A078)
    batch_process_mechanisms(
        reactions_folder="RxnIM_Reactions",
        images_folder="images",
        output_folder="mechanisms",
        image_suffix="a",
        prefix="C",
        start_num=34,
        end_num=34
    )
    
    # Example 4: Batch process all reactions with a specific prefix (e.g., all "B" reactions)
    # batch_process_mechanisms(
    #     reactions_folder="RxnIM_Reactions",
    #     images_folder="images",
    #     output_folder="mechanisms",
    #     image_suffix="a",
    #     prefix="C"
    # )
    
    # Example 5: Batch process all mechanisms without filtering
    # batch_process_mechanisms(
    #     reactions_folder="RxnIM_Reactions",
    #     images_folder="images",
    #     output_folder="mechanisms",
    #     image_suffix="a"
    # )
    
    # Example 6: Process specific mechanisms with complete workflow (uncomment to use)
    # labels = ["A001", "A002", "A003"]
    # for label in labels:
    #     image_path = f"images/{label}a.png"
    #     reaction_json_path = f"RxnIM_Reactions/{label}q/output.json"
    #     if os.path.exists(image_path) and os.path.exists(reaction_json_path):
    #         process_mechanism_complete(image_path, reaction_json_path)

