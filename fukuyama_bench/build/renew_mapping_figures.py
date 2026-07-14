import json
import sys
from pathlib import Path
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions

# ================= 配置区 =================
# 如果想处理所有文件夹，请保持为 None
# 如果想处理特定文件夹，请填写文件夹名，例如: TARGET_FOLDER = "reaction_001"
TARGET_FOLDER = None 
# ==========================================

def generate_reaction_image(mapped_rxn_smiles: str, output_path: str, 
                            sub_img_size: tuple = (600, 300)) -> bool:
    """从映射的反应 SMILES 生成反应图像。"""
    try:
        rxn = rdChemReactions.ReactionFromSmarts(mapped_rxn_smiles, useSmiles=True)
        if rxn is None:
            print(f"   Failed to parse reaction: {mapped_rxn_smiles[:50]}...")
            return False
        
        img = Draw.ReactionToImage(rxn, subImgSize=sub_img_size)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"   Error generating image: {e}")
        return False


def process_mechanism_folder(folder_path: Path) -> tuple:
    """处理单个机构文件夹。"""
    mapped_rxn_file = folder_path / "mapped_rxn.json"
    
    if not mapped_rxn_file.exists():
        print(f"   Warning: {mapped_rxn_file.name} not found in {folder_path.name}")
        return (0, 0)
    
    try:
        with open(mapped_rxn_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"   Error reading {mapped_rxn_file}: {e}")
        return (0, 0)
    
    mechanism_rxn = data.get("mechanism_rxn", [])
    if not mechanism_rxn:
        print(f"   No mechanism_rxn found in {folder_path.name}")
        return (0, 0)
    
    output_dir = folder_path / "mapped_mechanism_steps"
    output_dir.mkdir(exist_ok=True)
    
    success_count = 0
    total_steps = len(mechanism_rxn)
    
    for step_data in mechanism_rxn:
        step_num = step_data.get("step", "unknown")
        mapped_rxn = step_data.get("mapped_rxn", "")
        
        if not mapped_rxn:
            continue
        
        output_path = output_dir / f"mapped_step{step_num}.png"
        if generate_reaction_image(mapped_rxn, str(output_path)):
            success_count += 1
    
    return (success_count, total_steps)


def main():
    script_dir = Path(__file__).parent
    mechanisms_dir = script_dir / "mechanisms"
    
    if not mechanisms_dir.exists():
        print(f"Error: Mechanisms directory not found at {mechanisms_dir}")
        return

    # 优先级：1. 命令行参数 > 2. TARGET_FOLDER 变量 > 3. 处理所有文件夹
    target = sys.argv[1] if len(sys.argv) > 1 else TARGET_FOLDER

    if target:
        # 处理指定的单个文件夹
        specific_folder = mechanisms_dir / target
        if not specific_folder.is_dir():
            print(f"Error: Specified folder '{target}' does not exist in {mechanisms_dir}")
            return
        folders = [specific_folder]
        print(f"Mode: Processing single folder -> {target}")
    else:
        # 默认处理所有子文件夹
        folders = sorted([f for f in mechanisms_dir.iterdir() if f.is_dir()])
        print(f"Mode: Processing all {len(folders)} folders")

    print("=" * 60)
    total_success, total_steps, processed_folders = 0, 0, 0

    for folder in folders:
        print(f"Processing {folder.name}...")
        success, steps = process_mechanism_folder(folder)
        
        if steps > 0:
            processed_folders += 1
            total_success += success
            total_steps += steps
            print(f"   Generated {success}/{steps} images")
        else:
            print("   Skipped (no valid data)")

    print("=" * 60)
    if total_steps > 0:
        print(f"Summary: Processed {processed_folders} folder(s), total {total_success}/{total_steps} images.")
    else:
        print("No images were generated.")


if __name__ == "__main__":
    main()