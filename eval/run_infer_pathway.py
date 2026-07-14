#!/usr/bin/env python3
"""
Inference script for complete pathway mechanism prediction using various LLMs.

This script loads mechanism data and uses LLMs to predict the complete multi-step
reaction mechanism from starting materials to final products, supporting top-K predictions.
"""

import os
import json
import yaml
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, Optional, List, Tuple
from datetime import datetime

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def load_api_key() -> str:
    """Load Gemini API key from env or gemini.key at repo root."""
    env = os.environ.get("GOOGLE_API_KEY", "")
    if env:
        return env
    script_dir = os.path.dirname(os.path.abspath(__file__))
    key_file = os.path.join(script_dir, "..", "gemini.key")
    if os.path.exists(key_file):
        with open(key_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


# API configurations (keys from environment / local secret files only)
MODELS_CONFIG = {
    "Gemini-3.0-Pro": {
        "api_key": load_api_key(),
        "model": "gemini-3-pro-preview",
        "type": "google"
    },
    "GPT-5": {
        "api_key": _env("OPENAI_API_KEY"),
        "model": "gpt-5",
        "type": "openai"
    },
    "Qwen3-30B-A3B": {
        "api_key": _env("SILICONFLOW_API_KEY"),
        "model": "Qwen/Qwen3-30B-A3B",
        "type": "siliconflow"
    },
    "Qwen3-235B-A22B": {
        "api_key": _env("SILICONFLOW_API_KEY"),
        "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "type": "siliconflow"
    }
}

# vLLM runtime config (set via command line)
VLLM_CONFIG = {
    "base_url": None,
    "model": None,
    "api_key": "token-abc123",  # vLLM default, can be overridden
    "max_tokens": 14000,        # max response tokens; increase to reduce extraction errors
}

MODEL_ALIASES = {
    "gemini": "Gemini-3.0-Pro",
    "gpt": "GPT-5",
    "gpt5": "GPT-5",
    "qwen": "Qwen3-30B-A3B",
    "qwen3": "Qwen3-30B-A3B",
    "qwen30b": "Qwen3-30B-A3B",
    "qwen235": "Qwen3-235B-A22B",
    "qwen235b": "Qwen3-235B-A22B",
    "vllm": "vllm",  # Special case: use vLLM endpoint
    "all": None  # Special case: run all models
}

# Thread-safe print lock
print_lock = Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        print(*args, **kwargs)


def load_prompts(yaml_path: str) -> Dict[str, str]:
    """Load prompts from YAML file."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_pathway_data(
    mechanisms_dir: str,
    case_ids: Optional[List[str]] = None,
    prefix: Optional[str] = None
) -> List[Dict]:
    """
    Load pathway data from the mechanisms directory.
    
    Args:
        mechanisms_dir: Path to the mechanisms directory
        case_ids: Optional list of case IDs to load (e.g., ['A001', 'C013'])
        prefix: Optional prefix to filter cases (e.g., 'A', 'B', 'C')
    
    Returns:
        List of sample dictionaries with pathway data
    """
    samples = []
    
    # Get all subdirectories in the mechanisms folder
    subdirs = [d for d in os.listdir(mechanisms_dir) 
               if os.path.isdir(os.path.join(mechanisms_dir, d))]
    
    # Filter by prefix if specified
    if prefix:
        prefix_upper = prefix.upper()
        subdirs = [d for d in subdirs if d.upper().startswith(prefix_upper)]
    
    # Filter by case_ids if specified
    if case_ids:
        subdirs = [d for d in subdirs if d in case_ids]
    
    subdirs.sort()
    
    for case_id in subdirs:
        mechanism_path = os.path.join(mechanisms_dir, case_id, "mechanism.json")
        rxn_path = os.path.join(mechanisms_dir, case_id, "rxn.json")
        mapped_rxn_path = os.path.join(mechanisms_dir, case_id, "mapped_rxn.json")
        
        if not os.path.exists(mechanism_path):
            print(f"Warning: {mechanism_path} not found, skipping...")
            continue
        
        with open(mechanism_path, 'r', encoding='utf-8') as f:
            mechanism_data = json.load(f)
        
        # Load rxn.json if exists (simple reaction SMILES)
        rxn_data = None
        if os.path.exists(rxn_path):
            with open(rxn_path, 'r', encoding='utf-8') as f:
                rxn_data = json.load(f)
        
        # Load mapped_rxn if exists (atom-mapped reaction SMILES)
        mapped_rxn_data = None
        if os.path.exists(mapped_rxn_path):
            with open(mapped_rxn_path, 'r', encoding='utf-8') as f:
                mapped_rxn_data = json.load(f)
        
        # Get reactions info
        reactions = mechanism_data.get("reactions", [])
        mechanism_steps = mechanism_data.get("mechanism", [])
        
        if not reactions or not mechanism_steps:
            print(f"Warning: Empty reactions or mechanism in {case_id}, skipping...")
            continue
        
        # Get starting reactants from the first mechanism step
        first_step = mechanism_steps[0]
        starting_reactants = first_step.get("reactants", [])
        
        # Get conditions from reactions
        conditions = []
        for rxn in reactions:
            conditions.extend(rxn.get("conditions", []))
        
        # Get final product from mechanism
        final_step = mechanism_steps[-1]
        final_products = final_step.get("products", [])
        
        # Build ground truth pathway (list of intermediate products)
        ground_truth_pathway = []
        for step in mechanism_steps:
            step_products = step.get("products", [])
            ground_truth_pathway.append({
                "step_id": step.get("mechanism_id"),
                "products": [p.get("smiles") for p in step_products]
            })
        
        # Get rxn_smiles from rxn.json (first step only for input)
        rxn_smiles = ""
        if rxn_data:
            rxn_list = rxn_data.get("mechanism_rxn", [])
            if rxn_list:
                rxn_smiles = rxn_list[0].get("rxn_smiles", "")
        
        # Get mapped_rxn from mapped_rxn.json (first step only for input)
        mapped_rxn = ""
        mapped_rxns = []
        if mapped_rxn_data:
            mapped_rxns = mapped_rxn_data.get("mechanism_rxn", [])
            if mapped_rxns:
                mapped_rxn = mapped_rxns[0].get("mapped_rxn", "")

        # Extract atom-mapped starting reactants from first mapped_rxn
        mapped_starting_reactants = ""
        if mapped_rxn and ">>" in mapped_rxn:
            mapped_starting_reactants = mapped_rxn.split(">>")[0]

        samples.append({
            "case_id": case_id,
            "starting_reactants": [r.get("smiles") for r in starting_reactants],
            "mapped_starting_reactants": mapped_starting_reactants,
            "starting_reactants_full": starting_reactants,
            "rxn_smiles": rxn_smiles,
            "mapped_rxn": mapped_rxn,
            "conditions": conditions,
            "final_products": [p.get("smiles") for p in final_products],
            "ground_truth_pathway": ground_truth_pathway,
            "ground_truth_steps": len(mechanism_steps),
            "mapped_rxns": mapped_rxns
        })
    
    return samples


def load_previous_results(result_file: str) -> Tuple[Dict, List[Dict], List[int]]:
    """
    Load previous results and identify error items.
    
    Args:
        result_file: Path to the previous result JSON file
    
    Returns:
        Tuple of (full_data, error_samples, error_indices)
    """
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get("results", [])
    error_samples = []
    error_indices = []
    
    for idx, item in enumerate(results):
        # Check if any prediction has errors
        has_error = False
        for pred in item.get("predictions", []):
            if pred.get("error"):
                has_error = True
                break
        
        if has_error or item.get("error"):
            # Reconstruct sample for re-inference
            sample = {
                "case_id": item["case_id"],
                "starting_reactants": item["starting_reactants"],
                "conditions": item.get("conditions", []),
                "ground_truth_pathway": item["ground_truth_pathway"],
                "ground_truth_steps": item["ground_truth_steps"],
            }
            error_samples.append(sample)
            error_indices.append(idx)
    
    return data, error_samples, error_indices


def merge_retry_results(
    original_data: Dict,
    new_results: List[Dict],
    error_indices: List[int]
) -> Dict:
    """Merge retry results back into the original data."""
    results = original_data.get("results", [])
    
    for new_result, idx in zip(new_results, error_indices):
        results[idx] = new_result
    
    original_data["results"] = results
    original_data["last_retry"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return original_data


def call_google_api(system_prompt: str, user_prompt: str, config: Dict, temperature: float = 0.7) -> Tuple[str, Optional[Dict]]:
    """Call Google Gemini API with retry logic for rate limiting.

    Returns:
        Tuple of (response_text, usage_dict)
    """
    from google.genai import Client
    from google.genai.types import Part, Content, GenerateContentConfig
    from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

    client = Client(api_key=config["api_key"])

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def generate_with_retry():
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        text_part = Part(text=full_prompt)
        content = Content(parts=[text_part])

        gen_config = GenerateContentConfig(temperature=temperature)

        response = client.models.generate_content(
            model=config["model"],
            contents=content,
            config=gen_config
        )

        # Extract token usage
        usage = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                "thinking_tokens": getattr(response.usage_metadata, 'thoughts_token_count', 0) or 0,
                "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
            }
        return response.text, usage

    return generate_with_retry()


def call_openai_api(system_prompt: str, user_prompt: str, config: Dict, temperature: float = 0.7) -> str:
    """Call OpenAI API."""
    from openai import OpenAI
    
    client = OpenAI(api_key=config["api_key"])
    
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


def call_siliconflow_api(system_prompt: str, user_prompt: str, config: Dict, temperature: float = 0.7) -> str:
    """Call SiliconFlow API (Qwen)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=config["api_key"],
        base_url="https://api.siliconflow.cn/v1"
    )

    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=32768,
        temperature=temperature
    )
    return response.choices[0].message.content


def call_vllm_api(system_prompt: str, user_prompt: str, config: Dict, temperature: float = 0.7) -> Tuple[str, Optional[Dict]]:
    """
    Call vLLM OpenAI-compatible API endpoint.

    Args:
        system_prompt: System prompt for the model
        user_prompt: User prompt
        config: Dictionary with base_url, model, and api_key
        temperature: Sampling temperature

    Returns:
        Tuple of (response_text, usage_dict)
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=config.get("api_key", "token-abc123"),
        base_url=config["base_url"]
    )

    # Always disable thinking mode for vLLM (Qwen3 models emit <think> tags otherwise)
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=config.get("max_tokens", 14000),
        temperature=temperature,
        extra_body=extra_body
    )

    # Extract usage info
    usage = None
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

    text = response.choices[0].message.content or ""
    # Strip <think> blocks — Qwen3 emits these even with enable_thinking=False
    if "<think>" in text:
        import re
        # Case 1: closed <think>...</think> block — remove it
        text = re.sub(r'<think>[\s\S]*?</think>\s*', '', text)
        # Case 2: unclosed <think> with useful content after (e.g. "<think>\nizard\n\n## Reasoning...")
        if "<think>" in text:
            # Try to find where real content starts (## Reasoning or ```json)
            match = re.search(r'<think>[\s\S]*?(## Reasoning|```json|\[{)', text)
            if match:
                text = text[match.start(1):]
            else:
                text = ""  # all garbage (e.g. 闷闷闷 loop)
    return text, usage


def call_model(model_name: str, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> Tuple[str, Optional[str], Optional[Dict]]:
    """
    Call the appropriate API based on model type.

    Returns:
        Tuple of (response_text, error_message, usage_dict)
    """
    # Special handling for vLLM
    if model_name == "vllm":
        if not VLLM_CONFIG.get("base_url") or not VLLM_CONFIG.get("model"):
            return "", "vLLM not configured. Use --vllm_base_url and --vllm_model", None
        try:
            text, usage = call_vllm_api(system_prompt, user_prompt, VLLM_CONFIG, temperature)
            return text, None, usage
        except Exception as e:
            return "", f"Error: {str(e)}", None

    config = MODELS_CONFIG[model_name]

    try:
        if config["type"] == "google":
            text, usage = call_google_api(system_prompt, user_prompt, config, temperature)
            return text, None, usage
        elif config["type"] == "openai":
            return call_openai_api(system_prompt, user_prompt, config, temperature), None, None
        elif config["type"] == "siliconflow":
            return call_siliconflow_api(system_prompt, user_prompt, config, temperature), None, None
        else:
            return "", f"Unknown API type: {config['type']}", None
    except Exception as e:
        return "", f"Error: {str(e)}", None


def format_user_prompt(template: str, sample: Dict) -> str:
    """Format the user prompt template with sample data."""
    # Get rxn_smiles and mapped_rxn
    rxn_smiles = sample.get("rxn_smiles", "")
    mapped_rxn = sample.get("mapped_rxn", "")
    
    # Fallback to starting_reactants if rxn_smiles is empty
    if not rxn_smiles:
        rxn_smiles = ".".join(sample["starting_reactants"])
    
    # Format conditions if available
    conditions = sample.get("conditions", [])
    conditions_str = ""
    if conditions:
        condition_parts = []
        for cond in conditions:
            role = cond.get("role", "")
            text = cond.get("text", cond.get("smiles", ""))
            step = cond.get("reaction_step", "")
            if text:
                condition_parts.append(f"{role}: {text}" + (f" (step {step})" if step else ""))
        conditions_str = "\n".join(condition_parts)
    
    # Replace placeholders in template
    user_prompt = template.replace("{rxn_smiles}", rxn_smiles)
    user_prompt = user_prompt.replace("{mapped_rxn}", mapped_rxn)
    user_prompt = user_prompt.replace("{reaction_conditions}", conditions_str)
    
    # Use atom-mapped starting reactants if available (matches SFT/RL training format)
    mapped_sr = sample.get("mapped_starting_reactants", "")
    user_prompt = user_prompt.replace("{starting_reactants}", mapped_sr or rxn_smiles)
    if "{conditions}" in user_prompt:
        user_prompt = user_prompt.replace("{conditions}", conditions_str)
    
    return user_prompt


def process_single_sample_topk(
    sample: Dict,
    model_name: str,
    system_prompt: str,
    template_prompt: str,
    idx: int,
    total: int,
    sample_size: int = 8,
    temperature: float = 0.7
) -> Dict:
    """
    Process a single sample with multiple sampling.
    
    Args:
        sample: Sample dictionary
        model_name: Name of the model to use
        system_prompt: System prompt for the model
        template_prompt: Template for user prompt
        idx: Current index
        total: Total number of samples
        sample_size: Number of samples to generate
        temperature: Sampling temperature for diversity
    
    Returns:
        Result dictionary with predictions
    """
    case_id = sample["case_id"]
    
    # Format user prompt
    user_prompt = format_user_prompt(template_prompt, sample)
    
    safe_print(f"[{idx+1}/{total}] Processing {case_id} (sample_size={sample_size})...")
    
    predictions = []

    for k in range(sample_size):
        # Use temperature=0 for k=0 (greedy), higher temperature for diversity
        temp = 0.0 if k == 0 else temperature

        response, error, usage = call_model(model_name, system_prompt, user_prompt, temp)

        pred_data = {
            "k": k + 1,
            "temperature": temp,
            "model_response": response,
            "error": error
        }
        if usage:
            pred_data["usage"] = usage
        predictions.append(pred_data)

        if error:
            safe_print(f"  [{case_id}] k={k+1} ERROR: {error}")
        else:
            usage_str = ""
            if usage:
                think = usage.get('thinking_tokens', 0)
                if think:
                    usage_str = f" (in:{usage['prompt_tokens']} think:{think} out:{usage['completion_tokens']} total:{usage['total_tokens']})"
                else:
                    usage_str = f" (in:{usage['prompt_tokens']} out:{usage['completion_tokens']} total:{usage['total_tokens']})"
            safe_print(f"  [{case_id}] k={k+1} Done{usage_str}")
    
    return {
        "case_id": case_id,
        "starting_reactants": sample["starting_reactants"],
        "conditions": sample.get("conditions", []),
        "ground_truth_pathway": sample["ground_truth_pathway"],
        "ground_truth_steps": sample["ground_truth_steps"],
        "predictions": predictions
    }


def run_inference_parallel(
    model_name: str,
    samples: List[Dict],
    system_prompt: str,
    template_prompt: str,
    workers: int = 4,
    sample_size: int = 8,
    temperature: float = 0.7,
    checkpoint_path: Optional[str] = None,
    prefix: Optional[str] = None
) -> List[Dict]:
    """
    Run inference for all samples in parallel using ThreadPoolExecutor.

    Args:
        model_name: Name of the model to use
        samples: List of sample dictionaries
        system_prompt: System prompt for the model
        template_prompt: Template for user prompt
        workers: Number of parallel workers
        sample_size: Number of samples per input
        temperature: Sampling temperature
        checkpoint_path: Path to save checkpoints after each sample
        prefix: Prefix for checkpoint metadata

    Returns:
        List of result dictionaries (sorted by original order)
    """
    total = len(samples)
    print(f"\nRunning parallel inference with {model_name} on {total} samples using {workers} workers...")
    print(f"Sample size: {sample_size}, Temperature: {temperature}")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")

    results = [None] * total
    completed_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(
                process_single_sample_topk,
                sample,
                model_name,
                system_prompt,
                template_prompt,
                idx,
                total,
                sample_size,
                temperature
            ): idx
            for idx, sample in enumerate(samples)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                sample = samples[idx]
                results[idx] = {
                    "case_id": sample["case_id"],
                    "starting_reactants": sample["starting_reactants"],
                    "conditions": sample.get("conditions", []),
                    "ground_truth_pathway": sample["ground_truth_pathway"],
                    "ground_truth_steps": sample["ground_truth_steps"],
                    "predictions": [{"k": 1, "error": f"Unexpected error: {str(e)}"}],
                    "error": f"Unexpected error: {str(e)}"
                }
                safe_print(f"  [{sample['case_id']}] UNEXPECTED ERROR: {e}")

            # Save checkpoint after each sample completes
            completed_count += 1
            if checkpoint_path:
                save_checkpoint(results, checkpoint_path, model_name, prefix, sample_size)
                safe_print(f"  [Checkpoint] Saved {completed_count}/{total} samples")

    return results


def run_inference_sequential(
    model_name: str,
    samples: List[Dict],
    system_prompt: str,
    template_prompt: str,
    delay: float = 1.0,
    sample_size: int = 8,
    temperature: float = 0.7,
    checkpoint_path: Optional[str] = None,
    prefix: Optional[str] = None
) -> List[Dict]:
    """
    Run inference for all samples sequentially.
    """
    results = []
    total = len(samples)

    print(f"\nRunning sequential inference with {model_name} on {total} samples...")
    print(f"Sample size: {sample_size}, Temperature: {temperature}")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")

    for idx, sample in enumerate(samples):
        result = process_single_sample_topk(
            sample,
            model_name,
            system_prompt,
            template_prompt,
            idx,
            total,
            sample_size,
            temperature
        )
        results.append(result)

        # Save checkpoint after each sample
        if checkpoint_path:
            save_checkpoint(results, checkpoint_path, model_name, prefix, sample_size)
            safe_print(f"  [Checkpoint] Saved {idx + 1}/{total} samples")

        # Delay to avoid rate limiting
        if idx < total - 1:
            time.sleep(delay)

    return results


def get_checkpoint_path(output_dir: str, model_name: str, prefix: Optional[str] = None, sample_size: int = 8, tag: Optional[str] = None) -> str:
    """Get the checkpoint file path (fixed name for resuming)."""
    os.makedirs(output_dir, exist_ok=True)
    safe_model_name = model_name.replace("/", "_").replace(" ", "_")
    if tag:
        safe_model_name = f"{safe_model_name}_{tag}"
    if prefix:
        return os.path.join(output_dir, f"{safe_model_name}_{prefix}_s{sample_size}_checkpoint.json")
    else:
        return os.path.join(output_dir, f"{safe_model_name}_s{sample_size}_checkpoint.json")


def find_existing_result(output_dir: str, model_name: str, prefix: Optional[str] = None, sample_size: int = 8, tag: Optional[str] = None) -> Optional[str]:
    """
    Check if a completed result file already exists.

    Returns:
        Path to the existing result file, or None if not found.
    """
    import glob

    safe_model_name = model_name.replace("/", "_").replace(" ", "_")
    if tag:
        safe_model_name = f"{safe_model_name}_{tag}"
    if prefix:
        pattern = os.path.join(output_dir, f"{safe_model_name}_{prefix}_s{sample_size}_*.json")
    else:
        pattern = os.path.join(output_dir, f"{safe_model_name}_s{sample_size}_*.json")

    # Find all matching files (excluding checkpoint files)
    matches = [f for f in glob.glob(pattern) if not f.endswith("_checkpoint.json")]

    if matches:
        # Return the most recent one
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0]

    return None


def load_checkpoint(checkpoint_path: str) -> Tuple[Dict, set]:
    """
    Load checkpoint and return completed case IDs.

    Returns:
        Tuple of (checkpoint_data, set of completed case_ids)
    """
    if not os.path.exists(checkpoint_path):
        return None, set()

    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        completed = set()
        for result in data.get("results", []):
            # Only consider complete if no errors in predictions
            has_error = False
            for pred in result.get("predictions", []):
                if pred.get("error"):
                    has_error = True
                    break
            if not has_error and not result.get("error"):
                completed.add(result["case_id"])

        return data, completed
    except Exception as e:
        print(f"Warning: Failed to load checkpoint: {e}")
        return None, set()


def save_checkpoint(results: List[Dict], checkpoint_path: str, model_name: str,
                   prefix: Optional[str] = None, sample_size: int = 8):
    """Save checkpoint with current results (thread-safe)."""
    # Filter out None results
    valid_results = [r for r in results if r is not None]

    output_data = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "prefix": prefix,
        "sample_size": sample_size,
        "total_samples": len(valid_results),
        "results": valid_results
    }

    # Write to temp file first, then rename (atomic operation)
    temp_path = checkpoint_path + ".tmp"
    with print_lock:  # Use existing lock for thread safety
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, checkpoint_path)


def save_results(results: List[Dict], output_dir: str, model_name: str,
                 prefix: Optional[str] = None, sample_size: int = 8, temperature: float = 1.0, tag: Optional[str] = None):
    """Save inference results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    safe_model_name = model_name.replace("/", "_").replace(" ", "_")
    if tag:
        safe_model_name = f"{safe_model_name}_{tag}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if prefix:
        output_file = os.path.join(output_dir, f"{safe_model_name}_{prefix}_s{sample_size}_{timestamp}.json")
    else:
        output_file = os.path.join(output_dir, f"{safe_model_name}_s{sample_size}_{timestamp}.json")

    output_data = {
        "model": model_name,
        "timestamp": timestamp,
        "prefix": prefix,
        "sample_size": sample_size,
        "temperature": temperature,
        "total_samples": len(results),
        "results": results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    return output_file


def save_updated_results(data: Dict, output_file: str):
    """Save updated results back to the original file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nUpdated results saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM inference for complete pathway mechanism prediction"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen30b",
        choices=list(MODEL_ALIASES.keys()) + list(MODELS_CONFIG.keys()),
        help="Model to use (default: qwen30b)"
    )
    parser.add_argument(
        "--mechanisms_dir",
        type=str,
        default="../fukuyama_bench/mechanisms",
        help="Path to mechanisms directory"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="prompts/infer_pathway_prompts.yaml",
        help="Path to prompts YAML file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../outputs/eval/pathway",
        help="Directory to save results"
    )
    parser.add_argument(
        "--prefix", "-p",
        type=str,
        choices=["A", "B", "C", "a", "b", "c"],
        default=None,
        help="Filter cases by prefix (A, B, or C)"
    )
    parser.add_argument(
        "--cases",
        type=str,
        nargs="+",
        default=None,
        help="Specific case IDs to process (e.g., A001 C013)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds for sequential mode (default: 1.0)"
    )
    parser.add_argument(
        "--sample_size", "-s",
        type=int,
        default=8,
        help="Number of samples to generate per input (default: 8)"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Sampling temperature for top-K predictions (default: 0.7)"
    )
    parser.add_argument(
        "--retry", "-r",
        type=str,
        default=None,
        help="Path to previous result file to retry only error items"
    )
    # vLLM endpoint configuration
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default=None,
        help="vLLM server base URL (e.g., http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default=None,
        help="Model name/path for vLLM (e.g., ./checkpoints/qwen3-30b-pathway)"
    )
    parser.add_argument(
        "--vllm_api_key",
        type=str,
        default="token-abc123",
        help="API key for vLLM server (default: token-abc123)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=14000,
        help="Max response tokens for vLLM (default: 14000). Increase to reduce extraction errors."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Custom tag for output filename (e.g., 'pathway' or 'knowmol'). Useful to distinguish different vLLM model runs."
    )

    args = parser.parse_args()

    # Configure vLLM if specified
    if args.vllm_base_url:
        VLLM_CONFIG["base_url"] = args.vllm_base_url
        VLLM_CONFIG["model"] = args.vllm_model or "default"
        VLLM_CONFIG["api_key"] = args.vllm_api_key
        VLLM_CONFIG["max_tokens"] = args.max_tokens
    
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve paths relative to script directory
    if not os.path.isabs(args.mechanisms_dir):
        args.mechanisms_dir = os.path.join(script_dir, args.mechanisms_dir)
    if not os.path.isabs(args.prompt_path):
        args.prompt_path = os.path.join(script_dir, args.prompt_path)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)
    
    # Handle retry mode
    if args.retry:
        retry_file = args.retry
        if not os.path.isabs(retry_file):
            retry_file = os.path.join(script_dir, retry_file)
        
        if not os.path.exists(retry_file):
            print(f"Error: Retry file not found: {retry_file}")
            return
        
        print(f"Loading previous results from: {retry_file}")
        original_data, error_samples, error_indices = load_previous_results(retry_file)
        
        if not error_samples:
            print("No errors found in the previous results. Nothing to retry.")
            return
        
        print(f"Found {len(error_samples)} error items to retry")
        
        model_name = original_data.get("model")
        if not model_name or model_name not in MODELS_CONFIG:
            print(f"Error: Unknown model '{model_name}' in result file")
            return
        
        sample_size = original_data.get("sample_size", original_data.get("top_k", 8))
        # Use temperature from file if available, otherwise use command line arg
        temperature = original_data.get("temperature", args.temperature)

        # Load prompts
        print(f"Loading prompts from: {args.prompt_path}")
        prompts = load_prompts(args.prompt_path)
        system_prompt = prompts.get("pathway_student_system_prompt", "")
        template_prompt = prompts.get("pathway_student_template_prompt", "")
        
        if not system_prompt:
            raise ValueError("Could not find 'pathway_student_system_prompt' in prompts YAML")
        
        print(f"\n{'='*60}")
        print("RETRY MODE")
        print(f"Model: {model_name}")
        print(f"Error items: {len(error_samples)}")
        print(f"Sample size: {sample_size}")
        print(f"Temperature: {temperature}")
        print(f"Workers: {args.workers}")
        print(f"{'='*60}")

        # Get checkpoint path for retry mode
        retry_checkpoint_path = retry_file.replace(".json", "_retry_checkpoint.json")
        retry_checkpoint_data, retry_completed_ids = load_checkpoint(retry_checkpoint_path)

        # Filter out already retried samples
        samples_to_retry = error_samples
        previous_retry_results = {}
        if retry_completed_ids:
            print(f"\nFound retry checkpoint with {len(retry_completed_ids)} completed cases")
            samples_to_retry = [s for s in error_samples if s["case_id"] not in retry_completed_ids]
            if retry_checkpoint_data:
                for r in retry_checkpoint_data.get("results", []):
                    previous_retry_results[r["case_id"]] = r
            print(f"Resuming retry with {len(samples_to_retry)} remaining cases")

        if len(samples_to_retry) == 0:
            print("All retry samples already completed.")
            new_results = [previous_retry_results.get(s["case_id"]) for s in error_samples]
        elif args.workers > 1:
            new_results = run_inference_parallel(
                model_name=model_name,
                samples=samples_to_retry,
                system_prompt=system_prompt,
                template_prompt=template_prompt,
                workers=args.workers,
                sample_size=sample_size,
                temperature=temperature,
                checkpoint_path=retry_checkpoint_path,
                prefix=None
            )
            # Merge with previous retry results
            if previous_retry_results:
                new_results = [previous_retry_results.get(s["case_id"]) or
                              next((r for r in new_results if r and r["case_id"] == s["case_id"]), None)
                              for s in error_samples]
        else:
            new_results = run_inference_sequential(
                model_name=model_name,
                samples=samples_to_retry,
                system_prompt=system_prompt,
                template_prompt=template_prompt,
                delay=args.delay,
                sample_size=sample_size,
                temperature=temperature,
                checkpoint_path=retry_checkpoint_path,
                prefix=None
            )
            # Merge with previous retry results
            if previous_retry_results:
                new_results = [previous_retry_results.get(s["case_id"]) or
                              next((r for r in new_results if r and r["case_id"] == s["case_id"]), None)
                              for s in error_samples]

        updated_data = merge_retry_results(original_data, new_results, error_indices)
        save_updated_results(updated_data, retry_file)

        # Clean up retry checkpoint
        if os.path.exists(retry_checkpoint_path):
            os.remove(retry_checkpoint_path)
            print(f"Retry checkpoint removed: {retry_checkpoint_path}")
        
        still_errors = sum(
            1 for r in new_results 
            if any(p.get("error") for p in r.get("predictions", []))
        )
        print("\nRetry Summary:")
        print(f"  Retried items: {len(new_results)}")
        print(f"  Now successful: {len(new_results) - still_errors}")
        print(f"  Still failing: {still_errors}")
        
        return
    
    # Normal mode
    prefix = args.prefix.upper() if args.prefix else None
    
    # Resolve model name from alias
    model_name = args.model
    if model_name in MODEL_ALIASES:
        if MODEL_ALIASES[model_name] is None:
            models_to_run = list(MODELS_CONFIG.keys())
        else:
            models_to_run = [MODEL_ALIASES[model_name]]
    else:
        models_to_run = [model_name]
    
    # Load prompts
    print(f"Loading prompts from: {args.prompt_path}")
    prompts = load_prompts(args.prompt_path)
    system_prompt = prompts.get("pathway_student_system_prompt", "")
    template_prompt = prompts.get("pathway_student_template_prompt", "")
    
    if not system_prompt:
        raise ValueError("Could not find 'pathway_student_system_prompt' in prompts YAML")
    
    # Load pathway data
    print(f"Loading pathway data from: {args.mechanisms_dir}")
    if prefix:
        print(f"Filtering by prefix: {prefix}")
    samples = load_pathway_data(args.mechanisms_dir, args.cases, prefix)
    print(f"Loaded {len(samples)} cases")
    
    if args.limit:
        samples = samples[:args.limit]
        print(f"Limited to first {args.limit} samples")
    
    # Run inference for each model
    for model in models_to_run:
        # Check if result file already exists (completed run)
        existing_result = find_existing_result(args.output_dir, model, prefix, args.sample_size, args.tag)
        if existing_result:
            print(f"\n{'='*60}")
            print(f"Model: {model}")
            if args.tag:
                print(f"Tag: {args.tag}")
            if prefix:
                print(f"Prefix: {prefix}")
            print(f"Sample size: {args.sample_size}")
            print(f"Result already exists: {existing_result}")
            print("Skipping... (delete the file to re-run)")
            print(f"{'='*60}")
            continue

        # Get checkpoint path and check for existing progress
        checkpoint_path = get_checkpoint_path(args.output_dir, model, prefix, args.sample_size, args.tag)
        checkpoint_data, completed_ids = load_checkpoint(checkpoint_path)

        # Filter out already completed samples
        samples_to_run = samples
        previous_results = {}
        if completed_ids:
            print(f"\nFound checkpoint with {len(completed_ids)} completed cases")
            samples_to_run = [s for s in samples if s["case_id"] not in completed_ids]
            # Store previous results for merging
            if checkpoint_data:
                for r in checkpoint_data.get("results", []):
                    previous_results[r["case_id"]] = r
            print(f"Resuming with {len(samples_to_run)} remaining cases")

        print(f"\n{'='*60}")
        print(f"Model: {model}")
        if args.tag:
            print(f"Tag: {args.tag}")
        if prefix:
            print(f"Prefix: {prefix}")
        print(f"Sample size: {args.sample_size}")
        print(f"Temperature: {args.temperature}")
        print(f"Workers: {args.workers}")
        print(f"Total cases: {len(samples)}, To run: {len(samples_to_run)}")
        print(f"{'='*60}")

        if len(samples_to_run) == 0:
            print("All samples already completed. Skipping...")
            # Use previous results
            results = [previous_results.get(s["case_id"]) for s in samples]
        elif args.workers > 1:
            new_results = run_inference_parallel(
                model_name=model,
                samples=samples_to_run,
                system_prompt=system_prompt,
                template_prompt=template_prompt,
                workers=args.workers,
                sample_size=args.sample_size,
                temperature=args.temperature,
                checkpoint_path=checkpoint_path,
                prefix=prefix
            )
            # Merge with previous results
            new_results_dict = {r["case_id"]: r for r in new_results}
            previous_results.update(new_results_dict)
            results = [previous_results.get(s["case_id"]) for s in samples]
        else:
            new_results = run_inference_sequential(
                model_name=model,
                samples=samples_to_run,
                system_prompt=system_prompt,
                template_prompt=template_prompt,
                delay=args.delay,
                sample_size=args.sample_size,
                temperature=args.temperature,
                checkpoint_path=checkpoint_path,
                prefix=prefix
            )
            # Merge with previous results
            new_results_dict = {r["case_id"]: r for r in new_results}
            previous_results.update(new_results_dict)
            results = [previous_results.get(s["case_id"]) for s in samples]

        # Filter out None results
        results = [r for r in results if r is not None]

        save_results(results, args.output_dir, model, prefix, args.sample_size, args.temperature, args.tag)

        # Clean up checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"Checkpoint removed: {checkpoint_path}")

        # Print summary
        total_predictions = sum(len(r.get("predictions", [])) for r in results)
        errors = sum(
            sum(1 for p in r.get("predictions", []) if p.get("error"))
            for r in results
        )
        print("\nSummary:")
        print(f"  Total cases: {len(results)}")
        print(f"  Total predictions: {total_predictions}")
        print(f"  Successful: {total_predictions - errors}")
        print(f"  Errors: {errors}")


if __name__ == "__main__":
    main()
