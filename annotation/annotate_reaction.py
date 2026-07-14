#!/usr/bin/env python3
"""
Annotate complete reaction pathways using Google Gemini API or Together AI API.

This script processes reaction data in a reaction-wise manner:
- Input: flower_new_dataset_processed (each reaction has multiple steps)
- Output: flower_reaction_annotated (each reaction annotated as a complete pathway)

Uses the prompts defined in pathway_annotation_prompts.yaml.

Supported providers:
- gemini: Google Gemini API (default)
- together: Together AI API (OpenAI-compatible)
"""

import json
import os
import yaml
import argparse
import asyncio
import random
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass
from tqdm.asyncio import tqdm_asyncio


def load_gemini_api_key() -> str:
    """Load Gemini API key from gemini.key file or environment variable."""
    key_file = Path(__file__).resolve().parents[1] / "gemini.key"
    if key_file.exists():
        with open(key_file) as f:
            return f.read().strip()
    return os.environ.get("GOOGLE_API_KEY", "")


def load_together_api_key() -> str:
    """Load Together AI API key from together.key file or environment variable."""
    key_file = Path(__file__).resolve().parents[1] / "together.key"
    if key_file.exists():
        with open(key_file) as f:
            return f.read().strip()
    return os.environ.get("TOGETHER_API_KEY", "")


@dataclass
class AnnotationConfig:
    """Configuration for annotation."""
    provider: str = "gemini"  # gemini or together
    model: str = "gemini-3-flash-preview"
    max_concurrent: int = 5  # Lower to avoid rate limits
    retry_attempts: int = 5
    retry_delay: float = 2.0
    thinking_level: str = "low"  # minimal, low, medium, high (gemini only)


def load_prompts(prompts_path: Path) -> dict:
    """Load prompts from YAML file."""
    with open(prompts_path, "r") as f:
        return yaml.safe_load(f)


def format_trajectory(steps: list) -> str:
    """
    Format reaction steps into JSON trajectory for the prompt.

    Skips identity steps (reactants == products).
    """
    # Filter out identity steps
    active_steps = []
    for i, step in enumerate(steps):
        if step["reactants"].strip() == step["products"].strip():
            continue
        active_steps.append({
            "step_id": len(active_steps) + 1,
            "reactants": step["reactants"],
            "products": step["products"]
        })

    return json.dumps(active_steps, indent=2)


def format_conditions(conditions: list) -> str:
    """Format reaction conditions into a readable string."""
    if not conditions:
        return ""
    parts = []
    for c in conditions:
        role = c.get("role", "")
        text = c.get("text", "")
        smiles = c.get("smiles", "")
        if text:
            entry = f"{role}: {text}" if role else text
            if smiles:
                entry += f" ({smiles})"
            parts.append(entry)
    return "; ".join(parts)


def format_reaction_prompt(template: str, reaction: dict) -> str:
    """
    Format the template prompt with reaction data.

    Extracts starting reactants from the first step and formats the trajectory.
    Includes reaction conditions if available.
    """
    steps = reaction["steps"]

    # Get starting reactants from first step
    starting_reactants = steps[0]["reactants"]

    # Format trajectory
    trajectory = format_trajectory(steps)

    # Format conditions
    conditions = reaction.get("conditions", [])
    conditions_str = format_conditions(conditions)

    # Fill template
    prompt = template.replace("{starting_reactants}", starting_reactants)
    prompt = prompt.replace("{trajectory}", trajectory)
    prompt = prompt.replace("{conditions}", conditions_str if conditions_str else "Not specified")

    return prompt


class GeminiAnnotator:
    """Annotator using Google Gemini API."""

    def __init__(self, config: AnnotationConfig, prompts: dict):
        from google import genai
        from google.genai import types
        self.genai = genai
        self.types = types

        self.config = config
        self.prompts = prompts
        api_key = load_gemini_api_key()
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()

    async def annotate_reaction(
        self,
        reaction: dict,
        semaphore: asyncio.Semaphore
    ) -> dict:
        """Annotate a complete reaction pathway."""
        async with semaphore:
            reaction_id = reaction["reaction_id"]

            # Format the prompt
            user_prompt = format_reaction_prompt(
                self.prompts["pathway_annotator_template_prompt"],
                reaction
            )

            # Build generation config — only add thinking for models that support it
            gen_config_kwargs = {
                "system_instruction": self.prompts["pathway_annotator_system_prompt"],
                "max_output_tokens": 16384,
            }
            if "gemini-3" in self.config.model or "gemini-2.5-pro" in self.config.model:
                thinking_level_map = {
                    "minimal": "MINIMAL",
                    "low": "LOW",
                    "medium": "MEDIUM",
                    "high": "HIGH",
                }
                level_str = thinking_level_map.get(
                    self.config.thinking_level.lower(), "LOW"
                )
                thinking_level_enum = getattr(self.types.ThinkingLevel, level_str)
                gen_config_kwargs["thinking_config"] = self.types.ThinkingConfig(
                    thinking_level=thinking_level_enum
                )

            for attempt in range(self.config.retry_attempts):
                try:
                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.config.model,
                        contents=user_prompt,
                        config=self.types.GenerateContentConfig(**gen_config_kwargs)
                    )

                    # Extract active steps (non-identity)
                    steps = reaction["steps"]
                    active_steps = []
                    for i, step in enumerate(steps):
                        if step["reactants"].strip() != step["products"].strip():
                            active_steps.append({
                                "step_id": len(active_steps) + 1,
                                "reactants": step["reactants"],
                                "products": step["products"]
                            })

                    result = {
                        "reaction_id": reaction_id,
                        "num_steps": len(active_steps),
                        "steps": active_steps,
                        "annotation": response.text,
                        "status": "success"
                    }
                    if reaction.get("conditions"):
                        result["conditions"] = reaction["conditions"]
                    return result

                except Exception as e:
                    error_str = str(e)
                    if attempt < self.config.retry_attempts - 1:
                        # Longer wait for rate limit errors (429)
                        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                            wait_time = self.config.retry_delay * (2 ** attempt) * 5  # exponential backoff, longer for 429
                        else:
                            wait_time = self.config.retry_delay * (attempt + 1)
                        await asyncio.sleep(wait_time)
                    else:
                        # Extract active steps for error case too
                        steps = reaction["steps"]
                        active_steps = [
                            {"step_id": i + 1, "reactants": s["reactants"], "products": s["products"]}
                            for i, s in enumerate(steps)
                            if s["reactants"].strip() != s["products"].strip()
                        ]
                        result = {
                            "reaction_id": reaction_id,
                            "num_steps": len(active_steps),
                            "steps": active_steps,
                            "annotation": None,
                            "status": "error",
                            "error": error_str
                        }
                        if reaction.get("conditions"):
                            result["conditions"] = reaction["conditions"]
                        return result

    async def annotate_dataset(
        self,
        data: dict,
        output_path: Path,
        checkpoint_interval: int = 100,
        resume_from: str = None,
        shuffle: bool = True,
        seed: int = 42
    ) -> dict:
        """Annotate all reactions in a dataset."""
        # Load existing results if resuming
        existing_results = []
        completed_ids = set()
        if resume_from and Path(resume_from).exists():
            print(f"Resuming from checkpoint: {resume_from}")
            with open(resume_from, "r") as f:
                checkpoint_data = json.load(f)
                existing_results = checkpoint_data.get("annotations", [])
                for r in existing_results:
                    if r.get("status") == "success":
                        completed_ids.add(r["reaction_id"])
            print(f"Found {len(completed_ids)} completed reactions, skipping them")

        # Create semaphore in async context
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        # Increase thread pool size to match max_concurrent
        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_concurrent)
        loop.set_default_executor(executor)

        # Filter valid reactions (skip empty or invalid ones)
        valid_reactions = [
            r for r in data["reactions"]
            if r.get("reaction_id") != "RC" and r.get("num_steps", 0) > 0
        ]

        # Skip already completed reactions
        reactions_to_process = [
            r for r in valid_reactions
            if r["reaction_id"] not in completed_ids
        ]

        # Shuffle reactions for better distribution (important if stopping early)
        if shuffle:
            random.seed(seed)
            random.shuffle(reactions_to_process)
            print(f"Shuffled reactions with seed={seed}")

        print(f"Processing {len(reactions_to_process)} reactions (skipped {len(completed_ids)} already completed)")

        # Create all tasks upfront
        async_tasks = [
            self.annotate_reaction(reaction, semaphore)
            for reaction in reactions_to_process
        ]

        # Start with existing results
        results = list(existing_results)

        # Process with progress bar using tqdm_asyncio
        for coro in tqdm_asyncio.as_completed(async_tasks, desc="Annotating reactions", total=len(async_tasks)):
            result = await coro
            results.append(result)

            # Checkpoint periodically
            if len(results) % checkpoint_interval == 0:
                self._save_checkpoint(results, output_path)

        # Final save
        self._save_checkpoint(results, output_path)

        return {
            "annotations": results,
            "metadata": {
                "source": data.get("metadata", {}).get("source", "unknown"),
                "model": self.config.model,
                "total_reactions": len(valid_reactions),
                "total_annotated": len(results),
                "successful": sum(1 for r in results if r["status"] == "success"),
                "failed": sum(1 for r in results if r["status"] == "error")
            }
        }

    def _save_checkpoint(self, results: list, output_path: Path):
        """Save checkpoint of annotations."""
        checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint.json"
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "error")
        checkpoint_data = {
            "annotations": results,
            "metadata": {
                "total_annotated": len(results),
                "successful": successful,
                "failed": failed
            }
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"\n[Checkpoint] Saved {len(results)} | Success: {successful} | Failed: {failed}")


class TogetherAnnotator:
    """Annotator using Together AI API (OpenAI-compatible)."""

    def __init__(self, config: AnnotationConfig, prompts: dict):
        from openai import AsyncOpenAI
        import httpx

        self.config = config
        self.prompts = prompts
        api_key = load_together_api_key()
        if not api_key:
            raise ValueError("Together API key not found. Set TOGETHER_API_KEY or create together.key file.")

        # Use AsyncOpenAI with higher connection limits for true async concurrency
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=config.max_concurrent + 10,
                    max_keepalive_connections=config.max_concurrent
                ),
                timeout=httpx.Timeout(120.0, connect=30.0)
            )
        )

    async def annotate_reaction(
        self,
        reaction: dict,
        semaphore: asyncio.Semaphore
    ) -> dict:
        """Annotate a complete reaction pathway."""
        async with semaphore:
            reaction_id = reaction["reaction_id"]

            # Format the prompt
            user_prompt = format_reaction_prompt(
                self.prompts["pathway_annotator_template_prompt"],
                reaction
            )

            system_prompt = self.prompts["pathway_annotator_system_prompt"]

            for attempt in range(self.config.retry_attempts):
                try:
                    # Use native async call instead of asyncio.to_thread
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=16384,
                        temperature=0.7,
                    )

                    # Extract active steps (non-identity)
                    steps = reaction["steps"]
                    active_steps = []
                    for i, step in enumerate(steps):
                        if step["reactants"].strip() != step["products"].strip():
                            active_steps.append({
                                "step_id": len(active_steps) + 1,
                                "reactants": step["reactants"],
                                "products": step["products"]
                            })

                    return {
                        "reaction_id": reaction_id,
                        "num_steps": len(active_steps),
                        "steps": active_steps,
                        "annotation": response.choices[0].message.content,
                        "status": "success"
                    }

                except Exception as e:
                    error_str = str(e)
                    if attempt < self.config.retry_attempts - 1:
                        # Longer wait for rate limit errors (429)
                        if "429" in error_str or "rate" in error_str.lower():
                            wait_time = self.config.retry_delay * (2 ** attempt) * 5
                        else:
                            wait_time = self.config.retry_delay * (attempt + 1)
                        await asyncio.sleep(wait_time)
                    else:
                        # Extract active steps for error case too
                        steps = reaction["steps"]
                        active_steps = [
                            {"step_id": i + 1, "reactants": s["reactants"], "products": s["products"]}
                            for i, s in enumerate(steps)
                            if s["reactants"].strip() != s["products"].strip()
                        ]
                        result = {
                            "reaction_id": reaction_id,
                            "num_steps": len(active_steps),
                            "steps": active_steps,
                            "annotation": None,
                            "status": "error",
                            "error": error_str
                        }
                        if reaction.get("conditions"):
                            result["conditions"] = reaction["conditions"]
                        return result

    async def annotate_dataset(
        self,
        data: dict,
        output_path: Path,
        checkpoint_interval: int = 100,
        resume_from: str = None,
        shuffle: bool = True,
        seed: int = 42
    ) -> dict:
        """Annotate all reactions in a dataset."""
        # Load existing results if resuming
        existing_results = []
        completed_ids = set()
        if resume_from and Path(resume_from).exists():
            print(f"Resuming from checkpoint: {resume_from}")
            with open(resume_from, "r") as f:
                checkpoint_data = json.load(f)
                existing_results = checkpoint_data.get("annotations", [])
                for r in existing_results:
                    if r.get("status") == "success":
                        completed_ids.add(r["reaction_id"])
            print(f"Found {len(completed_ids)} completed reactions, skipping them")

        # Create semaphore in async context
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        # Increase thread pool size to match max_concurrent
        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_concurrent)
        loop.set_default_executor(executor)

        # Filter valid reactions (skip empty or invalid ones)
        valid_reactions = [
            r for r in data["reactions"]
            if r.get("reaction_id") != "RC" and r.get("num_steps", 0) > 0
        ]

        # Skip already completed reactions
        reactions_to_process = [
            r for r in valid_reactions
            if r["reaction_id"] not in completed_ids
        ]

        # Shuffle reactions for better distribution (important if stopping early)
        if shuffle:
            random.seed(seed)
            random.shuffle(reactions_to_process)
            print(f"Shuffled reactions with seed={seed}")

        print(f"Processing {len(reactions_to_process)} reactions (skipped {len(completed_ids)} already completed)")

        # Create all tasks upfront
        async_tasks = [
            self.annotate_reaction(reaction, semaphore)
            for reaction in reactions_to_process
        ]

        # Start with existing results
        results = list(existing_results)

        # Process with progress bar using tqdm_asyncio
        for coro in tqdm_asyncio.as_completed(async_tasks, desc="Annotating reactions", total=len(async_tasks)):
            result = await coro
            results.append(result)

            # Checkpoint periodically
            if len(results) % checkpoint_interval == 0:
                self._save_checkpoint(results, output_path)

        # Final save
        self._save_checkpoint(results, output_path)

        return {
            "annotations": results,
            "metadata": {
                "source": data.get("metadata", {}).get("source", "unknown"),
                "model": self.config.model,
                "total_reactions": len(valid_reactions),
                "total_annotated": len(results),
                "successful": sum(1 for r in results if r["status"] == "success"),
                "failed": sum(1 for r in results if r["status"] == "error")
            }
        }

    def _save_checkpoint(self, results: list, output_path: Path):
        """Save checkpoint of annotations."""
        checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint.json"
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "error")
        checkpoint_data = {
            "annotations": results,
            "metadata": {
                "total_annotated": len(results),
                "successful": successful,
                "failed": failed
            }
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"\n[Checkpoint] Saved {len(results)} | Success: {successful} | Failed: {failed}")


def annotate_sync(
    data: dict,
    prompts: dict,
    config: AnnotationConfig,
    output_path: Path,
    resume_from: str = None,
    shuffle: bool = True,
    seed: int = 42
) -> dict:
    """Synchronous wrapper for annotation."""
    if config.provider == "together":
        annotator = TogetherAnnotator(config, prompts)
    else:
        annotator = GeminiAnnotator(config, prompts)

    return asyncio.run(annotator.annotate_dataset(
        data, output_path, resume_from=resume_from, shuffle=shuffle, seed=seed
    ))


def main():
    parser = argparse.ArgumentParser(description="Annotate reaction pathways with Gemini or Together AI")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSON file from preprocessing (flower_new_dataset_processed)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file for annotations (flower_reaction_annotated)")
    parser.add_argument("--prompts", type=str, default="pathway_annotation_prompts.yaml",
                        help="Path to prompts YAML file")
    parser.add_argument("--provider", type=str, default="gemini",
                        choices=["gemini", "together"],
                        help="API provider: gemini or together (default: gemini)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model to use (default: gemini-2.5-flash for gemini, Qwen/Qwen3-235B-A22B-Instruct-2507-tput for together)")
    parser.add_argument("--max-concurrent", type=int, default=30,
                        help="Maximum concurrent API requests")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of reactions to annotate (for testing)")
    parser.add_argument("--thinking-level", type=str, default="low",
                        choices=["minimal", "low", "medium", "high"],
                        help="Gemini thinking level (gemini only)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint file (auto-detects if not specified)")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="Shuffle reactions before processing (default: True)")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                        help="Do not shuffle reactions")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")

    args = parser.parse_args()

    # Load data
    input_path = Path(args.input)
    output_path = Path(args.output)
    prompts_path = Path(args.prompts)

    print(f"Loading prompts from {prompts_path}...")
    prompts = load_prompts(prompts_path)

    print(f"Loading data from {input_path}...")
    with open(input_path, "r") as f:
        data = json.load(f)

    # Apply limit if specified
    if args.limit:
        data["reactions"] = data["reactions"][:args.limit]

    # Count valid reactions
    valid_reactions = [
        r for r in data["reactions"]
        if r.get("reaction_id") != "RC" and r.get("num_steps", 0) > 0
    ]

    print(f"Loaded {len(valid_reactions)} valid reactions")

    # Set default model based on provider
    model = args.model
    if model is None:
        if args.provider == "together":
            model = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
        else:
            model = "gemini-2.5-flash"

    # Configure
    config = AnnotationConfig(
        provider=args.provider,
        model=model,
        max_concurrent=args.max_concurrent,
        thinking_level=args.thinking_level
    )

    # Auto-detect checkpoint file if not specified
    resume_from = args.resume
    if resume_from is None:
        checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint.json"
        if checkpoint_path.exists():
            resume_from = str(checkpoint_path)
            print(f"Auto-detected checkpoint file: {resume_from}")

    # Annotate
    print(f"Starting annotation with {config.provider} provider, model: {config.model}...")
    results = annotate_sync(
        data, prompts, config, output_path,
        resume_from=resume_from, shuffle=args.shuffle, seed=args.seed
    )

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAnnotation complete!")
    print(f"  Total reactions: {results['metadata']['total_reactions']}")
    print(f"  Successful: {results['metadata']['successful']}")
    print(f"  Failed: {results['metadata']['failed']}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
