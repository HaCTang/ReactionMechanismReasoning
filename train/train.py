#!/usr/bin/env python3
"""
Unified training script for Qwen3-30B-A3B with TRL.

Supports multiple dataset types:
  - chebi: Molecule captioning (SMILES/SELFIES -> description)
  - reaction: Chemical reaction prediction (Alpaca format from Gemini annotations)
  - knowmol: Multi-task molecular understanding (T1: desc->smiles, T2: selfies->smiles, T3: smiles->analysis)

Full fine-tuning with bf16 on multi-GPU (8x B200) setup.
Uses FSDP for distributed training.
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

import torch
import yaml
from datasets import load_dataset, load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>\s*")


# ============================================================================
# Prompt Templates for CheBI (Molecule Captioning)
# ============================================================================

CHEBI_SYSTEM_PROMPT = """You are a helpful chemistry assistant specialized in molecular analysis. Given a molecular representation (SMILES or SELFIES format), you provide accurate and detailed descriptions of the molecule's structure, properties, and potential roles."""

CHEBI_INSTRUCTION_TEMPLATE = """Analyze the following molecule and provide a detailed description of its structure, chemical properties, and potential biological or chemical roles.

Molecule (SMILES): {smiles}
Molecule (SELFIES): {selfies}

Please describe this molecule:"""


def format_chebi_example(example: dict, tokenizer) -> str:
    """Format a CheBI example into chat format for training."""
    smiles = example.get("SMILES", "")
    selfies = example.get("SELFIES", "")
    description = example.get("description", "")

    messages = [
        {"role": "system", "content": CHEBI_SYSTEM_PROMPT},
        {"role": "user", "content": CHEBI_INSTRUCTION_TEMPLATE.format(smiles=smiles, selfies=selfies)},
        {"role": "assistant", "content": description},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)


# ============================================================================
# Prompt Templates for Reaction Prediction (from prompts.yaml)
# ============================================================================

def load_prompts(prompt_path: str) -> dict:
    """Load prompts from yaml file."""
    with open(prompt_path, "r") as f:
        return yaml.safe_load(f)


def format_reaction_example(example: dict, tokenizer, system_prompt: str) -> str:
    """Format a reaction prediction example into chat format for training."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    # Combine instruction and input for user message
    user_content = f"{instruction}\n\n{input_text}" if input_text else instruction

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)



# ============================================================================
# Data Processing
# ============================================================================

def load_chebi_dataset(data_path: str, tokenizer):
    """Load and prepare the chebi-20-new dataset."""
    logger.info(f"Loading CheBI dataset from {data_path}")

    dataset = load_dataset(data_path)

    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Validation samples: {len(dataset['validation'])}")

    def preprocess_function(examples):
        texts = []
        for i in range(len(examples["SMILES"])):
            example = {
                "SMILES": examples["SMILES"][i],
                "SELFIES": examples["SELFIES"][i],
                "description": examples["description"][i],
            }
            text = format_chebi_example(example, tokenizer)
            texts.append(text)
        return {"text": texts}

    train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Processing train dataset",
    )

    eval_dataset = dataset["validation"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["validation"].column_names,
        desc="Processing validation dataset",
    )

    return train_dataset, eval_dataset


def load_reaction_dataset(data_path: str, eval_path: str, tokenizer, prompt_path: str):
    """Load and prepare the reaction prediction dataset (Alpaca format JSON)."""
    logger.info(f"Loading reaction dataset from {data_path}")

    # Load prompts
    prompts = load_prompts(prompt_path)
    system_prompt = prompts.get("student_system_prompt", "").strip()

    if not system_prompt:
        raise ValueError("Could not find 'student_system_prompt' in prompts.yaml")

    # Load train data
    with open(data_path, "r") as f:
        train_data = json.load(f)
    logger.info(f"Train samples: {len(train_data)}")

    # Load eval data if provided
    eval_data = None
    if eval_path and os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
        logger.info(f"Eval samples: {len(eval_data)}")

    def preprocess_data(data):
        texts = []
        for item in data:
            text = format_reaction_example(item, tokenizer, system_prompt)
            texts.append(text)
        return texts

    # Create HuggingFace datasets
    train_texts = preprocess_data(train_data)
    train_dataset = Dataset.from_dict({"text": train_texts})

    eval_dataset = None
    if eval_data:
        eval_texts = preprocess_data(eval_data)
        eval_dataset = Dataset.from_dict({"text": eval_texts})

    return train_dataset, eval_dataset


def load_knowmol_dataset(data_path: str, tokenizer):  # noqa: ARG001
    """Load and prepare the KnowMol multi-task dataset.

    Dataset uses conversational format with 'messages' field.
    SFTTrainer will automatically apply chat template.
    tokenizer is kept for API consistency but not used.
    """
    logger.info(f"Loading KnowMol dataset from {data_path}")

    # Use load_from_disk for datasets saved with save_to_disk()
    dataset = load_from_disk(data_path)

    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Validation samples: {len(dataset['validation'])}")

    # Log task distribution
    for split in ["train", "validation"]:
        task_counts = {}
        for task_type in dataset[split]["task_type"]:
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
        logger.info(f"{split} task distribution: {task_counts}")

    # Remove non-essential columns, keep 'messages' and 'chat_template_kwargs' for SFTTrainer
    columns_to_remove = ["task_type", "cid"]
    train_dataset = dataset["train"].remove_columns(columns_to_remove)
    eval_dataset = dataset["validation"].remove_columns(columns_to_remove)

    return train_dataset, eval_dataset


def load_pathway_dataset(data_path: str, eval_path: str, tokenizer):  # noqa: ARG001
    """Load and prepare the pathway prediction dataset (TRL messages format).

    Supports two modes:
    1. HuggingFace dataset: data_path like "johnny-w/flower:sft" or "johnny-w/flower"
       (contains "/" and doesn't exist as a local path)
    2. Local JSON file: data_path is a path to a JSON file with messages format

    Dataset uses conversational format with 'messages' field.
    SFTTrainer will automatically apply chat template.
    tokenizer is kept for API consistency but not used.
    """
    logger.info(f"Loading pathway dataset from {data_path}")

    if "/" in data_path and not os.path.exists(data_path):
        # HuggingFace dataset: "johnny-w/flower:sft" or "johnny-w/flower"
        if ":" in data_path:
            name, config = data_path.rsplit(":", 1)
        else:
            name, config = data_path, "sft"
        logger.info(f"Loading from HuggingFace: {name} (config={config})")
        ds = load_dataset(name, config)

        if "validation" in ds:
            train_dataset = ds["train"]
            eval_dataset = ds["validation"]
        else:
            logger.info("No validation split found, creating 5% holdout from train")
            split = ds["train"].train_test_split(test_size=0.05, seed=42)
            train_dataset = split["train"]
            eval_dataset = split["test"]

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Eval samples: {len(eval_dataset)}")
        return train_dataset, eval_dataset

    # Local JSON file
    with open(data_path, "r") as f:
        train_data = json.load(f)
    logger.info(f"Train samples: {len(train_data)}")

    # Load eval data if provided
    eval_data = None
    if eval_path and os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
        logger.info(f"Eval samples: {len(eval_data)}")

    # Create HuggingFace datasets - data already has 'messages' field
    train_dataset = Dataset.from_list(train_data)

    eval_dataset = None
    if eval_data:
        eval_dataset = Dataset.from_list(eval_data)

    return train_dataset, eval_dataset


# ============================================================================
# Model Setup
# ============================================================================

def setup_model_and_tokenizer(model_name: str, chat_template_path: str | None = None):
    """Setup model and tokenizer for full fine-tuning with bf16."""
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if chat_template_path:
        template_text = Path(chat_template_path).read_text()
        tokenizer.chat_template = template_text
        logger.info(f"Loaded custom chat template from {chat_template_path}")

    # Load model in bf16 for full fine-tuning
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    logger.info(f"Model loaded with {model.num_parameters():,} parameters")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, tokenizer


# ============================================================================
# Training
# ============================================================================

def build_common_training_kwargs(args, has_eval: bool) -> dict:
    """Build training kwargs shared by TRL and plain Transformers trainers."""
    return {
        "output_dir": args.output_dir,
        "do_train": True,
        "do_eval": has_eval,
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps if args.save_strategy == "steps" else None,
        "eval_strategy": "steps" if has_eval else "no",
        "eval_steps": args.eval_steps if has_eval else None,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": False,
        "fp16": False,
        "bf16": True,
        "bf16_full_eval": has_eval,
        "max_grad_norm": 1.0,
        "optim": "adamw_torch_fused",
        "gradient_checkpointing": not args.use_fsdp_activation_checkpointing,
        "gradient_checkpointing_kwargs": (
            {"use_reentrant": False} if not args.use_fsdp_activation_checkpointing else None
        ),
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {
            "fsdp_offload_params": False,
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            "fsdp_transformer_layer_cls_to_wrap": "Qwen3MoeDecoderLayer",
            **({"fsdp_activation_checkpointing": True} if args.use_fsdp_activation_checkpointing else {}),
        },
        "report_to": args.report_to,
        "run_name": args.run_name,
        "logging_first_step": True,
        "use_liger_kernel": args.use_liger_kernel,
        "save_safetensors": True,
        "save_only_model": True,
    }


def normalize_chat_template_output(processed: dict) -> dict:
    """Flatten single-example chat-template outputs for consistency with TRL."""
    return {k: v[0] if isinstance(v[0], list) else v for k, v in processed.items()}


def apply_chat_template_tokens(
    tokenizer,
    messages: list[dict],
    max_length: int | None,
    *,
    tools=None,
    chat_template_kwargs: dict | None = None,
    add_generation_prompt: bool = False,
    return_assistant_tokens_mask: bool = False,
) -> dict:
    """Tokenize one conversational example through the tokenizer chat template."""
    processed = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        truncation=max_length is not None,
        max_length=max_length,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        return_assistant_tokens_mask=return_assistant_tokens_mask,
        **(chat_template_kwargs or {}),
    )
    return normalize_chat_template_output(processed)


def strip_think_blocks(text: str) -> str:
    """Remove `<think>...</think>` blocks from assistant text.

    Some Qwen3 templates inject an empty think block even in non-thinking mode.
    We also recover from malformed unclosed think prefixes by keeping the first
    reasoning marker when possible.
    """
    if not text:
        return ""

    stripped = THINK_BLOCK_RE.sub("", text)
    if "<think>" in stripped:
        match = re.search(r"<think>[\s\S]*?(## Reasoning|```json|\[\{)", stripped)
        if match:
            stripped = stripped[match.start(1):]
        else:
            stripped = ""

    return stripped.replace("</think>", "").lstrip("\n")


def encode_text_chunk(tokenizer, text: str) -> list[int]:
    """Encode a pre-rendered chat chunk without adding tokenizer-level wrappers."""
    return tokenizer.encode(text, add_special_tokens=False)


def render_non_assistant_message(message: dict) -> str:
    """Render system/user messages in Qwen chat format without think blocks."""
    role = message.get("role")
    content = message.get("content") or ""

    if role in ("system", "user"):
        return f"<|im_start|>{role}\n{content}<|im_end|>\n"
    if role == "tool":
        return f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"

    raise ValueError(f"Unsupported non-assistant role for manual rendering: {role}")


def build_manual_conversation_example(
    messages: list[dict],
    tokenizer,
    max_length: int | None,
    *,
    assistant_only_loss: bool,
    tools=None,
) -> dict:
    """Manually render a conversational training example without injected think blocks."""
    if tools is not None:
        raise ValueError("Manual HF conversational rendering does not support tool schemas yet")

    input_ids: list[int] = []
    labels: list[int] = []
    prompt_text_parts: list[str] = []
    target_text_parts: list[str] = []

    for message in messages:
        role = message.get("role")

        if role == "assistant":
            header_text = "<|im_start|>assistant\n"
            target_text = f"{strip_think_blocks(message.get('content') or '')}<|im_end|>\n"

            header_ids = encode_text_chunk(tokenizer, header_text)
            target_ids = encode_text_chunk(tokenizer, target_text)

            input_ids.extend(header_ids)
            labels.extend([-100] * len(header_ids))
            input_ids.extend(target_ids)
            if assistant_only_loss:
                labels.extend(target_ids)
            else:
                labels.extend(target_ids)

            prompt_text_parts.append(header_text)
            target_text_parts.append(target_text)
        else:
            chunk_text = render_non_assistant_message(message)
            chunk_ids = encode_text_chunk(tokenizer, chunk_text)
            input_ids.extend(chunk_ids)
            labels.extend([-100] * len(chunk_ids) if assistant_only_loss else chunk_ids)
            prompt_text_parts.append(chunk_text)

    if max_length is not None:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt_text": "".join(prompt_text_parts),
        "target_text": "".join(target_text_parts),
        "text": tokenizer.decode(input_ids),
    }


def tokenize_for_hf_trainer(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    split_name: str,
    *,
    assistant_only_loss: bool,
) -> Dataset:
    """Convert `messages` or preformatted `text` datasets into tokenized LM samples."""
    logger.info(f"Tokenizing {split_name} dataset for plain HF Trainer")

    def preprocess_function(examples):
        if "messages" in examples:
            input_ids = []
            attention_masks = []
            labels = []
            tools_list = examples.get("tools")

            for idx, messages in enumerate(examples["messages"]):
                tools = tools_list[idx] if tools_list is not None else None

                processed = build_manual_conversation_example(
                    messages,
                    tokenizer,
                    max_length,
                    assistant_only_loss=assistant_only_loss,
                    tools=tools,
                )

                input_ids.append(processed["input_ids"])
                attention_masks.append(processed.get("attention_mask", [1] * len(processed["input_ids"])))
                labels.append(processed["labels"])

            tokenized = {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
                "labels": labels,
            }
        elif "text" in examples:
            if assistant_only_loss:
                raise ValueError("assistant_only_loss with HF backend requires conversational `messages` datasets")
            tokenized = tokenizer(
                examples["text"],
                truncation=max_length is not None,
                max_length=max_length,
                padding=False,
            )
            tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        else:
            raise ValueError("Expected dataset to contain either 'messages' or 'text' field")

        return tokenized

    return dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {split_name} dataset",
    )


def build_trl_trainer(model, tokenizer, train_dataset, eval_dataset, args):
    """Construct the existing TRL SFT trainer."""
    from trl import SFTConfig, SFTTrainer

    if args.assistant_only_loss and "{% generation %}" not in (tokenizer.chat_template or ""):
        raise ValueError(
            "assistant_only_loss with TRL requires a chat template that emits assistant masks. "
            "This tokenizer template lacks `{% generation %}` support; use `--trainer_backend hf` instead."
        )

    training_args = SFTConfig(
        **build_common_training_kwargs(args, eval_dataset is not None),
        max_length=args.max_length,
        packing=False,
        dataset_text_field="text" if args.dataset_type not in ("knowmol", "pathway") else None,
        assistant_only_loss=args.assistant_only_loss,
    )

    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )


def build_hf_trainer(model, tokenizer, train_dataset, eval_dataset, args):
    """Construct a plain Transformers Trainer that avoids TRL dataset-prep barriers."""
    tokenized_train_dataset = tokenize_for_hf_trainer(
        train_dataset,
        tokenizer,
        args.max_length,
        "train",
        assistant_only_loss=args.assistant_only_loss,
    )
    tokenized_eval_dataset = None
    if eval_dataset is not None:
        tokenized_eval_dataset = tokenize_for_hf_trainer(
            eval_dataset,
            tokenizer,
            args.max_length,
            "eval",
            assistant_only_loss=args.assistant_only_loss,
        )

    training_args = TrainingArguments(
        **build_common_training_kwargs(args, tokenized_eval_dataset is not None),
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
    )

def train(args):
    """Main training function."""
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_name=args.model_name,
        chat_template_path=args.chat_template_path,
    )

    # Load dataset based on type
    if args.dataset_type == "chebi":
        train_dataset, eval_dataset = load_chebi_dataset(args.data_path, tokenizer)
    elif args.dataset_type == "reaction":
        train_dataset, eval_dataset = load_reaction_dataset(
            args.data_path, args.eval_path, tokenizer, args.prompt_path
        )
    elif args.dataset_type == "knowmol":
        train_dataset, eval_dataset = load_knowmol_dataset(args.data_path, tokenizer)
    elif args.dataset_type == "pathway":
        train_dataset, eval_dataset = load_pathway_dataset(
            args.data_path, args.eval_path, tokenizer
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    if args.trainer_backend == "trl":
        trainer = build_trl_trainer(model, tokenizer, train_dataset, eval_dataset, args)
    elif args.trainer_backend == "hf":
        trainer = build_hf_trainer(model, tokenizer, train_dataset, eval_dataset, args)
    else:
        raise ValueError(f"Unknown trainer backend: {args.trainer_backend}")

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Training complete!")
    return trainer


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified training script for Qwen3-30B-A3B"
    )

    # Dataset type
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["chebi", "reaction", "knowmol", "pathway"],
        help="Dataset type: 'chebi' for molecule captioning, 'reaction' for reaction prediction, 'knowmol' for multi-task molecular understanding, 'pathway' for reaction-wise pathway prediction",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="./checkpoints/Qwen3-30B-A3B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--chat_template_path",
        type=str,
        default=None,
        help="Optional path to a custom tokenizer chat_template Jinja file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the training dataset (directory for chebi, JSON file for reaction)",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default=None,
        help="Path to the evaluation dataset (JSON file, for reaction type only)",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="prompts.yaml",
        help="Path to prompts.yaml (for reaction type only)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/output",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--trainer_backend",
        type=str,
        default="trl",
        choices=["trl", "hf"],
        help="Training backend: 'trl' uses SFTTrainer, 'hf' uses plain Transformers Trainer",
    )

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type (cosine, constant_with_warmup, linear, etc.)")
    parser.add_argument("--use_fsdp_activation_checkpointing", action="store_true", help="Use FSDP activation checkpointing instead of gradient checkpointing (avoids redundant AllGather)")
    parser.add_argument("--use_liger_kernel", action="store_true", help="Use Liger fused cross-entropy to avoid materializing full logits tensor")
    parser.add_argument(
        "--assistant_only_loss",
        action="store_true",
        help="Mask prompt tokens and train only on assistant tokens for conversational datasets",
    )

    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch", "no"], help="Checkpoint save strategy")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint frequency (only used with --save_strategy steps)")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Max number of checkpoints to keep")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting backend (wandb, tensorboard, none)")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for logging")

    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set default run name based on dataset type
    if args.run_name is None:
        if args.dataset_type == "chebi":
            args.run_name = "qwen3-30b-chebi20-full-ft"
        elif args.dataset_type == "knowmol":
            args.run_name = "qwen3-30b-knowmol-full-ft"
        elif args.dataset_type == "pathway":
            args.run_name = "qwen3-30b-pathway-full-ft"
        else:
            args.run_name = "qwen3-30b-reaction-full-ft"

    # Calculate effective batch size
    effective_batch_size = 8 * args.batch_size * args.gradient_accumulation_steps

    logger.info("=" * 60)
    logger.info(f"Training - {args.dataset_type.upper()} Dataset")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Train data: {args.data_path}")
    if args.eval_path:
        logger.info(f"Eval data: {args.eval_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Precision: bf16")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Per-device batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size (8 GPUs): {effective_batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Max sequence length: {args.max_length}")
    logger.info("=" * 60)

    train(args)
