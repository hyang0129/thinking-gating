"""
capture_inference_thinking.py — Thinking-mode gating experiment data capture.

What it does:
1. Generate paired inference: thinking-off + thinking-on for each query
2. Extract only the last-prompt-token (prefill) hidden state
3. Store correctness labels for both runs to enable downstream label generation

Usage:
    python scripts/capture_inference_thinking.py \\
        --task gsm8k \\
        --model Qwen/Qwen3-8B \\
        --out-dir shared/icr_capture/gsm8k_thinking_qwen3 \\
        --max-samples 500 \\
        --chat-template

Output format:
  shared/icr_capture/gsm8k_thinking_qwen3/
  ├── config.json                    # capture params
  ├── meta.jsonl                     # per-query metadata (prompt_hash, generated text, correctness for both thinking modes)
  ├── activations_thinking_off.npz   # (N, num_layers, hidden_dim) prefill hidden states
  ├── activations_thinking_on.npz    # (N, num_layers, hidden_dim) prefill hidden states
  └── eval_results.json              # correctness metrics (generated post-capture)

Task module contract (see tasks/__init__.py):
  - load_<task>(split) -> list[dict] with keys "question", "answer", "key", "difficulty"
  - format_prompt(question) -> str
  - is_correct(generated_text, answer) -> bool
  - Supported: gsm8k (primary), lsat (transfer)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

def _correct_str(task_module: Any, generation: str, sample: dict) -> bool:
    """Adapter for tasks where is_correct(generation, answer: str)."""
    return task_module.is_correct(generation, sample["answer"])


def _prompt_default(task_module: Any, sample: dict) -> str:
    """Use task_module.format_prompt if available, else sample['question']."""
    if hasattr(task_module, "format_prompt"):
        return task_module.format_prompt(sample["question"])
    return sample["question"]


# task name -> (loader function name, correctness adapter, default split).
# Every entry must have a matching module in tasks/ — this repo ships its own
# task modules rather than importing them from anywhere else.
_TASK_REGISTRY: dict[str, tuple[str, Any, str]] = {
    "gsm8k": ("load_gsm8k", _correct_str, "test"),
    "lsat": ("load_lsat_logic", _correct_str, "test"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def apply_chat_template_to_prompt(
    raw_prompt: str, tokenizer: Any, model_name: str, enable_thinking: bool = False
) -> str:
    """Wrap raw_prompt as a single user message and render via chat template.

    Args:
        raw_prompt: The task prompt string.
        tokenizer: HuggingFace tokenizer with apply_chat_template method.
        model_name: Model identifier (used to detect qwen/smollm).
        enable_thinking: If True, pass enable_thinking=True for thinking-mode rendering.
    """
    messages = [{"role": "user", "content": raw_prompt}]
    template_kwargs: dict[str, Any] = dict(add_generation_prompt=True, tokenize=False)

    model_name_lower = model_name.lower()
    if "qwen3" in model_name_lower or "smollm3" in model_name_lower:
        template_kwargs["enable_thinking"] = enable_thinking

    return tokenizer.apply_chat_template(messages, **template_kwargs)


def build_prompt(
    sample: dict,
    task_module: Any,
    tokenizer: Any = None,
    model_name: str | None = None,
    chat_template: bool = False,
    enable_thinking: bool = False,
) -> str:
    """Build the prompt string for a sample."""
    raw_prompt = _prompt_default(task_module, sample)
    if not chat_template:
        return raw_prompt
    assert tokenizer is not None and model_name is not None
    return apply_chat_template_to_prompt(
        raw_prompt, tokenizer, model_name, enable_thinking=enable_thinking
    )


def load_model_eager(model_name: str):
    """Load model with eager attention (required for output_attentions)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Loading model (eager, fp16): %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def normalize_eos_ids(generation_config_eos: Any, tokenizer_eos_id: Any) -> set[int]:
    """Normalize eos_token_id (int | list | None) to a set of ints."""
    eos_ids: set[int] = set()
    if generation_config_eos is None:
        pass
    elif isinstance(generation_config_eos, (list, tuple, set)):
        eos_ids.update(int(x) for x in generation_config_eos)
    else:
        eos_ids.add(int(generation_config_eos))
    if tokenizer_eos_id is not None:
        eos_ids.add(int(tokenizer_eos_id))
    return eos_ids


def response_len_from_ids(resp_ids: np.ndarray, eos_ids: set[int]) -> int:
    """Length up to and including the first EOS token, else full length."""
    resp_ids = np.asarray(resp_ids)
    hits = np.nonzero(np.isin(resp_ids, list(eos_ids)))[0]
    if hits.size > 0:
        return int(hits[0]) + 1
    return int(resp_ids.shape[0])


def extract_prefill_hidden_states_batched(
    hidden_states_tuple: tuple, prompt_lens: np.ndarray
) -> np.ndarray:
    """Extract last-prompt-token hidden state for each layer.

    Args:
        hidden_states_tuple: From model.generate(..., output_hidden_states=True).
                            Tuple of (num_layers+1, B, T, hidden_dim) tensors.
        prompt_lens: (B,) array of actual prompt lengths (before padding).

    Returns:
        (B, num_layers, hidden_dim) array of prefill hidden states at token position
        prompt_len-1 (the last prompt token, 0-indexed).
    """
    import torch

    B = prompt_lens.shape[0]
    num_layers = len(hidden_states_tuple)
    hidden_dim = hidden_states_tuple[0].shape[-1]

    prefill_hs = np.zeros((B, num_layers, hidden_dim), dtype=np.float16)

    for b in range(B):
        prompt_len = int(prompt_lens[b])
        # Last prompt token is at position (prompt_len - 1)
        last_prompt_pos = prompt_len - 1

        for layer in range(num_layers):
            # hidden_states_tuple[layer] is (B, T, hidden_dim)
            hs = hidden_states_tuple[layer][b, last_prompt_pos, :].cpu().numpy().astype(np.float16)
            prefill_hs[b, layer, :] = hs

    return prefill_hs


def build_writer_config(model: Any, args: argparse.Namespace) -> dict:
    """Assemble config.json dict from model and CLI args."""
    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = getattr(cfg, "n_layer", None)
    hidden_dim = getattr(cfg, "hidden_size", None)

    return {
        "model_name": args.model,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "chat_template": bool(args.chat_template),
        "max_prompt_len": args.max_prompt_len,
        "max_response_len": args.max_response_len,
    }


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------

def run_capture(args: argparse.Namespace):
    """Main capture routine: thinking-off + thinking-on pairs."""
    import torch
    from pathlib import Path

    # Setup
    tokenizer, model = load_model_eager(args.model)
    task_name, (load_fn_name, is_correct_adapter, default_split) = (
        args.task,
        _TASK_REGISTRY[args.task],
    )

    # Load task module and dataset
    try:
        task_module = importlib.import_module(f"tasks.{task_name}")
    except ImportError:
        logger.error(f"Task module not found: tasks.{task_name}")
        raise

    load_fn = getattr(task_module, load_fn_name)
    dataset = load_fn(args.split)
    logger.info(f"Loaded {task_name} {args.split} split: {len(dataset)} examples")

    # Output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize output file
    config_path = out_dir / "config.json"
    meta_path = out_dir / "meta.jsonl"
    activations_off_path = out_dir / "activations_thinking_off.npz"
    activations_on_path = out_dir / "activations_thinking_on.npz"

    writer_mode = "a" if config_path.exists() else "w"
    logger.info(f"Output mode: {writer_mode}")

    if writer_mode == "w":
        config = build_writer_config(model, args)
        config_path.write_text(json.dumps(config, indent=2))
        logger.info(f"Wrote config to {config_path}")

    # Process samples
    max_samples = args.max_samples if args.max_samples > 0 else len(dataset)
    dataset_subset = dataset[:max_samples]
    logger.info(f"Processing {len(dataset_subset)} samples")

    activations_off_list = []
    activations_on_list = []
    meta_rows = []

    eos_ids = normalize_eos_ids(
        getattr(model.generation_config, "eos_token_id", None), tokenizer.eos_token_id
    )

    for idx, sample in enumerate(dataset_subset):
        if (idx + 1) % max(1, len(dataset_subset) // 10) == 0:
            logger.info(f"Progress: {idx + 1} / {len(dataset_subset)}")

        # Build prompts (thinking-off and thinking-on)
        prompt_off = build_prompt(
            sample, task_module, tokenizer, args.model,
            chat_template=args.chat_template, enable_thinking=False
        )
        prompt_on = build_prompt(
            sample, task_module, tokenizer, args.model,
            chat_template=args.chat_template, enable_thinking=True
        )

        prompt_hash = sha256(prompt_off)  # Use thinking-off prompt as canonical

        # Tokenize
        tokens_off = tokenizer(
            [prompt_off], padding=False, truncation=True,
            max_length=args.max_prompt_len, return_tensors="pt",
            add_special_tokens=not args.chat_template
        ).to(model.device)

        tokens_on = tokenizer(
            [prompt_on], padding=False, truncation=True,
            max_length=args.max_prompt_len, return_tensors="pt",
            add_special_tokens=not args.chat_template
        ).to(model.device)

        prompt_len_off = int(tokens_off.attention_mask.sum(dim=1).cpu().numpy()[0])
        prompt_len_on = int(tokens_on.attention_mask.sum(dim=1).cpu().numpy()[0])

        # Generate thinking-off
        with torch.no_grad():
            out_off = model.generate(
                tokens_off.input_ids,
                attention_mask=tokens_off.attention_mask,
                max_new_tokens=args.max_response_len,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Generate thinking-on
        with torch.no_grad():
            out_on = model.generate(
                tokens_on.input_ids,
                attention_mask=tokens_on.attention_mask,
                max_new_tokens=args.max_response_len,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Extract responses
        response_off = tokenizer.decode(
            out_off.sequences[0, prompt_len_off:], skip_special_tokens=True
        )
        response_on = tokenizer.decode(
            out_on.sequences[0, prompt_len_on:], skip_special_tokens=True
        )

        # Compute correctness
        correct_off = is_correct_adapter(task_module, response_off, sample)
        correct_on = is_correct_adapter(task_module, response_on, sample)

        # Extract prefill hidden states (last prompt token)
        prefill_hs_off = extract_prefill_hidden_states_batched(
            out_off.hidden_states, np.array([prompt_len_off])
        )[0]  # (num_layers, hidden_dim)

        prefill_hs_on = extract_prefill_hidden_states_batched(
            out_on.hidden_states, np.array([prompt_len_on])
        )[0]  # (num_layers, hidden_dim)

        activations_off_list.append(prefill_hs_off)
        activations_on_list.append(prefill_hs_on)

        # Metadata
        meta_row = {
            "prompt_hash": prompt_hash,
            "prompt_off": prompt_off,
            "prompt_on": prompt_on,
            "response_off": response_off,
            "response_on": response_on,
            "correct_off": correct_off,
            "correct_on": correct_on,
            "sample_id": sample.get("key", str(idx)),
            "difficulty": sample.get("difficulty"),
        }
        meta_rows.append(meta_row)

    # Stack activations
    activations_off = np.stack(activations_off_list, axis=0)  # (N, num_layers, hidden_dim)
    activations_on = np.stack(activations_on_list, axis=0)

    logger.info(f"Activations OFF shape: {activations_off.shape}")
    logger.info(f"Activations ON shape: {activations_on.shape}")

    # Write outputs
    np.savez_compressed(activations_off_path, activations_off)
    np.savez_compressed(activations_on_path, activations_on)
    logger.info(f"Wrote activations to {activations_off_path} and {activations_on_path}")

    with open(meta_path, "a") as f:
        for row in meta_rows:
            f.write(json.dumps(row) + "\n")
    logger.info(f"Wrote {len(meta_rows)} metadata rows to {meta_path}")

    # Summary
    correct_counts = sum(1 for row in meta_rows if row["correct_off"])
    correct_on_counts = sum(1 for row in meta_rows if row["correct_on"])
    logger.info(f"Thinking-OFF accuracy: {correct_counts}/{len(meta_rows)} ({100*correct_counts/len(meta_rows):.1f}%)")
    logger.info(f"Thinking-ON accuracy: {correct_on_counts}/{len(meta_rows)} ({100*correct_on_counts/len(meta_rows):.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Capture thinking-mode paired inference for hallucination gating."
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=list(_TASK_REGISTRY.keys()),
        help="Task name (gsm8k, lsat)."
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="Dataset split (train/test/validation). Default: task-specific default."
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-8B",
        help="HuggingFace model ID."
    )
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Output directory for icr_capture format."
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Max samples to process (0=all). Useful for smoke tests."
    )
    parser.add_argument(
        "--max-prompt-len", type=int, default=512,
        help="Max prompt token length."
    )
    parser.add_argument(
        "--max-response-len", type=int, default=128,
        help="Max response token length."
    )
    parser.add_argument(
        "--chat-template", action="store_true",
        help="Apply chat template via tokenizer.apply_chat_template()."
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level."
    )

    args = parser.parse_args()

    # Set split to task default if not specified
    if args.split is None:
        _, (_, _, args.split) = args.task, _TASK_REGISTRY[args.task]

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    run_capture(args)


if __name__ == "__main__":
    main()
