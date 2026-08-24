"""Task modules for the thinking-gating experiment.

Each task module is self-contained (HuggingFace `datasets` + stdlib only) and
exposes the contract that scripts/capture_inference_thinking.py relies on:

    load_<task>(split) -> list[dict]   # rows with at least "question", "answer", "key"
    format_prompt(question) -> str     # raw (pre-chat-template) prompt text
    is_correct(generation, answer) -> bool
"""
