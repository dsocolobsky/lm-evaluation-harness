#!/usr/bin/env python3
"""
Debug script to extract and save token numbers from LM evaluation harness.
This script patches the construct_requests method to capture context tokens
and writes both the text and token numbers to tokens_lmeval.out file.
"""

import os
import sys

# Add the current directory to sys.path to import lm_eval modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after adding to path
from lm_eval.api.task import ConfigurableTask
from lm_eval.__main__ import cli_evaluate

# Global variable to store the model reference
_debug_model = None

def set_debug_model(model):
    """Set the global model reference for tokenization."""
    global _debug_model
    _debug_model = model

def debug_log_tokens(text, task_name="unknown"):
    """Log text and its tokens to the debug file."""
    global _debug_model
    if _debug_model is not None and hasattr(_debug_model, 'tok_encode'):
        try:
            tokens = _debug_model.tok_encode(text)
            # Write both text and tokens to file
            with open('tokens_lmeval.out', 'a', encoding='utf-8') as f:
                f.write(f"# Task: {task_name}\n")
                f.write(f"# Text: {repr(text)}\n")
                f.write(f"{tokens}\n\n")
        except Exception as e:
            print(f"Debug tokenization failed: {e}")

# Store the original construct_requests method
_original_construct_requests = ConfigurableTask.construct_requests

def patched_construct_requests(self, doc, ctx, **kwargs):
    """Patched version of construct_requests that captures context tokens."""
    # Log the context tokens
    if ctx:
        debug_log_tokens(ctx, getattr(self.config, 'task', 'unknown'))

    # For multiple choice tasks, also log the individual choice contexts
    if self.OUTPUT_TYPE == "multiple_choice":
        apply_chat_template = kwargs.get("apply_chat_template", False)
        chat_template = kwargs.get("chat_template", None)
        choices = self.doc_to_choice(doc)
        target_delimiter = self.config.target_delimiter
        if apply_chat_template:
            target_delimiter = ""

        if hasattr(self, 'multiple_input') and self.multiple_input:
            # Log each choice context for multiple input tasks
            for i, choice in enumerate(choices):
                choice_ctx = ctx + (
                    chat_template([{"role": "user", "content": choice}])
                    if apply_chat_template and chat_template
                    else choice
                )
                debug_log_tokens(choice_ctx, f"{getattr(self.config, 'task', 'unknown')}_choice_{i}")

    # Call the original method
    return _original_construct_requests(self, doc, ctx, **kwargs)

# Patch the method
ConfigurableTask.construct_requests = patched_construct_requests

# Patch the evaluator to capture the model reference
from lm_eval import evaluator
_original_evaluate = evaluator.evaluate

def patched_evaluate(lm, task_dict, **kwargs):
    """Patched version of evaluate that captures the model reference."""
    set_debug_model(lm)
    return _original_evaluate(lm, task_dict, **kwargs)

evaluator.evaluate = patched_evaluate

# Clear the output file
with open('tokens_lmeval.out', 'w') as f:
    f.write("# LM Evaluation Harness Token Debug Output\n")
    f.write("# Format: Text followed by token list\n\n")

if __name__ == "__main__":
    print("Clean token capture patch applied!")
    print("Running lm_eval with clean token capture...")
    print("Tokens and text will be saved to tokens_lmeval.out")
    cli_evaluate()