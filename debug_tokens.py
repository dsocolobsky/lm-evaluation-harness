#!/usr/bin/env python3
"""
Debug script to extract and save token numbers from LM evaluation harness.
This script patches the HFLM.tok_encode method to capture all tokenization calls
and writes the token numbers to tokens_lmeval.out file in list format.

Run this before running your lm_eval command.
"""

import os
import sys

# Add the current directory to sys.path to import lm_eval modules
sys.path.insert(0, '/home/admin/lm-evaluation-harness')

# Import after adding to path
from lm_eval.models.huggingface import HFLM

# Store the original tok_encode method
_original_tok_encode = HFLM.tok_encode

def patched_tok_encode(self, string, left_truncate_len=None, add_special_tokens=None):
    """Patched version of tok_encode that captures tokens and saves them."""
    # Call the original method to get tokens
    tokens = _original_tok_encode(self, string, left_truncate_len, add_special_tokens)

    # Write tokens to file as a list on a single line (append mode)
    if tokens:  # Only write non-empty token lists
        with open('/home/admin/lm-evaluation-harness/tokens_lmeval.out', 'a') as f:
            f.write(f"{tokens}\n")

    return tokens

# Patch the method globally
HFLM.tok_encode = patched_tok_encode

# Clear the output file
with open('/home/admin/lm-evaluation-harness/tokens_lmeval.out', 'w') as f:
    f.write("")  # Clear file

# Import and run the command line interface
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    print("Token capture patch applied!")
    print("Running lm_eval with token capture...")
    print("Tokens will be saved to tokens_lmeval.out in format: [token1, token2, token3]")
    cli_evaluate()