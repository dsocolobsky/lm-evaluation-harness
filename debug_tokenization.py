#!/usr/bin/env python3

import os
import sys

# Run a simple CEval test to see tokenization behavior
def main():
    # Use a small number of samples for quick debugging
    cmd = [
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", "pretrained=deepseek-ai/DeepSeek-V2-Lite,trust_remote_code=True,dtype=float16",
        "--tasks", "ceval-valid_urban_and_rural_planner",
        "--num_fewshot", "0",
        "--limit", "10",
        "--device", "cuda:0",
        "--batch_size", "1"
    ]

    print("Running lm_eval with debug output...")
    print("Command:", " ".join(cmd))

    # Execute the command
    os.execvp("python", cmd)

if __name__ == "__main__":
    main()