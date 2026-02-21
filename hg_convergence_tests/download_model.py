#!/usr/bin/env python3
"""Pre-download model to avoid distributed loading issues."""

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Pre-download model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-3-1b-it",
        help="Model name to download",
    )
    args = parser.parse_args()

    print(f"Downloading model {args.model_name}...")

    # Download model and tokenizer to cache
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    print(f"Model {args.model_name} downloaded successfully to cache")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
