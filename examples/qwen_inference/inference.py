#!/usr/bin/env python3
"""
Simple Inference Script for Qwen 2.5 7B Instruct Model

This script demonstrates how to perform text generation using fairseq2's
Qwen 2.5 7B Instruct model loaded from the asset system.

Usage:
    # CPU mode
    python inference.py --config config.yaml

    # Single GPU mode with custom config
    python inference.py --config config.yaml --device cuda:0

    # Multi-GPU mode (using gang abstraction)
    torchrun --nproc_per_node=2 inference.py --config config.yaml --device cuda

Features:
    - Loads Qwen 2.5 7B model from fairseq2 assets
    - Loads corresponding tokenizer from fairseq2 assets
    - Uses fairseq2's TextCompleter for text generation
    - Supports CPU, single-GPU, and multi-GPU execution via gang
    - Demonstrates sampling-based text generation
"""

import argparse
from pathlib import Path
from typing import Any, Final

import yaml

from fairseq2.device import Device
from fairseq2.gang import Gangs, get_default_gangs

from generator import create_text_completer
from model import load_qwen_model, load_qwen_tokenizer


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_inference(
    config: dict[str, Any],
    device_override: str | None = None,
) -> None:
    """
    Run inference with the given configuration.

    Args:
        config: Configuration dictionary loaded from YAML
        device_override: Optional device override from command line
    """
    # Setup device
    device_type = device_override or config["device"]["type"]
    device: Final[Device] = Device(device_type)

    # Setup gangs for distributed execution
    gangs: Final[Gangs] = get_default_gangs()

    # Load model and tokenizer
    model = load_qwen_model(
        model_name=config["model"]["name"],
        device=device,
        gangs=gangs,
    )

    tokenizer = load_qwen_tokenizer(
        model_name=config["model"]["name"],
        gangs=gangs,
    )

    # Create text completer
    text_completer = create_text_completer(
        model=model,
        tokenizer=tokenizer,
        gangs=gangs,
        max_gen_len=config["generation"]["max_gen_len"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
        echo_prompt=config["generation"]["echo_prompt"],
        skip_special_tokens=config["generation"]["skip_special_tokens"],
    )

    # Run inference on prompts
    prompts = config["prompts"]

    if gangs.root.rank == 0:
        print("\nRunning inference on example prompts...\n")

        for i, prompt in enumerate(prompts, 1):
            print(f"Prompt {i}: {prompt}")
            print("-" * 60)

            # Generate completion for the prompt
            completed_text, generator_output = text_completer(prompt)

            print(f"Generated: {completed_text}")
            print("-" * 60)
            print()

        print("Inference complete!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: cpu, cuda, or cuda:N",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Run inference
    run_inference(config, device_override=args.device)


if __name__ == "__main__":
    main()
