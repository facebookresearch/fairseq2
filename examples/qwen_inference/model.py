"""Model and tokenizer loading for Qwen inference."""

from typing import Any

import torch
from torch.nn import Module

from fairseq2.data.tokenizers import Tokenizer, load_tokenizer
from fairseq2.device import Device
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.models import load_model


def load_qwen_model(
    model_name: str,
    device: Device,
    gangs: Gangs,
) -> Module:
    """
    Load Qwen model from fairseq2 assets.

    Args:
        model_name: Model name from fairseq2 assets
        device: Target device for the model
        gangs: Gang abstraction for distributed execution

    Returns:
        Loaded model in evaluation mode
    """
    if gangs.root.rank == 0:
        log.info("Loading model '{}' from fairseq2 assets...", model_name)
        log.info("Device: {}", device)
        log.info("Gang size: {}", gangs.root.size)

    # Load model from fairseq2 assets
    model = load_model(model_name, device=device, dtype=torch.bfloat16)
    model.eval()

    if gangs.root.rank == 0:
        log.info("Model loaded successfully!")
        log.info("Model type: {}", type(model).__name__)

    return model


def load_qwen_tokenizer(
    model_name: str,
    gangs: Gangs,
) -> Tokenizer:
    """
    Load tokenizer for Qwen model.

    Args:
        model_name: Model name from fairseq2 assets
        gangs: Gang abstraction for distributed execution

    Returns:
        Loaded tokenizer
    """
    tokenizer = load_tokenizer(model_name)

    if gangs.root.rank == 0:
        log.info("Tokenizer loaded successfully!")
        log.info("Vocabulary size: {}", tokenizer.vocab_info.size)

    return tokenizer
