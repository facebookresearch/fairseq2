# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
High-level API for HuggingFace model integration.

This module provides the main entry points for loading HuggingFace models and
tokenizers within the fairseq2 framework. It offers simplified functions that
handle common use cases while providing flexibility for advanced
configurations.

Key Functions:
    - load_hg_model_simple: Load any HuggingFace model with minimal config
    - load_hg_tokenizer_simple: Load a HuggingFace tokenizer with custom tokens
    - load_causal_lm: Convenient wrapper for causal language models (GPT-style)
    - load_seq2seq_lm: Convenient wrapper for seq2seq models (T5-style)
    - load_multimodal_model: Convenient wrapper for multimodal models

Example:
    Basic usage for loading a GPT-2 model::

        from fairseq2.models.hg_qwen_omni import (
            load_hg_model_simple,
            load_hg_tokenizer_simple,
        )

        model = load_hg_model_simple("gpt2")
        tokenizer = load_hg_tokenizer_simple("gpt2")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fairseq2.models.hg_qwen_omni.config import HuggingFaceModelConfig
from fairseq2.models.hg_qwen_omni.factory import create_hg_model
from fairseq2.models.hg_qwen_omni.tokenizer import (
    HgTokenizer,
    HgTokenizerConfig,
    load_hg_tokenizer,
)


def load_hg_model_simple(
    name: str,
    *,
    model_type: str = "auto",
    use_processor: bool = False,
    device: str = "cpu",
    trust_remote_code: bool = False,
    dtype: str = "auto",
    **kwargs: Any,
) -> Any:
    """Load a HuggingFace model with simplified configuration.

    This is the main entry point for users who want to load HuggingFace models
    into fairseq2 with minimal configuration.

    :param name: HuggingFace model identifier (e.g., 'gpt2', 'microsoft/DialoGPT')
    :param model_type: Type of AutoModel to use ('auto', 'causal_lm', 'seq2seq_lm', 'custom')
    :param use_processor: Whether to use AutoProcessor instead of AutoTokenizer
    :param device: Device placement ('cpu', 'cuda:0', or 'auto' for HF accelerate)
    :param trust_remote_code: Whether to trust remote code for custom architectures
    :param dtype: PyTorch dtype to use ('auto', 'float16', 'bfloat16', etc.)
    :param kwargs: Additional kwargs passed to from_pretrained
    :returns: The loaded HuggingFace model

    Examples:
        Load a standard causal language model::

            model = load_hg_model_simple("gpt2")

        Load a seq2seq model::

            model = load_hg_model_simple("t5-small", model_type="seq2seq_lm")

        Load a multimodal model with processor::

            model = load_hg_model_simple(
                "Qwen/Qwen2.5-Omni-7B",
                use_processor=True,
                trust_remote_code=True
            )
    """
    config = HuggingFaceModelConfig(
        hf_name=name,
        model_type=model_type,
        use_processor=use_processor,
        device=device,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        load_kwargs=kwargs if kwargs else None,
    )

    return create_hg_model(config)


def load_hg_tokenizer_simple(
    name: str,
    *,
    unk_token: str | None = None,
    bos_token: str | None = None,
    eos_token: str | None = None,
    pad_token: str | None = None,
    boh_token: str | None = None,
    eoh_token: str | None = None,
) -> HgTokenizer:
    """Load a HuggingFace tokenizer with custom special tokens.

    :param name: HuggingFace tokenizer identifier (same as model name)
    :param unk_token: Custom unknown token
    :param bos_token: Custom beginning of sequence token
    :param eos_token: Custom end of sequence token
    :param pad_token: Custom padding token
    :param boh_token: Custom beginning of human token
    :param eoh_token: Custom end of human token
    :returns: The loaded tokenizer with custom tokens

    Examples:
        Load a tokenizer with default settings::

            tokenizer = load_hg_tokenizer_simple("gpt2")

        Load with custom tokens::

            tokenizer = load_hg_tokenizer_simple(
                "gpt2",
                pad_token="<pad>",
                eos_token="<end>"
            )
    """
    config = HgTokenizerConfig(
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        boh_token=boh_token,
        eoh_token=eoh_token,
    )
    return load_hg_tokenizer(Path(name), config)


# Convenience aliases for common use cases
def load_causal_lm(name: str, **kwargs: Any) -> Any:
    """Load a causal language model (GPT-style).

    Convenience function for loading causal language models like GPT-2,
    DialoGPT, or LLaMA.

    :param name: HuggingFace model identifier
    :param kwargs: Additional arguments passed to load_hg_model_simple
    :returns: A causal language model

    Example:
        Load GPT-2 for text generation::

            model = load_causal_lm("gpt2")
    """
    return load_hg_model_simple(name, model_type="causal_lm", **kwargs)


def load_seq2seq_lm(name: str, **kwargs: Any) -> Any:
    """Load a sequence-to-sequence model (T5-style).

    Convenience function for loading seq2seq models like T5, BART, or
    Pegasus for tasks like translation, summarization, and question
    answering.

    :param name: HuggingFace model identifier
    :param kwargs: Additional arguments passed to load_hg_model_simple
    :returns: A sequence-to-sequence model

    Example:
        Load T5 for translation::

            model = load_seq2seq_lm("t5-small")
    """
    return load_hg_model_simple(name, model_type="seq2seq_lm", **kwargs)


def load_multimodal_model(name: str, **kwargs: Any) -> Any:
    """Load a multimodal model with processor.

    Convenience function for loading multimodal models that require
    processors instead of tokenizers (e.g., vision-language models).

    :param name: HuggingFace model identifier
    :param kwargs: Additional arguments passed to load_hg_model_simple
    :returns: A multimodal model

    Example:
        Load a multimodal model::

            model = load_multimodal_model(
                "Qwen/Qwen2.5-Omni-3B",
                trust_remote_code=True
            )
    """
    return load_hg_model_simple(name, use_processor=True, **kwargs)
