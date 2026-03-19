# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration classes for HuggingFace model integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from fairseq2.data_type import DataType
from fairseq2.runtime.config_registry import ConfigRegistrar
from fairseq2.runtime.dependency import DependencyContainer

HG_FAMILY: Final = "hg"


@dataclass(kw_only=True)
class HuggingFaceModelConfig:
    """Configuration for loading HuggingFace models.

    This dataclass contains all the parameters needed to configure how a
    HuggingFace model should be loaded, including device placement, dtype,
    custom classes, and special loading options.

    :param hf_name: The HuggingFace model identifier (e.g., 'gpt2')
    :param model_type: Type of AutoModel ('auto', 'causal_lm', 'seq2seq_lm',
        'custom')
    :param use_processor: Whether to use AutoProcessor for multimodal models
    :param device: Device placement ('cpu', 'cuda:0', or 'auto')
    :param custom_model_class: Custom model class name for special cases
    :param custom_processor_class: Custom processor class name for special
        cases
    :param trust_remote_code: Whether to trust remote code for custom
        architectures
    :param dtype: PyTorch dtype to use. None means 'auto' (let HuggingFace decide)
    :param load_kwargs: Additional kwargs to pass to from_pretrained
    :param enable_gradient_checkpointing: Whether to enable gradient checkpointing
        to reduce memory usage during training (only for causal_lm models)

    Example:
        Create a configuration for GPT-2::

            config = HuggingFaceModelConfig(
                hf_name="gpt2",
                model_type="causal_lm",
                device="cuda:0"
            )
    """

    hf_name: str
    """The HuggingFace model identifier (e.g., 'gpt2')."""

    model_type: str = "auto"
    """Type of AutoModel ('auto', 'causal_lm', 'seq2seq_lm', 'custom')."""

    use_processor: bool = False
    """Whether to use AutoProcessor for multimodal models."""

    device: str = "cpu"
    """Device placement: 'cpu', 'cuda:0', or 'auto' for HF accelerate."""

    custom_model_class: str | None = None
    """Custom model class name for special cases."""

    custom_processor_class: str | None = None
    """Custom processor class name for special cases."""

    trust_remote_code: bool = False
    """Whether to trust remote code for custom architectures."""

    dtype: DataType | None = None
    """PyTorch dtype to use. None means 'auto' (let HuggingFace decide)."""

    load_kwargs: dict[str, Any] | None = None
    """Additional kwargs to pass to from_pretrained."""

    enable_gradient_checkpointing: bool = False
    """Whether to enable gradient checkpointing to reduce memory usage (causal_lm only)."""


def register_hg_configs(container: DependencyContainer) -> None:
    """Register predefined HuggingFace model configurations.

    :param container: The dependency container to register configurations with.
    """
    arch = ConfigRegistrar(container, HuggingFaceModelConfig)

    @arch("causal_lm")
    def causal_lm() -> HuggingFaceModelConfig:
        return HuggingFaceModelConfig(
            hf_name="gpt2",
            model_type="causal_lm",
        )

    @arch("seq2seq_lm")
    def seq2seq_lm() -> HuggingFaceModelConfig:
        return HuggingFaceModelConfig(
            hf_name="google-t5/t5-small",
            model_type="seq2seq_lm",
        )

    @arch("auto")
    def auto() -> HuggingFaceModelConfig:
        return HuggingFaceModelConfig(
            hf_name="bert-base-uncased",
            model_type="auto",
        )

    @arch("custom")
    def custom() -> HuggingFaceModelConfig:
        return HuggingFaceModelConfig(
            hf_name="Qwen/Qwen2.5-Omni-3B",
            model_type="custom",
        )
