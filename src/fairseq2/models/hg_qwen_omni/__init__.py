# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
HuggingFace model integration for fairseq2.

This module provides seamless integration between HuggingFace Transformers
and fairseq2, allowing you to use any HuggingFace model within fairseq2's
training and inference pipelines.

Key Features:
    - Load any HuggingFace model with minimal configuration
    - Automatic device placement and dtype handling
    - Support for custom model architectures
    - Integration with fairseq2's model hub system
    - Compatible tokenizer interface
    - Comprehensive error handling

Quick Start:
    Load a GPT-2 model and tokenizer::

        from fairseq2.models.hg_qwen_omni import (
            load_hg_model_simple,
            load_hg_tokenizer_simple,
        )

        model = load_hg_model_simple("gpt2")
        tokenizer = load_hg_tokenizer_simple("gpt2")

    Use convenience functions for specific model types::

        from fairseq2.models.hg_qwen_omni import load_causal_lm, load_seq2seq_lm

        gpt_model = load_causal_lm("gpt2")
        t5_model = load_seq2seq_lm("t5-small")

    Register custom model classes::

        from fairseq2.models.hg_qwen_omni import register_hg_model_class

        register_hg_model_class(
            "Qwen2_5OmniConfig",
            "Qwen2_5OmniForConditionalGeneration",
            processor_class="Qwen2_5OmniProcessor",
        )

Modules:
    api: High-level API functions for model and tokenizer loading
    config: Configuration classes for HuggingFace models
    factory: Factory classes for model creation and loading
    hub: Integration with fairseq2's model hub system
    tokenizer: HuggingFace tokenizer integration

For detailed documentation, see: :doc:`/reference/fairseq2.models.hg_qwen_omni
"""

from __future__ import annotations

from fairseq2.models.hg_qwen_omni.api import load_causal_lm as load_causal_lm
from fairseq2.models.hg_qwen_omni.api import (
    load_hg_model_simple as load_hg_model_simple,
)
from fairseq2.models.hg_qwen_omni.api import (
    load_hg_tokenizer_simple as load_hg_tokenizer_simple,
)
from fairseq2.models.hg_qwen_omni.api import (
    load_multimodal_model as load_multimodal_model,
)
from fairseq2.models.hg_qwen_omni.api import load_seq2seq_lm as load_seq2seq_lm
from fairseq2.models.hg_qwen_omni.config import HG_FAMILY as HG_FAMILY
from fairseq2.models.hg_qwen_omni.config import (
    HuggingFaceModelConfig as HuggingFaceModelConfig,
)
from fairseq2.models.hg_qwen_omni.config import (
    register_hg_configs as register_hg_configs,
)
from fairseq2.models.hg_qwen_omni.factory import HgFactory as HgFactory
from fairseq2.models.hg_qwen_omni.factory import (
    HuggingFaceModelError as HuggingFaceModelError,
)
from fairseq2.models.hg_qwen_omni.factory import create_hg_model as create_hg_model
from fairseq2.models.hg_qwen_omni.factory import (
    register_hg_model_class as register_hg_model_class,
)
from fairseq2.models.hg_qwen_omni.fsdp import (
    apply_fsdp_to_hg_transformer_lm as apply_fsdp_to_hg_transformer_lm,
)
from fairseq2.models.hg_qwen_omni.hub import get_hg_model_hub as get_hg_model_hub
from fairseq2.models.hg_qwen_omni.hub import (
    get_hg_tokenizer_hub as get_hg_tokenizer_hub,
)
from fairseq2.models.hg_qwen_omni.tokenizer import HgTokenizer as HgTokenizer
from fairseq2.models.hg_qwen_omni.tokenizer import (
    HgTokenizerConfig as HgTokenizerConfig,
)
from fairseq2.models.hg_qwen_omni.tokenizer import (
    load_hg_tokenizer as load_hg_tokenizer,
)
