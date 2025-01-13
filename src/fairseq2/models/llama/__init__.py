# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.llama.config import LLAMA_MODEL_FAMILY as LLAMA_MODEL_FAMILY
from fairseq2.models.llama.config import LLaMAConfig as LLaMAConfig
from fairseq2.models.llama.config import (
    LLaMARopeScalingConfig as LLaMARopeScalingConfig,
)
from fairseq2.models.llama.config import (
    register_llama_configs as register_llama_configs,
)
from fairseq2.models.llama.factory import LLaMAFactory as LLaMAFactory
from fairseq2.models.llama.handler import LLaMAModelHandler as LLaMAModelHandler
from fairseq2.models.llama.handler import (
    convert_llama_checkpoint as convert_llama_checkpoint,
)
from fairseq2.models.llama.lora import get_llama_lora_config as get_llama_lora_config

# isort: split

from fairseq2.models.hub import ModelHubAccessor
from fairseq2.models.transformer_decoder import TransformerDecoderModel

get_llama_model_hub = ModelHubAccessor(TransformerDecoderModel, LLaMAConfig)
