# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.llama4._checkpoint import (
    convert_llama4_checkpoint as convert_llama4_checkpoint,
)
from fairseq2.models.llama4._config import LLAMA4_MODEL_FAMILY as LLAMA4_MODEL_FAMILY
from fairseq2.models.llama4._config import LLaMA4DecoderConfig as LLaMA4DecoderConfig
from fairseq2.models.llama._config import (
    LLaMARopeScalingConfig as LLaMARopeScalingConfig,
)
from fairseq2.models.llama4._config import (
    register_llama4_configs as register_llama4_configs,
)
from fairseq2.models.llama4._factory import LLaMA4Factory as LLaMA4Factory
from fairseq2.models.llama4._factory import create_llama4_model as create_llama4_model
from fairseq2.models.llama4._shard import shard_llama4_model as shard_llama4_model

# isort: split

from fairseq2.models import ModelHubAccessor
from fairseq2.models.transformer_decoder import TransformerDecoderModel

get_llama4_model_hub = ModelHubAccessor(TransformerDecoderModel, LLaMA4DecoderConfig)
