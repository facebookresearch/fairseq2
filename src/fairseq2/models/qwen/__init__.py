# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.qwen._checkpoint import (
    convert_qwen_checkpoint as convert_qwen_checkpoint,
    convert_qwen_fs2_to_hf_checkpoint as convert_qwen_fs2_to_hf_checkpoint,
)
from fairseq2.models.qwen._config import QWEN_MODEL_FAMILY as QWEN_MODEL_FAMILY
from fairseq2.models.qwen._config import QwenConfig as QwenConfig
from fairseq2.models.qwen._config import (
    register_qwen_configs as register_qwen_configs,
)
from fairseq2.models.qwen._factory import QwenFactory as QwenFactory
from fairseq2.models.qwen._factory import create_qwen_model as create_qwen_model
from fairseq2.models.qwen._shard import shard_qwen_model as shard_qwen_model

# isort: split

from fairseq2.models import ModelHubAccessor
from fairseq2.models.transformer_decoder import TransformerDecoderModel

get_llama_model_hub = ModelHubAccessor(TransformerDecoderModel, QwenConfig)
