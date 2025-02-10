# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.llama._config import LLAMA_MODEL_FAMILY as LLAMA_MODEL_FAMILY
from fairseq2.models.llama._config import LLaMAConfig as LLaMAConfig
from fairseq2.models.llama._config import (
    LLaMARopeScalingConfig as LLaMARopeScalingConfig,
)
from fairseq2.models.llama._config import (
    register_llama_configs as register_llama_configs,
)
from fairseq2.models.llama._factory import LLaMAFactory as LLaMAFactory
from fairseq2.models.llama._factory import (
    init_llama_scaled_freqs as init_llama_scaled_freqs,
)
from fairseq2.models.llama._handler import LLaMAModelHandler as LLaMAModelHandler
from fairseq2.models.llama._handler import (
    convert_llama_checkpoint as convert_llama_checkpoint,
)

# isort: split

from fairseq2.models import ModelHubAccessor
from fairseq2.models.transformer_decoder import TransformerDecoderModel

get_llama_model_hub = ModelHubAccessor(TransformerDecoderModel, LLaMAConfig)
