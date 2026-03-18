# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HuggingFace model integration for fairseq2."""

from __future__ import annotations

from fairseq2.models.hg.adapter import HgCausalLMAdapter as HgCausalLMAdapter
from fairseq2.models.hg.adapter import (
    wrap_hg_model_if_causal_lm as wrap_hg_model_if_causal_lm,
)
from fairseq2.models.hg.api import load_causal_lm as load_causal_lm
from fairseq2.models.hg.api import load_hg_model_simple as load_hg_model_simple
from fairseq2.models.hg.api import load_hg_tokenizer_simple as load_hg_tokenizer_simple
from fairseq2.models.hg.api import load_multimodal_model as load_multimodal_model
from fairseq2.models.hg.api import load_seq2seq_lm as load_seq2seq_lm
from fairseq2.models.hg.config import HG_FAMILY as HG_FAMILY
from fairseq2.models.hg.config import HuggingFaceModelConfig as HuggingFaceModelConfig
from fairseq2.models.hg.config import register_hg_configs as register_hg_configs
from fairseq2.models.hg.converter import HuggingFaceConfig as HuggingFaceConfig
from fairseq2.models.hg.converter import HuggingFaceConverter as HuggingFaceConverter
from fairseq2.models.hg.converter import (
    _LegacyHuggingFaceConverter as _LegacyHuggingFaceConverter,
)
from fairseq2.models.hg.converter import (
    get_hugging_face_converter as get_hugging_face_converter,
)
from fairseq2.models.hg.converter import (
    save_hugging_face_model as save_hugging_face_model,
)
from fairseq2.models.hg.factory import HgFactory as HgFactory
from fairseq2.models.hg.factory import HuggingFaceModelError as HuggingFaceModelError
from fairseq2.models.hg.factory import create_hg_model as create_hg_model
from fairseq2.models.hg.factory import (
    register_hg_model_class as register_hg_model_class,
)
from fairseq2.models.hg.fsdp import (
    apply_fsdp_to_hg_transformer_lm as apply_fsdp_to_hg_transformer_lm,
)
from fairseq2.models.hg.hub import get_hg_model_hub as get_hg_model_hub
from fairseq2.models.hg.hub import get_hg_tokenizer_hub as get_hg_tokenizer_hub
from fairseq2.models.hg.tokenizer import HgTokenizer as HgTokenizer
from fairseq2.models.hg.tokenizer import HgTokenizerConfig as HgTokenizerConfig
from fairseq2.models.hg.tokenizer import load_hg_tokenizer as load_hg_tokenizer
