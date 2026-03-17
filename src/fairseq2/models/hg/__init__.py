# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HuggingFace model integration for fairseq2."""

from __future__ import annotations

from fairseq2.models.hg.adapter import (
    HgCausalLMAdapter as HgCausalLMAdapter,
    wrap_hg_model_if_causal_lm as wrap_hg_model_if_causal_lm,
)
from fairseq2.models.hg.api import (
    load_causal_lm as load_causal_lm,
    load_hg_model_simple as load_hg_model_simple,
    load_hg_tokenizer_simple as load_hg_tokenizer_simple,
    load_multimodal_model as load_multimodal_model,
    load_seq2seq_lm as load_seq2seq_lm,
)
from fairseq2.models.hg.config import (
    HG_FAMILY as HG_FAMILY,
    HuggingFaceModelConfig as HuggingFaceModelConfig,
    register_hg_configs as register_hg_configs,
)
from fairseq2.models.hg.factory import (
    HgFactory as HgFactory,
    HuggingFaceModelError as HuggingFaceModelError,
    create_hg_model as create_hg_model,
    register_hg_model_class as register_hg_model_class,
)
from fairseq2.models.hg.fsdp import (
    apply_fsdp_to_hg_transformer_lm as apply_fsdp_to_hg_transformer_lm,
)
from fairseq2.models.hg.hub import (
    get_hg_model_hub as get_hg_model_hub,
    get_hg_tokenizer_hub as get_hg_tokenizer_hub,
)
from fairseq2.models.hg.tokenizer import (
    HgTokenizer as HgTokenizer,
    HgTokenizerConfig as HgTokenizerConfig,
    load_hg_tokenizer as load_hg_tokenizer,
)

# Re-export converter API (formerly in fairseq2/models/hg.py)
from fairseq2.models.hg.converter import (
    HuggingFaceConfig as HuggingFaceConfig,
    HuggingFaceConverter as HuggingFaceConverter,
    _LegacyHuggingFaceConverter as _LegacyHuggingFaceConverter,
    get_hugging_face_converter as get_hugging_face_converter,
    save_hugging_face_model as save_hugging_face_model,
)
