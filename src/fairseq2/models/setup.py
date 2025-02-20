# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.models.jepa import register_jepa_family
from fairseq2.models.jepa.classifier import register_jepa_classifier_family
from fairseq2.models.llama import register_llama_family
from fairseq2.models.mistral import register_mistral_family
from fairseq2.models.nllb import register_nllb_configs
from fairseq2.models.s2t_transformer import register_s2t_transformer_family
from fairseq2.models.transformer import register_transformer_family
from fairseq2.models.w2vbert import register_w2vbert_family
from fairseq2.models.wav2vec2 import register_wav2vec2_family
from fairseq2.models.wav2vec2.asr import register_wav2vec2_asr_family


def register_model_families(context: RuntimeContext) -> None:
    register_jepa_classifier_family(context)
    register_jepa_family(context)
    register_llama_family(context)
    register_mistral_family(context)
    register_s2t_transformer_family(context)
    register_transformer_family(context)
    register_w2vbert_family(context)
    register_wav2vec2_asr_family(context)
    register_wav2vec2_family(context)

    register_nllb_configs(context)
