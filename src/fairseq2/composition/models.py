# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import register_model_family
from fairseq2.models.jepa import (
    JEPA_FAMILY,
    JepaConfig,
    JepaModel,
    convert_jepa_state_dict,
    create_jepa_model,
    register_jepa_configs,
)
from fairseq2.models.jepa.classifier import (
    JEPA_CLASSIFIER_FAMILY,
    JepaClassifierConfig,
    JepaClassifierModel,
    create_jepa_classifier_model,
    register_jepa_classifier_configs,
)
from fairseq2.models.llama import (
    LLAMA_FAMILY,
    LLaMAConfig,
    convert_llama_state_dict,
    create_llama_model,
    get_llama_shard_specs,
    register_llama_configs,
    save_as_hg_llama,
)
from fairseq2.models.mistral import (
    MISTRAL_FAMILY,
    MistralConfig,
    convert_mistral_state_dict,
    create_mistral_model,
    register_mistral_configs,
)
from fairseq2.models.nllb import register_nllb_configs
from fairseq2.models.opt import (
    OPT_MODEL_FAMILY,
    OPTConfig,
    convert_opt_state_dict,
    create_opt_model,
    register_opt_configs,
)
from fairseq2.models.qwen import (
    QWEN_FAMILY,
    QwenConfig,
    convert_qwen_state_dict,
    create_qwen_model,
    get_qwen_shard_specs,
    register_qwen_configs,
    save_as_hg_qwen,
)
from fairseq2.models.s2t_transformer import (
    S2T_TRANSFORMER_FAMILY,
    S2TTransformerConfig,
    convert_s2t_transformer_state_dict,
    create_s2t_transformer_model,
    register_s2t_transformer_configs,
)
from fairseq2.models.transformer import (
    TRANSFORMER_FAMILY,
    TransformerConfig,
    TransformerModel,
    convert_transformer_state_dict,
    create_transformer_model,
    register_transformer_configs,
)
from fairseq2.models.transformer_lm import TransformerLM, compile_transformer_lm
from fairseq2.models.w2vbert import (
    W2VBERT_FAMILY,
    W2VBertConfig,
    W2VBertModel,
    convert_w2vbert_state_dict,
    create_w2vbert_model,
    register_w2vbert_configs,
)
from fairseq2.models.wav2vec2 import (
    WAV2VEC2_FAMILY,
    Wav2Vec2Config,
    Wav2Vec2Model,
    convert_wav2vec2_state_dict,
    create_wav2vec2_model,
    register_wav2vec2_configs,
)
from fairseq2.models.wav2vec2.asr import (
    WAV2VEC2_ASR_FAMILY,
    Wav2Vec2AsrConfig,
    Wav2Vec2AsrModel,
    convert_wav2vec2_asr_state_dict,
    create_wav2vec2_asr_model,
    register_wav2vec2_asr_configs,
)
from fairseq2.runtime.dependency import DependencyContainer


def _register_model_families(container: DependencyContainer) -> None:
    # JEPA
    register_model_family(
        container,
        JEPA_FAMILY,
        kls=JepaModel,
        config_kls=JepaConfig,
        factory=create_jepa_model,
        state_dict_converter=convert_jepa_state_dict,
    )

    register_jepa_configs(container)

    # JEPA Classifier
    register_model_family(
        container,
        JEPA_CLASSIFIER_FAMILY,
        kls=JepaClassifierModel,
        config_kls=JepaClassifierConfig,
        factory=create_jepa_classifier_model,
    )

    register_jepa_classifier_configs(container)

    # LLaMA
    register_model_family(
        container,
        LLAMA_FAMILY,
        kls=TransformerLM,
        config_kls=LLaMAConfig,
        factory=create_llama_model,
        state_dict_converter=convert_llama_state_dict,
        shard_specs=get_llama_shard_specs,
        compiler=compile_transformer_lm,
        hugging_face_saver=save_as_hg_llama,
    )

    register_llama_configs(container)

    # Mistral
    register_model_family(
        container,
        MISTRAL_FAMILY,
        kls=TransformerLM,
        config_kls=MistralConfig,
        factory=create_mistral_model,
        state_dict_converter=convert_mistral_state_dict,
        compiler=compile_transformer_lm,
    )

    register_mistral_configs(container)

    # OPT
    register_model_family(
        container,
        OPT_MODEL_FAMILY,
        kls=TransformerLM,
        config_kls=OPTConfig,
        factory=create_opt_model,
        state_dict_converter=convert_opt_state_dict,
        compiler=compile_transformer_lm,
    )

    register_opt_configs(container)

    # NLLB
    register_nllb_configs(container)

    # Qwen
    register_model_family(
        container,
        QWEN_FAMILY,
        kls=TransformerLM,
        config_kls=QwenConfig,
        factory=create_qwen_model,
        state_dict_converter=convert_qwen_state_dict,
        shard_specs=get_qwen_shard_specs,
        compiler=compile_transformer_lm,
        hugging_face_saver=save_as_hg_qwen,
    )

    register_qwen_configs(container)

    # S2T Transformer
    register_model_family(
        container,
        S2T_TRANSFORMER_FAMILY,
        kls=TransformerModel,
        config_kls=S2TTransformerConfig,
        factory=create_s2t_transformer_model,
        state_dict_converter=convert_s2t_transformer_state_dict,
    )

    register_s2t_transformer_configs(container)

    # Transformer
    register_model_family(
        container,
        TRANSFORMER_FAMILY,
        kls=TransformerModel,
        config_kls=TransformerConfig,
        factory=create_transformer_model,
        state_dict_converter=convert_transformer_state_dict,
    )

    register_transformer_configs(container)

    # w2v-BERT
    register_model_family(
        container,
        W2VBERT_FAMILY,
        kls=W2VBertModel,
        config_kls=W2VBertConfig,
        factory=create_w2vbert_model,
        state_dict_converter=convert_w2vbert_state_dict,
    )

    register_w2vbert_configs(container)

    # wav2vec 2.0
    register_model_family(
        container,
        WAV2VEC2_FAMILY,
        kls=Wav2Vec2Model,
        config_kls=Wav2Vec2Config,
        factory=create_wav2vec2_model,
        state_dict_converter=convert_wav2vec2_state_dict,
    )

    register_wav2vec2_configs(container)

    # wav2vec 2.0 ASR
    register_model_family(
        container,
        WAV2VEC2_ASR_FAMILY,
        kls=Wav2Vec2AsrModel,
        config_kls=Wav2Vec2AsrConfig,
        factory=create_wav2vec2_asr_model,
        state_dict_converter=convert_wav2vec2_asr_state_dict,
    )

    register_wav2vec2_asr_configs(container)
