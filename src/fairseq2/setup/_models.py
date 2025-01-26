# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.config_registry import ConfigProvider
from fairseq2.context import RuntimeContext
from fairseq2.models import ModelHandler
from fairseq2.models.jepa import JepaConfig, JepaModelHandler, register_jepa_configs
from fairseq2.models.jepa.classifier import (
    JepaClassifierConfig,
    JepaClassifierModelHandler,
    register_jepa_classifier_configs,
)
from fairseq2.models.llama import LLaMAConfig, LLaMAModelHandler, register_llama_configs
from fairseq2.models.mistral import (
    MistralConfig,
    MistralModelHandler,
    register_mistral_configs,
)
from fairseq2.models.nllb import register_nllb_configs
from fairseq2.models.s2t_transformer import (
    S2TTransformerConfig,
    S2TTransformerModelHandler,
    register_s2t_transformer_configs,
)
from fairseq2.models.transformer import (
    TransformerConfig,
    TransformerModelHandler,
    register_transformer_configs,
)
from fairseq2.models.w2vbert import (
    W2VBertConfig,
    W2VBertModelHandler,
    register_w2vbert_configs,
)
from fairseq2.models.wav2vec2 import (
    Wav2Vec2Config,
    Wav2Vec2ModelHandler,
    register_wav2vec2_configs,
)
from fairseq2.models.wav2vec2.asr import (
    Wav2Vec2AsrConfig,
    Wav2Vec2AsrModelHandler,
    register_wav2vec2_asr_configs,
)
from fairseq2.utils.file import StandardTensorLoader, TorchTensorLoader


def _register_models(context: RuntimeContext) -> None:
    asset_download_manager = context.asset_download_manager

    tensor_loader = StandardTensorLoader(context.file_system)

    unsafe_tensor_loader = TorchTensorLoader(context.file_system, restrict=False)

    registry = context.get_registry(ModelHandler)

    handler: ModelHandler

    configs: ConfigProvider[object]

    # JEPA
    configs = context.get_config_registry(JepaConfig)

    default_arch = "base"

    handler = JepaModelHandler(
        configs, default_arch, asset_download_manager, tensor_loader
    )

    registry.register(handler.family, handler)

    register_jepa_configs(context)

    # JEPA Classifier
    configs = context.get_config_registry(JepaClassifierConfig)

    default_arch = "base"

    handler = JepaClassifierModelHandler(
        configs, default_arch, asset_download_manager, tensor_loader
    )

    registry.register(handler.family, handler)

    register_jepa_classifier_configs(context)

    # LLaMA
    configs = context.get_config_registry(LLaMAConfig)

    default_arch = "llama3_1_8b"

    handler = LLaMAModelHandler(
        configs, default_arch, asset_download_manager, tensor_loader
    )

    registry.register(handler.family, handler)

    register_llama_configs(context)

    # Mistral
    configs = context.get_config_registry(MistralConfig)

    default_arch = "7b"

    handler = MistralModelHandler(
        configs, default_arch, asset_download_manager, tensor_loader
    )

    registry.register(handler.family, handler)

    register_mistral_configs(context)

    # NLLB
    register_nllb_configs(context)

    # S2T Transformer
    configs = context.get_config_registry(S2TTransformerConfig)

    default_arch = "medium"

    handler = S2TTransformerModelHandler(
        configs, default_arch, asset_download_manager, unsafe_tensor_loader
    )

    registry.register(handler.family, handler)

    register_s2t_transformer_configs(context)

    # Transformer
    configs = context.get_config_registry(TransformerConfig)

    default_arch = "base"

    handler = TransformerModelHandler(
        configs, default_arch, asset_download_manager, unsafe_tensor_loader
    )

    registry.register(handler.family, handler)

    register_transformer_configs(context)

    # w2v-BERT
    configs = context.get_config_registry(W2VBertConfig)

    default_arch = "300m"

    handler = W2VBertModelHandler(
        configs, default_arch, asset_download_manager, tensor_loader
    )

    registry.register(handler.family, handler)

    register_w2vbert_configs(context)

    # wav2vec 2.0
    configs = context.get_config_registry(Wav2Vec2Config)

    default_arch = "base"

    handler = Wav2Vec2ModelHandler(
        configs, default_arch, asset_download_manager, unsafe_tensor_loader
    )

    registry.register(handler.family, handler)

    register_wav2vec2_configs(context)

    # wav2vec 2.0 ASR
    configs = context.get_config_registry(Wav2Vec2AsrConfig)

    default_arch = "base_10h"

    handler = Wav2Vec2AsrModelHandler(
        configs, default_arch, asset_download_manager, unsafe_tensor_loader
    )

    registry.register(handler.family, handler)

    register_wav2vec2_asr_configs(context)
