# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final, TypeVar

from fairseq2.context import RuntimeContext
from fairseq2.models import (
    CheckpointConverter,
    ModelCompiler,
    ModelFactory,
    ModelHandler,
    ModelSharder,
    StandardModelHandler,
)
from fairseq2.models.jepa import (
    convert_jepa_checkpoint,
    create_jepa_model,
    JEPA_MODEL_FAMILY,
    JepaConfig,
    JepaModel,
    register_jepa_configs,
)
from fairseq2.models.jepa.classifier import (
    create_jepa_classifier_model,
    JEPA_CLASSIFIER_MODEL_FAMILY,
    JepaClassifierConfig,
    JepaClassifierModel,
    register_jepa_classifier_configs,
)
from fairseq2.models.llama import (
    compile_llama_model,
    convert_llama_checkpoint,
    create_llama_model,
    LLAMA_MODEL_FAMILY,
    LLaMAConfig,
    register_llama_configs,
    shard_llama_model,
)
from fairseq2.models.mistral import (
    convert_mistral_checkpoint,
    create_mistral_model,
    MISTRAL_MODEL_FAMILY,
    MistralConfig,
    register_mistral_configs,
)
from fairseq2.models.nllb import register_nllb_configs
from fairseq2.models.s2t_transformer import (
    convert_s2t_transformer_checkpoint,
    create_s2t_transformer_model,
    register_s2t_transformer_configs,
    S2T_TRANSFORMER_MODEL_FAMILY,
    S2TTransformerConfig,
)
from fairseq2.models.transformer import (
    convert_transformer_checkpoint,
    create_transformer_model,
    register_transformer_configs,
    TRANSFORMER_MODEL_FAMILY,
    TransformerConfig,
    TransformerModel,
)
from fairseq2.models.transformer_lm import TransformerLanguageModel
from fairseq2.models.w2vbert import (
    convert_w2vbert_checkpoint,
    create_w2vbert_model,
    register_w2vbert_configs,
    W2VBERT_MODEL_FAMILY,
    W2VBertConfig,
    W2VBertModel,
)
from fairseq2.models.wav2vec2 import (
    convert_wav2vec2_checkpoint,
    create_wav2vec2_model,
    register_wav2vec2_configs,
    WAV2VEC2_MODEL_FAMILY,
    Wav2Vec2Config,
    Wav2Vec2Model,
)
from fairseq2.models.wav2vec2.asr import (
    convert_wav2vec2_asr_checkpoint,
    create_wav2vec2_asr_model,
    register_wav2vec2_asr_configs,
    WAV2VEC2_ASR_MODEL_FAMILY,
    Wav2Vec2AsrConfig,
    Wav2Vec2AsrModel,
)
from fairseq2.models.wav2vec2.sonar import (
    create_sonar_speech_model,
    register_sonar_speech_encoder_configs,
    SonarSpeechEncoderConfig,
    SonarSpeechEncoderModel,
    WAV2VEC2_SONAR_SPEECH_MODEL_FAMILY,
)
from fairseq2.registry import Registry
from fairseq2.utils.file import StandardTensorLoader

from torch.nn import Module


def register_model_families(context: RuntimeContext) -> None:
    # fmt: off
    registrar = ModelRegistrar(context)

    # JEPA
    default_arch = "base"

    registrar.register_family(
        JEPA_MODEL_FAMILY,
        JepaModel,
        JepaConfig,
        default_arch,
        create_jepa_model,
        checkpoint_converter=convert_jepa_checkpoint,
    )

    register_jepa_configs(context)

    # JEPA Classifier
    default_arch = "base"

    registrar.register_family(
        JEPA_CLASSIFIER_MODEL_FAMILY,
        JepaClassifierModel,
        JepaClassifierConfig,
        default_arch,
        create_jepa_classifier_model,
    )

    register_jepa_classifier_configs(context)

    # LLaMA
    default_arch = "llama3_1_8b"

    registrar.register_family(
        LLAMA_MODEL_FAMILY,
        TransformerLanguageModel,
        LLaMAConfig,
        default_arch,
        create_llama_model,
        checkpoint_converter=convert_llama_checkpoint,
        sharder=shard_llama_model,
        compiler=compile_llama_model,
    )

    register_llama_configs(context)

    # Mistral
    default_arch = "7b"

    registrar.register_family(
        MISTRAL_MODEL_FAMILY,
        TransformerLanguageModel,
        MistralConfig,
        default_arch,
        create_mistral_model,
        checkpoint_converter=convert_mistral_checkpoint,
    )

    register_mistral_configs(context)

    # NLLB
    register_nllb_configs(context)

    # S2T Transformer
    default_arch = "medium"

    registrar.register_family(
        S2T_TRANSFORMER_MODEL_FAMILY,
        TransformerModel,
        S2TTransformerConfig,
        default_arch,
        create_s2t_transformer_model,
        checkpoint_converter=convert_s2t_transformer_checkpoint,
    )

    register_s2t_transformer_configs(context)

    # Transformer
    default_arch = "base"

    registrar.register_family(
        TRANSFORMER_MODEL_FAMILY,
        TransformerModel,
        TransformerConfig,
        default_arch,
        create_transformer_model,
        checkpoint_converter=convert_transformer_checkpoint,
    )

    register_transformer_configs(context)

    # w2v-BERT
    default_arch = "300m"

    registrar.register_family(
        W2VBERT_MODEL_FAMILY,
        W2VBertModel,
        W2VBertConfig,
        default_arch,
        create_w2vbert_model,
        checkpoint_converter=convert_w2vbert_checkpoint,
    )

    register_w2vbert_configs(context)

    # wav2vec 2.0
    default_arch = "base"

    registrar.register_family(
        WAV2VEC2_MODEL_FAMILY,
        Wav2Vec2Model,
        Wav2Vec2Config,
        default_arch,
        create_wav2vec2_model,
        checkpoint_converter=convert_wav2vec2_checkpoint,
    )

    register_wav2vec2_configs(context)

    # wav2vec 2.0 ASR
    default_arch = "base_10h"

    registrar.register_family(
        WAV2VEC2_ASR_MODEL_FAMILY,
        Wav2Vec2AsrModel,
        Wav2Vec2AsrConfig,
        default_arch,
        create_wav2vec2_asr_model,
        checkpoint_converter=convert_wav2vec2_asr_checkpoint,
    )

    register_wav2vec2_asr_configs(context)

    # wav2vec 2.0 Sonar Speech
    default_arch = "7b"

    registrar.register_family(
        WAV2VEC2_SONAR_SPEECH_MODEL_FAMILY,
        SonarSpeechEncoderModel,
        SonarSpeechEncoderConfig,
        default_arch,
        create_sonar_speech_model,
        checkpoint_converter=convert_wav2vec2_asr_checkpoint,
    )

    register_sonar_speech_encoder_configs(context)

    # fmt: on


ModelT = TypeVar("ModelT", bound=Module)

ModelConfigT = TypeVar("ModelConfigT")


@final
class ModelRegistrar:
    _context: RuntimeContext
    _registry: Registry[ModelHandler]

    def __init__(self, context: RuntimeContext) -> None:
        self._context = context

        self._registry = context.get_registry(ModelHandler)

    def register_family(
        self,
        family: str,
        kls: type[ModelT],
        config_kls: type[ModelConfigT],
        default_arch: str,
        factory: ModelFactory[ModelConfigT, ModelT],
        *,
        supports_meta: bool = True,
        restrict: bool = True,
        checkpoint_converter: CheckpointConverter[ModelConfigT] | None = None,
        sharder: ModelSharder[ModelT, ModelConfigT] | None = None,
        compiler: ModelCompiler[ModelT] | None = None,
    ) -> None:
        file_system = self._context.file_system

        asset_download_manager = self._context.asset_download_manager

        tensor_loader = StandardTensorLoader(file_system)

        configs = self._context.get_config_registry(config_kls)

        handler = StandardModelHandler(
            family,
            kls,
            configs,
            default_arch,
            factory,
            asset_download_manager,
            tensor_loader,
            supports_meta=supports_meta,
            restrict=restrict,
            checkpoint_converter=checkpoint_converter,
            sharder=sharder,
            compiler=compiler,
        )

        self._registry.register(handler.family, handler)
