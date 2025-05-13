# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, TypeVar, final

import torch
from torch.nn import Module

from fairseq2.context import RuntimeContext
from fairseq2.models import (
    ActivationCheckpointApplier,
    CheckpointConverter,
    DelegatingModelHandler,
    FsdpApplier,
    HuggingFaceExporter,
    ModelCompiler,
    ModelFactory,
    ModelHandler,
    ModelSharder,
)
from fairseq2.models.ac import apply_default_activation_checkpointing
from fairseq2.models.fsdp import apply_default_fsdp
from fairseq2.models.jepa import (
    JEPA_MODEL_FAMILY,
    JepaConfig,
    JepaModel,
    convert_jepa_checkpoint,
    create_jepa_model,
    register_jepa_configs,
)
from fairseq2.models.jepa.classifier import (
    JEPA_CLASSIFIER_MODEL_FAMILY,
    JepaClassifierConfig,
    JepaClassifierModel,
    create_jepa_classifier_model,
    register_jepa_classifier_configs,
)
from fairseq2.models.llama import (
    LLAMA_MODEL_FAMILY,
    LLaMAConfig,
    convert_llama_checkpoint,
    create_llama_model,
    export_llama_checkpoint,
    register_llama_configs,
    shard_llama_model,
)
from fairseq2.models.mistral import (
    MISTRAL_MODEL_FAMILY,
    MistralConfig,
    convert_mistral_checkpoint,
    create_mistral_model,
    register_mistral_configs,
)
from fairseq2.models.nllb import register_nllb_configs
from fairseq2.models.s2t_transformer import (
    S2T_TRANSFORMER_MODEL_FAMILY,
    S2TTransformerConfig,
    convert_s2t_transformer_checkpoint,
    create_s2t_transformer_model,
    register_s2t_transformer_configs,
)
from fairseq2.models.transformer import (
    TRANSFORMER_MODEL_FAMILY,
    TransformerConfig,
    TransformerModel,
    convert_transformer_checkpoint,
    create_transformer_model,
    register_transformer_configs,
)
from fairseq2.models.transformer_lm import (
    TransformerLanguageModel,
    compile_transformer_lm,
)
from fairseq2.models.w2vbert import (
    W2VBERT_MODEL_FAMILY,
    W2VBertConfig,
    W2VBertModel,
    convert_w2vbert_checkpoint,
    create_w2vbert_model,
    register_w2vbert_configs,
)
from fairseq2.models.wav2vec2 import (
    WAV2VEC2_MODEL_FAMILY,
    Wav2Vec2Config,
    Wav2Vec2Model,
    convert_wav2vec2_checkpoint,
    create_wav2vec2_model,
    register_wav2vec2_configs,
)
from fairseq2.models.wav2vec2.asr import (
    WAV2VEC2_ASR_MODEL_FAMILY,
    Wav2Vec2AsrConfig,
    Wav2Vec2AsrModel,
    convert_wav2vec2_asr_checkpoint,
    create_wav2vec2_asr_model,
    register_wav2vec2_asr_configs,
)
from fairseq2.registry import Registry
from fairseq2.utils.io import AutoTensorLoader


def _register_model_families(context: RuntimeContext) -> None:
    registrar = ModelRegistrar(context)

    # JEPA
    default_jepa_arch = "base"

    registrar.register_family(
        JEPA_MODEL_FAMILY,
        JepaModel,
        JepaConfig,
        default_jepa_arch,
        factory=create_jepa_model,
        checkpoint_converter=convert_jepa_checkpoint,
    )

    register_jepa_configs(context)

    # JEPA Classifier
    default_jepa_classifier_arch = "base"

    registrar.register_family(
        JEPA_CLASSIFIER_MODEL_FAMILY,
        JepaClassifierModel,
        JepaClassifierConfig,
        default_jepa_classifier_arch,
        factory=create_jepa_classifier_model,
    )

    register_jepa_classifier_configs(context)

    # LLaMA
    default_llama_arch = "llama3_1_8b"

    registrar.register_family(
        LLAMA_MODEL_FAMILY,
        TransformerLanguageModel,
        LLaMAConfig,
        default_llama_arch,
        factory=create_llama_model,
        checkpoint_converter=convert_llama_checkpoint,
        sharder=shard_llama_model,
        compiler=compile_transformer_lm,
        hugging_face_exporter=export_llama_checkpoint,
    )

    register_llama_configs(context)

    # Mistral
    default_mistral_arch = "7b"

    registrar.register_family(
        MISTRAL_MODEL_FAMILY,
        TransformerLanguageModel,
        MistralConfig,
        default_mistral_arch,
        factory=create_mistral_model,
        checkpoint_converter=convert_mistral_checkpoint,
        compiler=compile_transformer_lm,
    )

    register_mistral_configs(context)

    # NLLB
    register_nllb_configs(context)

    # S2T Transformer
    default_s2t_transformer_arch = "medium"

    registrar.register_family(
        S2T_TRANSFORMER_MODEL_FAMILY,
        TransformerModel,
        S2TTransformerConfig,
        default_s2t_transformer_arch,
        factory=create_s2t_transformer_model,
        checkpoint_converter=convert_s2t_transformer_checkpoint,
    )

    register_s2t_transformer_configs(context)

    # Transformer
    default_transformer_arch = "base"

    registrar.register_family(
        TRANSFORMER_MODEL_FAMILY,
        TransformerModel,
        TransformerConfig,
        default_transformer_arch,
        factory=create_transformer_model,
        checkpoint_converter=convert_transformer_checkpoint,
    )

    register_transformer_configs(context)

    # w2v-BERT
    default_w2vbert_arch = "300m"

    registrar.register_family(
        W2VBERT_MODEL_FAMILY,
        W2VBertModel,
        W2VBertConfig,
        default_w2vbert_arch,
        factory=create_w2vbert_model,
        checkpoint_converter=convert_w2vbert_checkpoint,
    )

    register_w2vbert_configs(context)

    # wav2vec 2.0
    default_wav2vec2_arch = "base"

    registrar.register_family(
        WAV2VEC2_MODEL_FAMILY,
        Wav2Vec2Model,
        Wav2Vec2Config,
        default_wav2vec2_arch,
        factory=create_wav2vec2_model,
        checkpoint_converter=convert_wav2vec2_checkpoint,
    )

    register_wav2vec2_configs(context)

    # wav2vec 2.0 ASR
    default_wav2vec2_asr_arch = "base_10h"

    registrar.register_family(
        WAV2VEC2_ASR_MODEL_FAMILY,
        Wav2Vec2AsrModel,
        Wav2Vec2AsrConfig,
        default_wav2vec2_asr_arch,
        factory=create_wav2vec2_asr_model,
        checkpoint_converter=convert_wav2vec2_asr_checkpoint,
    )

    register_wav2vec2_asr_configs(context)


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
        supports_compilation: bool = True,
        supports_ac: bool = True,
        supports_fsdp: bool = True,
        restrict: bool = True,
        checkpoint_converter: CheckpointConverter[ModelConfigT] | None = None,
        sharder: ModelSharder[ModelT, ModelConfigT] | None = None,
        compiler: ModelCompiler[ModelT] | None = None,
        ac_applier: ActivationCheckpointApplier[ModelT] | None = None,
        fsdp_applier: FsdpApplier[ModelT] | None = None,
        hugging_face_exporter: HuggingFaceExporter[ModelConfigT] | None = None,
    ) -> None:
        file_system = self._context.file_system

        asset_download_manager = self._context.asset_download_manager

        tensor_loader = AutoTensorLoader(file_system)

        configs = self._context.get_config_registry(config_kls)

        if supports_compilation:
            if compiler is None:

                def compile(model: ModelT, **kwargs: Any) -> None:
                    torch.compile(model)

                compiler = compile
        else:
            compiler = None

        if supports_ac:
            if ac_applier is None:
                ac_applier = apply_default_activation_checkpointing
        else:
            ac_applier = None

        if supports_fsdp:
            if fsdp_applier is None:
                fsdp_applier = apply_default_fsdp
        else:
            fsdp_applier = None

        handler = DelegatingModelHandler(
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
            ac_applier=ac_applier,
            fsdp_applier=fsdp_applier,
            hugging_face_exporter=hugging_face_exporter,
        )

        self._registry.register(handler.family, handler)
