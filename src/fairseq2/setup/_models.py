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
    FSDPApplier,
    HuggingFaceSaver,
    ModelCompiler,
    ModelFactory,
    ModelHandler,
    ShardSpecsProvider,
    create_checkpoint_loader,
)
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
    get_llama_shard_specs,
    register_llama_configs,
    save_as_hg_llama,
)
from fairseq2.models.mistral import (
    MISTRAL_MODEL_FAMILY,
    MistralConfig,
    convert_mistral_checkpoint,
    create_mistral_model,
    register_mistral_configs,
)
from fairseq2.models.nllb import register_nllb_configs
from fairseq2.models.qwen import (
    QWEN_MODEL_FAMILY,
    QwenConfig,
    convert_qwen_checkpoint,
    create_qwen_model,
    get_qwen_shard_specs,
    register_qwen_configs,
    save_as_hg_qwen,
)
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
    TransformerLM,
    compile_transformer_lm,
)
from fairseq2.models.utils.ac import apply_default_activation_checkpointing
from fairseq2.models.utils.fsdp import apply_default_fsdp
from fairseq2.models.utils.sharder import create_model_sharder
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
        TransformerLM,
        LLaMAConfig,
        default_llama_arch,
        factory=create_llama_model,
        checkpoint_converter=convert_llama_checkpoint,
        shard_specs=get_llama_shard_specs,
        compiler=compile_transformer_lm,
        hugging_face_saver=save_as_hg_llama,
    )

    register_llama_configs(context)

    # Mistral
    default_mistral_arch = "7b"

    registrar.register_family(
        MISTRAL_MODEL_FAMILY,
        TransformerLM,
        MistralConfig,
        default_mistral_arch,
        factory=create_mistral_model,
        checkpoint_converter=convert_mistral_checkpoint,
        compiler=compile_transformer_lm,
    )

    register_mistral_configs(context)

    # NLLB
    register_nllb_configs(context)

    # Qwen
    default_qwen_arch = "qwen25_7b"

    registrar.register_family(
        QWEN_MODEL_FAMILY,
        TransformerLM,
        QwenConfig,
        default_qwen_arch,
        factory=create_qwen_model,
        checkpoint_converter=convert_qwen_checkpoint,
        shard_specs=get_qwen_shard_specs,
        compiler=compile_transformer_lm,
        hugging_face_saver=save_as_hg_qwen,
    )

    register_qwen_configs(context)

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
        shard_specs: ShardSpecsProvider[ModelConfigT] | None = None,
        compiler: ModelCompiler[ModelT] | None = None,
        ac_applier: ActivationCheckpointApplier[ModelT] | None = None,
        fsdp_applier: FSDPApplier[ModelT] | None = None,
        hugging_face_saver: HuggingFaceSaver[ModelConfigT] | None = None,
    ) -> None:
        file_system = self._context.file_system

        asset_download_manager = self._context.asset_download_manager

        progress_reporter = self._context.progress_reporter

        checkpoint_loader = create_checkpoint_loader(file_system)

        sharder = create_model_sharder()

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
            checkpoint_loader,
            sharder,
            progress_reporter,
            supports_meta=supports_meta,
            restrict=restrict,
            checkpoint_converter=checkpoint_converter,
            shard_specs=shard_specs,
            compiler=compiler,
            ac_applier=ac_applier,
            fsdp_applier=fsdp_applier,
            hugging_face_saver=hugging_face_saver,
        )

        self._registry.register(handler.family, handler)
