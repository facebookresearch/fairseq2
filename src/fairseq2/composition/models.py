# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Protocol, TypeVar

from torch.nn import Module

from fairseq2.error import InternalError
from fairseq2.models import (
    HuggingFaceExporter,
    LayerwiseACApplier,
    ModelCompiler,
    ModelFactory,
    ModelFamily,
    ModelFSDPApplier,
    ModelStateDictConverter,
    ShardSpecsProvider,
    StandardModelFamily,
)
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
    export_llama,
    register_llama_configs,
)
from fairseq2.models.llama4 import (
    LLAMA4_FAMILY,
    Llama4Config,
    convert_llama4_state_dict,
    create_llama4_model,
    get_llama4_shard_specs,
    register_llama4_configs,
)
from fairseq2.models.mistral import (
    MISTRAL_FAMILY,
    MistralConfig,
    convert_mistral_state_dict,
    create_mistral_model,
    register_mistral_configs,
)
from fairseq2.models.nllb import (
    NLLB_FAMILY,
    NllbConfig,
    convert_nllb_state_dict,
    create_nllb_model,
    register_nllb_configs,
)
from fairseq2.models.qwen import (
    QWEN_FAMILY,
    QwenConfig,
    convert_qwen_state_dict,
    create_qwen_model,
    export_qwen,
    register_qwen_configs,
)
from fairseq2.models.s2t_conformer import (
    S2T_CONFORMER_FAMILY,
    S2TConformerConfig,
    convert_s2t_conformer_state_dict,
    create_s2t_conformer_model,
    register_s2t_conformer_configs,
)
from fairseq2.models.s2t_transformer import (
    S2T_TRANSFORMER_FAMILY,
    S2TTransformerConfig,
    convert_s2t_transformer_state_dict,
    create_s2t_transformer_model,
    register_s2t_transformer_configs,
)
from fairseq2.models.transformer import (
    TransformerModel,
    apply_ac_to_transformer,
    apply_fsdp_to_transformer,
)
from fairseq2.models.transformer_lm import (
    TransformerLM,
    apply_ac_to_transformer_lm,
    apply_fsdp_to_transformer_lm,
    compile_transformer_lm,
)
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
    apply_ac_to_wav2vec2,
    apply_fsdp_to_wav2vec2,
    convert_wav2vec2_state_dict,
    create_wav2vec2_model,
    register_wav2vec2_configs,
)
from fairseq2.models.wav2vec2.asr import (
    WAV2VEC2_ASR_FAMILY,
    Wav2Vec2AsrConfig,
    Wav2Vec2AsrModel,
    apply_ac_to_wav2vec2_asr,
    apply_fsdp_to_wav2vec2_asr,
    convert_wav2vec2_asr_state_dict,
    create_wav2vec2_asr_model,
    register_wav2vec2_asr_configs,
)
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyLookup,
    DependencyResolver,
    wire_object,
)

ModelT_co = TypeVar("ModelT_co", bound=Module, covariant=True)

ModelConfigT_contra = TypeVar("ModelConfigT_contra", contravariant=True)


class AdvancedModelFactory(Protocol[ModelConfigT_contra, ModelT_co]):
    def __call__(
        self, resolver: DependencyResolver, config: ModelConfigT_contra
    ) -> ModelT_co: ...


ModelT = TypeVar("ModelT", bound=Module)

ModelConfigT = TypeVar("ModelConfigT")


def register_model_family(
    container: DependencyContainer,
    name: str,
    kls: type[ModelT],
    config_kls: type[ModelConfigT],
    *,
    factory: ModelFactory[ModelConfigT, ModelT] | None = None,
    advanced_factory: AdvancedModelFactory[ModelConfigT, ModelT] | None = None,
    supports_meta: bool = True,
    restrict: bool = True,
    state_dict_converter: ModelStateDictConverter[ModelConfigT] | None = None,
    shard_specs: ShardSpecsProvider[ModelConfigT] | None = None,
    compiler: ModelCompiler[ModelT] | None = None,
    fsdp_applier: ModelFSDPApplier[ModelT] | None = None,
    layerwise_ac_applier: LayerwiseACApplier[ModelT] | None = None,
    hg_exporter: HuggingFaceExporter[ModelConfigT] | None = None,
) -> None:
    if advanced_factory is not None:
        if factory is not None:
            raise ValueError(
                "`factory` and `advanced_factory` must not be specified at the same time."
            )
    elif factory is None:
        raise ValueError("`factory` or `advanced_factory` must be specified.")

    def create_family(resolver: DependencyResolver) -> ModelFamily:
        nonlocal factory

        if advanced_factory is not None:

            def create_model(config: ModelConfigT) -> ModelT:
                return advanced_factory(resolver, config)

            factory = create_model
        elif factory is None:
            raise InternalError("`factory` is `None`.")

        configs = DependencyLookup(resolver, config_kls)

        return wire_object(
            resolver,
            StandardModelFamily,
            name=name,
            kls=kls,
            configs=configs,
            factory=factory,
            supports_meta=supports_meta,
            restrict=restrict,
            state_dict_converter=state_dict_converter,
            shard_specs=shard_specs,
            compiler=compiler,
            fsdp_applier=fsdp_applier,
            layerwise_ac_applier=layerwise_ac_applier,
            hg_exporter=hg_exporter,
        )

    container.register(ModelFamily, create_family, key=name)


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
        compiler=compile_transformer_lm,
        fsdp_applier=apply_fsdp_to_transformer_lm,
        layerwise_ac_applier=apply_ac_to_transformer_lm,
        hg_exporter=export_llama,
    )

    register_llama_configs(container)

    # Llama 4
    register_model_family(
        container,
        LLAMA4_FAMILY,
        kls=TransformerLM,
        config_kls=Llama4Config,
        factory=create_llama4_model,
        state_dict_converter=convert_llama4_state_dict,
        shard_specs=get_llama4_shard_specs,
        compiler=compile_transformer_lm,
        fsdp_applier=apply_fsdp_to_transformer_lm,
        layerwise_ac_applier=apply_ac_to_transformer_lm,
        hg_exporter=None,  # export not yet implemented
    )

    register_llama4_configs(container)

    # Mistral
    register_model_family(
        container,
        MISTRAL_FAMILY,
        kls=TransformerLM,
        config_kls=MistralConfig,
        factory=create_mistral_model,
        state_dict_converter=convert_mistral_state_dict,
        compiler=compile_transformer_lm,
        fsdp_applier=apply_fsdp_to_transformer_lm,
        layerwise_ac_applier=apply_ac_to_transformer_lm,
    )

    register_mistral_configs(container)

    # NLLB
    register_model_family(
        container,
        NLLB_FAMILY,
        kls=TransformerModel,
        config_kls=NllbConfig,
        factory=create_nllb_model,
        state_dict_converter=convert_nllb_state_dict,
        fsdp_applier=apply_fsdp_to_transformer,
        layerwise_ac_applier=apply_ac_to_transformer,
    )

    register_nllb_configs(container)

    # Qwen
    register_model_family(
        container,
        QWEN_FAMILY,
        kls=TransformerLM,
        config_kls=QwenConfig,
        factory=create_qwen_model,
        state_dict_converter=convert_qwen_state_dict,
        compiler=compile_transformer_lm,
        fsdp_applier=apply_fsdp_to_transformer_lm,
        layerwise_ac_applier=apply_ac_to_transformer_lm,
        hg_exporter=export_qwen,
    )

    register_qwen_configs(container)

    # S2T Conformer
    register_model_family(
        container,
        S2T_CONFORMER_FAMILY,
        kls=TransformerModel,
        config_kls=S2TConformerConfig,
        factory=create_s2t_conformer_model,
        state_dict_converter=convert_s2t_conformer_state_dict,
        fsdp_applier=apply_fsdp_to_transformer,
        layerwise_ac_applier=apply_ac_to_transformer,
    )

    register_s2t_conformer_configs(container)

    # S2T Transformer
    register_model_family(
        container,
        S2T_TRANSFORMER_FAMILY,
        kls=TransformerModel,
        config_kls=S2TTransformerConfig,
        factory=create_s2t_transformer_model,
        state_dict_converter=convert_s2t_transformer_state_dict,
        fsdp_applier=apply_fsdp_to_transformer,
        layerwise_ac_applier=apply_ac_to_transformer,
    )

    register_s2t_transformer_configs(container)

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
        fsdp_applier=apply_fsdp_to_wav2vec2,
        layerwise_ac_applier=apply_ac_to_wav2vec2,
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
        fsdp_applier=apply_fsdp_to_wav2vec2_asr,
        layerwise_ac_applier=apply_ac_to_wav2vec2_asr,
    )

    register_wav2vec2_asr_configs(container)
