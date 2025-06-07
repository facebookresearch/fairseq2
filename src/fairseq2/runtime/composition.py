# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from collections.abc import MutableMapping
from types import NoneType

from importlib_metadata import entry_points

from fairseq2.assets import (
    AssetDownloadManager,
    AssetStore,
    _create_asset_download_manager,
    _load_asset_store,
)
from fairseq2.data.tokenizers import register_tokenizer_family
from fairseq2.data.tokenizers.char import CHAR_TOKENIZER_FAMILY, _load_char_tokenizer
from fairseq2.file_system import FileSystem, LocalFileSystem
from fairseq2.logging import log
from fairseq2.model.checkpoint import (
    ModelCheckpointLoader,
    _create_model_checkpoint_loader,
)
from fairseq2.model.sharder import ModelSharder, _create_model_sharder
from fairseq2.models import register_model_family
from fairseq2.models.jepa import (
    JEPA_FAMILY,
    JepaConfig,
    JepaModel,
    _convert_jepa_checkpoint,
    _create_jepa_model,
    _register_jepa_configs,
)
from fairseq2.models.jepa.classifier import (
    JEPA_CLASSIFIER_FAMILY,
    JepaClassifierConfig,
    JepaClassifierModel,
    _create_jepa_classifier_model,
    _register_jepa_classifier_configs,
)
from fairseq2.models.llama import (
    LLAMA_FAMILY,
    LLaMAConfig,
    LLaMATokenizerConfig,
    _convert_llama_checkpoint,
    _create_llama_model,
    _get_llama_shard_specs,
    _load_llama_tokenizer,
    _register_llama_configs,
    save_as_hg_llama,
)
from fairseq2.models.mistral import (
    MISTRAL_FAMILY,
    MistralConfig,
    _convert_mistral_checkpoint,
    _create_mistral_model,
    _load_mistral_tokenizer,
    _register_mistral_configs,
)
from fairseq2.models.nllb import (
    NLLB_FAMILY,
    NllbTokenizerConfig,
    _load_nllb_tokenizer,
    _register_nllb_configs,
)
from fairseq2.models.qwen import (
    QWEN_FAMILY,
    QwenConfig,
    QwenTokenizerConfig,
    _convert_qwen_checkpoint,
    _create_qwen_model,
    _get_qwen_shard_specs,
    _load_qwen_tokenizer,
    _register_qwen_configs,
    save_as_hg_qwen,
)
from fairseq2.models.s2t_transformer import (
    S2T_TRANSFORMER_FAMILY,
    S2TTransformerConfig,
    S2TTransformerTokenizerConfig,
    _convert_s2t_transformer_checkpoint,
    _create_s2t_transformer_model,
    _load_s2t_transformer_tokenizer,
    _register_s2t_transformer_configs,
)
from fairseq2.models.transformer import (
    TRANSFORMER_FAMILY,
    TransformerConfig,
    TransformerModel,
    _convert_transformer_checkpoint,
    _create_transformer_model,
    _register_transformer_configs,
)
from fairseq2.models.transformer_lm import TransformerLM, compile_transformer_lm
from fairseq2.models.w2vbert import (
    W2VBERT_FAMILY,
    W2VBertConfig,
    W2VBertModel,
    _convert_w2vbert_checkpoint,
    _create_w2vbert_model,
    _register_w2vbert_configs,
)
from fairseq2.models.wav2vec2 import (
    WAV2VEC2_FAMILY,
    Wav2Vec2Config,
    Wav2Vec2Model,
    _convert_wav2vec2_checkpoint,
    _create_wav2vec2_model,
    _register_wav2vec2_configs,
)
from fairseq2.models.wav2vec2.asr import (
    WAV2VEC2_ASR_FAMILY,
    Wav2Vec2AsrConfig,
    Wav2Vec2AsrModel,
    _convert_wav2vec2_asr_checkpoint,
    _create_wav2vec2_asr_model,
    _register_wav2vec2_asr_configs,
)
from fairseq2.runtime.dependency import DependencyContainer
from fairseq2.utils.progress import NoopProgressReporter, ProgressReporter
from fairseq2.utils.structured import StandardValueConverter, ValueConverter
from fairseq2.utils.validation import (
    ObjectValidator,
    StandardObjectValidator,
)
from fairseq2.utils.yaml import (
    RuamelYamlDumper,
    RuamelYamlLoader,
    YamlDumper,
    YamlLoader,
)


def _register_library(container: DependencyContainer) -> None:
    # Environment Variables
    container.register_instance(MutableMapping, os.environ, key="env")

    # FileSystem
    file_system = LocalFileSystem()

    container.register_instance(FileSystem, file_system)

    # YAML
    yaml_loader = RuamelYamlLoader(file_system)
    yaml_dumper = RuamelYamlDumper(file_system)

    container.register_instance(YamlLoader, yaml_loader)
    container.register_instance(YamlDumper, yaml_dumper)

    # ValueConverter
    value_converter = StandardValueConverter()

    container.register_instance(ValueConverter, value_converter)

    # ProgressReporter
    progress_reporter = NoopProgressReporter()

    container.register_instance(ProgressReporter, progress_reporter)

    # Validator
    validator = StandardObjectValidator()

    container.register_instance(ObjectValidator, validator)

    # AssetStore
    container.register(AssetStore, _load_asset_store)

    # AssetDownloadManager
    container.register(AssetDownloadManager, _create_asset_download_manager)

    # ModelCheckpointLoader
    container.register(ModelCheckpointLoader, _create_model_checkpoint_loader)

    # ModelSharder
    container.register(ModelSharder, _create_model_sharder)

    # Models
    _register_model_families(container)

    # Tokenizers
    _register_tokenizer_families(container)

    # Extensions
    _register_extensions(container)


def _register_model_families(container: DependencyContainer) -> None:
    # JEPA
    register_model_family(
        container,
        JEPA_FAMILY,
        kls=JepaModel,
        config_kls=JepaConfig,
        factory=_create_jepa_model,
        checkpoint_converter=_convert_jepa_checkpoint,
    )

    _register_jepa_configs(container)

    # JEPA Classifier
    register_model_family(
        container,
        JEPA_CLASSIFIER_FAMILY,
        kls=JepaClassifierModel,
        config_kls=JepaClassifierConfig,
        factory=_create_jepa_classifier_model,
    )

    _register_jepa_classifier_configs(container)

    # LLaMA
    register_model_family(
        container,
        LLAMA_FAMILY,
        kls=TransformerLM,
        config_kls=LLaMAConfig,
        factory=_create_llama_model,
        checkpoint_converter=_convert_llama_checkpoint,
        shard_specs=_get_llama_shard_specs,
        compiler=compile_transformer_lm,
        hugging_face_saver=save_as_hg_llama,
    )

    _register_llama_configs(container)

    # Mistral
    register_model_family(
        container,
        MISTRAL_FAMILY,
        kls=TransformerLM,
        config_kls=MistralConfig,
        factory=_create_mistral_model,
        checkpoint_converter=_convert_mistral_checkpoint,
        compiler=compile_transformer_lm,
    )

    _register_mistral_configs(container)

    # NLLB
    _register_nllb_configs(container)

    # Qwen
    register_model_family(
        container,
        QWEN_FAMILY,
        kls=TransformerLM,
        config_kls=QwenConfig,
        factory=_create_qwen_model,
        checkpoint_converter=_convert_qwen_checkpoint,
        shard_specs=_get_qwen_shard_specs,
        compiler=compile_transformer_lm,
        hugging_face_saver=save_as_hg_qwen,
    )

    _register_qwen_configs(container)

    # S2T Transformer
    register_model_family(
        container,
        S2T_TRANSFORMER_FAMILY,
        kls=TransformerModel,
        config_kls=S2TTransformerConfig,
        factory=_create_s2t_transformer_model,
        checkpoint_converter=_convert_s2t_transformer_checkpoint,
    )

    _register_s2t_transformer_configs(container)

    # Transformer
    register_model_family(
        container,
        TRANSFORMER_FAMILY,
        kls=TransformerModel,
        config_kls=TransformerConfig,
        factory=_create_transformer_model,
        checkpoint_converter=_convert_transformer_checkpoint,
    )

    _register_transformer_configs(container)

    # w2v-BERT
    register_model_family(
        container,
        W2VBERT_FAMILY,
        kls=W2VBertModel,
        config_kls=W2VBertConfig,
        factory=_create_w2vbert_model,
        checkpoint_converter=_convert_w2vbert_checkpoint,
    )

    _register_w2vbert_configs(container)

    # wav2vec 2.0
    register_model_family(
        container,
        WAV2VEC2_FAMILY,
        kls=Wav2Vec2Model,
        config_kls=Wav2Vec2Config,
        factory=_create_wav2vec2_model,
        checkpoint_converter=_convert_wav2vec2_checkpoint,
    )

    _register_wav2vec2_configs(container)

    # wav2vec 2.0 ASR
    register_model_family(
        container,
        WAV2VEC2_ASR_FAMILY,
        kls=Wav2Vec2AsrModel,
        config_kls=Wav2Vec2AsrConfig,
        factory=_create_wav2vec2_asr_model,
        checkpoint_converter=_convert_wav2vec2_asr_checkpoint,
    )

    _register_wav2vec2_asr_configs(container)


def _register_tokenizer_families(container: DependencyContainer) -> None:
    # Char
    register_tokenizer_family(
        container,
        CHAR_TOKENIZER_FAMILY,
        config_kls=NoneType,
        loader=_load_char_tokenizer,
    )

    # LLaMA
    register_tokenizer_family(
        container,
        LLAMA_FAMILY,
        config_kls=LLaMATokenizerConfig,
        loader=_load_llama_tokenizer,
    )

    # Mistral
    register_tokenizer_family(
        container,
        MISTRAL_FAMILY,
        config_kls=NoneType,
        loader=_load_mistral_tokenizer,
    )

    # Qwen
    register_tokenizer_family(
        container,
        QWEN_FAMILY,
        config_kls=QwenTokenizerConfig,
        loader=_load_qwen_tokenizer,
    )

    # NLLB
    register_tokenizer_family(
        container,
        NLLB_FAMILY,
        config_kls=NllbTokenizerConfig,
        loader=_load_nllb_tokenizer,
    )

    # S2T Transformer
    register_tokenizer_family(
        container,
        S2T_TRANSFORMER_FAMILY,
        config_kls=S2TTransformerTokenizerConfig,
        loader=_load_s2t_transformer_tokenizer,
    )


def _register_extensions(container: DependencyContainer) -> None:
    should_trace = "FAIRSEQ2_EXTENSION_TRACE" in os.environ

    for entry_point in entry_points(group="fairseq2.extension"):
        try:
            extension = entry_point.load()

            extension(container)
        except TypeError as ex:
            # Not ideal, but there is no other way to find out whether the error
            # was raised due to a wrong function signature.
            if not str(ex).startswith(f"{entry_point.attr}()"):
                raise

            if should_trace:
                raise ExtensionError(
                    entry_point.value, f"The '{entry_point.value}' entry point cannot be run as an extension since its signature does not match `extension_function(container: DependencyContainer)`."  # fmt: skip
                ) from None

            log.warning("'{}' entry point is not a valid extension. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.", entry_point.value)  # fmt: skip
        except Exception as ex:
            if should_trace:
                raise ExtensionError(
                    entry_point.value, f"'{entry_point.value}' extension has failed. See the nested exception for details."  # fmt: skip
                ) from ex

            log.warning("'{}' extension failed. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.", entry_point.value)  # fmt: skip

        if should_trace:
            log.info("`{}` extension registered successfully.", entry_point.value)  # fmt: skip


class ExtensionError(Exception):
    entry_point: str

    def __init__(self, entry_point: str, message: str) -> None:
        super().__init__(message)

        self.entry_point = entry_point
