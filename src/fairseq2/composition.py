# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from collections.abc import MutableMapping

from importlib_metadata import entry_points

from fairseq2.assets import (
    AssetDownloadManager,
    AssetStore,
    create_asset_download_manager,
    load_asset_store,
)
from fairseq2.data.tokenizers import register_tokenizer_family
from fairseq2.data.tokenizers.char import CHAR_TOKENIZER_FAMILY, load_char_tokenizer
from fairseq2.dependency import DependencyContainer
from fairseq2.device import Device
from fairseq2.file_system import FileSystem, LocalFileSystem
from fairseq2.logging import log
from fairseq2.models import register_model_family
from fairseq2.models.checkpoint import CheckpointLoader, create_checkpoint_loader
from fairseq2.models.jepa import (
    JEPA_FAMILY,
    JepaConfig,
    JepaModel,
    convert_jepa_checkpoint,
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
    convert_llama_checkpoint,
    create_llama_model,
    get_llama_shard_specs,
    load_llama_tokenizer,
    register_llama_configs,
    save_as_hg_llama,
)
from fairseq2.models.mistral import (
    MISTRAL_FAMILY,
    MistralConfig,
    convert_mistral_checkpoint,
    create_mistral_model,
    load_mistral_tokenizer,
    register_mistral_configs,
)
from fairseq2.models.nllb import NLLB_FAMILY, load_nllb_tokenizer, register_nllb_configs
from fairseq2.models.qwen import (
    QWEN_FAMILY,
    QwenConfig,
    convert_qwen_checkpoint,
    create_qwen_model,
    get_qwen_shard_specs,
    load_qwen_tokenizer,
    register_qwen_configs,
    save_as_hg_qwen,
)
from fairseq2.models.s2t_transformer import (
    S2T_TRANSFORMER_FAMILY,
    S2TTransformerConfig,
    convert_s2t_transformer_checkpoint,
    create_s2t_transformer_model,
    load_s2t_transformer_tokenizer,
    register_s2t_transformer_configs,
)
from fairseq2.models.transformer import (
    TRANSFORMER_FAMILY,
    TransformerConfig,
    TransformerModel,
    convert_transformer_checkpoint,
    create_transformer_model,
    register_transformer_configs,
)
from fairseq2.models.transformer_lm import TransformerLM, compile_transformer_lm
from fairseq2.models.utils.sharder import ModelSharder, create_model_sharder
from fairseq2.models.w2vbert import (
    W2VBERT_FAMILY,
    W2VBertConfig,
    W2VBertModel,
    convert_w2vbert_checkpoint,
    create_w2vbert_model,
    register_w2vbert_configs,
)
from fairseq2.models.wav2vec2 import (
    WAV2VEC2_FAMILY,
    Wav2Vec2Config,
    Wav2Vec2Model,
    convert_wav2vec2_checkpoint,
    create_wav2vec2_model,
    register_wav2vec2_configs,
)
from fairseq2.models.wav2vec2.asr import (
    WAV2VEC2_ASR_FAMILY,
    Wav2Vec2AsrConfig,
    Wav2Vec2AsrModel,
    convert_wav2vec2_asr_checkpoint,
    create_wav2vec2_asr_model,
    register_wav2vec2_asr_configs,
)
from fairseq2.utils.device import determine_default_device
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.rich import create_rich_progress_reporter
from fairseq2.utils.threading import ThreadPool, create_thread_pool
from fairseq2.utils.yaml import (
    RuamelYamlDumper,
    RuamelYamlLoader,
    YamlDumper,
    YamlLoader,
)


def register_library(container: DependencyContainer) -> None:
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

    # ProgressReporter
    container.register(ProgressReporter, create_rich_progress_reporter)

    # AssetStore
    container.register(AssetStore, load_asset_store)

    # AssetDownloadManager
    container.register(AssetDownloadManager, create_asset_download_manager)

    # Device
    container.register(Device, determine_default_device)

    # ThreadPool
    container.register(ThreadPool, create_thread_pool)

    # CheckpointLoader
    container.register(CheckpointLoader, create_checkpoint_loader)

    # ModelSharder
    container.register(ModelSharder, create_model_sharder)

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
        JepaModel,
        JepaConfig,
        default_arch="base",
        factory=create_jepa_model,
        checkpoint_converter=convert_jepa_checkpoint,
    )

    register_jepa_configs(container)

    # JEPA Classifier
    register_model_family(
        container,
        JEPA_CLASSIFIER_FAMILY,
        JepaClassifierModel,
        JepaClassifierConfig,
        default_arch="base",
        factory=create_jepa_classifier_model,
    )

    register_jepa_classifier_configs(container)

    # LLaMA
    register_model_family(
        container,
        LLAMA_FAMILY,
        TransformerLM,
        LLaMAConfig,
        default_arch="llama3_1_8b",
        factory=create_llama_model,
        checkpoint_converter=convert_llama_checkpoint,
        shard_specs=get_llama_shard_specs,
        compiler=compile_transformer_lm,
        hugging_face_saver=save_as_hg_llama,
    )

    register_llama_configs(container)

    # Mistral
    register_model_family(
        container,
        MISTRAL_FAMILY,
        TransformerLM,
        MistralConfig,
        default_arch="7b",
        factory=create_mistral_model,
        checkpoint_converter=convert_mistral_checkpoint,
        compiler=compile_transformer_lm,
    )

    register_mistral_configs(container)

    # NLLB
    register_nllb_configs(container)

    # Qwen
    register_model_family(
        container,
        QWEN_FAMILY,
        TransformerLM,
        QwenConfig,
        default_arch="qwen25_7b",
        factory=create_qwen_model,
        checkpoint_converter=convert_qwen_checkpoint,
        shard_specs=get_qwen_shard_specs,
        compiler=compile_transformer_lm,
        hugging_face_saver=save_as_hg_qwen,
    )

    register_qwen_configs(container)

    # S2T Transformer
    register_model_family(
        container,
        S2T_TRANSFORMER_FAMILY,
        TransformerModel,
        S2TTransformerConfig,
        default_arch="7b",
        factory=create_s2t_transformer_model,
        checkpoint_converter=convert_s2t_transformer_checkpoint,
    )

    register_s2t_transformer_configs(container)

    # Transformer
    register_model_family(
        container,
        TRANSFORMER_FAMILY,
        TransformerModel,
        TransformerConfig,
        default_arch="base",
        factory=create_transformer_model,
        checkpoint_converter=convert_transformer_checkpoint,
    )

    register_transformer_configs(container)

    # w2v-BERT
    register_model_family(
        container,
        W2VBERT_FAMILY,
        W2VBertModel,
        W2VBertConfig,
        default_arch="300m",
        factory=create_w2vbert_model,
        checkpoint_converter=convert_w2vbert_checkpoint,
    )

    register_w2vbert_configs(container)

    # wav2vec 2.0
    register_model_family(
        container,
        WAV2VEC2_FAMILY,
        Wav2Vec2Model,
        Wav2Vec2Config,
        default_arch="base",
        factory=create_wav2vec2_model,
        checkpoint_converter=convert_wav2vec2_checkpoint,
    )

    register_wav2vec2_configs(container)

    # wav2vec 2.0 ASR
    register_model_family(
        container,
        WAV2VEC2_ASR_FAMILY,
        Wav2Vec2AsrModel,
        Wav2Vec2AsrConfig,
        default_arch="base_10h",
        factory=create_wav2vec2_asr_model,
        checkpoint_converter=convert_wav2vec2_asr_checkpoint,
    )

    register_wav2vec2_asr_configs(container)


def _register_tokenizer_families(container: DependencyContainer) -> None:
    # Char
    register_tokenizer_family(
        container,
        CHAR_TOKENIZER_FAMILY,
        loader=load_char_tokenizer,
    )

    # LLaMA
    register_tokenizer_family(
        container,
        LLAMA_FAMILY,
        loader=load_llama_tokenizer,
    )

    # Mistral
    register_tokenizer_family(
        container,
        MISTRAL_FAMILY,
        loader=load_mistral_tokenizer,
    )

    # Qwen
    register_tokenizer_family(
        container,
        QWEN_FAMILY,
        loader=load_qwen_tokenizer,
    )

    # NLLB
    register_tokenizer_family(
        container,
        NLLB_FAMILY,
        loader=load_nllb_tokenizer,
    )

    # S2T Transformer
    register_tokenizer_family(
        container,
        S2T_TRANSFORMER_FAMILY,
        loader=load_s2t_transformer_tokenizer,
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
                    entry_point.value, f"The '{entry_point.value}' entry point cannot be run as an extension function since its signature does not match `extension_function(register: DependencyContainer)`."  # fmt: skip
                ) from None

            log.warning("'{}' entry point is not a valid extension function. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.", entry_point.value)  # fmt: skip
        except Exception as ex:
            if should_trace:
                raise ExtensionError(
                    entry_point.value, f"'{entry_point.value}' extension function has failed. See the nested exception for details."  # fmt: skip
                ) from ex

            log.warning("'{}' extension function failed. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.", entry_point.value)  # fmt: skip

        if should_trace:
            log.info("`{}` extension function run successfully.", entry_point.value)  # fmt: skip


class ExtensionError(Exception):
    entry_point: str

    def __init__(self, entry_point: str, message: str) -> None:
        super().__init__(message)

        self.entry_point = entry_point
