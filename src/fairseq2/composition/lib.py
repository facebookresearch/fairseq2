# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from rich.console import Console

from fairseq2.assets import AssetConfigLoader, StandardAssetConfigLoader
from fairseq2.cluster import (
    ClusterHandler,
    SlurmHandler,
    _ClusterResolver,
    _StandardClusterResolver,
)
from fairseq2.composition.assets import _register_asset
from fairseq2.composition.datasets import _register_dataset_families
from fairseq2.composition.extensions import _register_extensions
from fairseq2.composition.models import _register_model_families
from fairseq2.composition.tokenizers import _register_tokenizer_families
from fairseq2.data.tokenizers.hub import GlobalTokenizerLoader
from fairseq2.data_type import (
    DataTypeContext,
    _DataTypeModeStack,
    _StandardDataTypeContext,
    _tensor_constructors,
)
from fairseq2.device import DeviceContext, _StandardDeviceContext
from fairseq2.file_system import FileSystem, LocalFileSystem
from fairseq2.gang import GangContext, _GangManager, _StandardGangContext
from fairseq2.io import (
    SafetensorsLoader,
    TensorFileDumper,
    TensorFileLoader,
    _HuggingFaceSafetensorsLoader,
    _TorchTensorFileDumper,
    _TorchTensorFileLoader,
)
from fairseq2.model_checkpoint import (
    ModelCheckpointLoader,
    _BasicModelCheckpointLoader,
    _DelegatingModelCheckpointLoader,
    _NativeModelCheckpointLoader,
    _SafetensorsCheckpointLoader,
)
from fairseq2.models.hub import GlobalModelLoader
from fairseq2.models.llama4.sharder import MoESharder
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyResolver,
    wire_object,
)
from fairseq2.sharder import (
    EmbeddingSharder,
    LinearSharder,
    ModelSharder,
    ModuleSharder,
    StandardModelSharder,
)
from fairseq2.utils.config import (
    ConfigDirective,
    ConfigMerger,
    ConfigProcessor,
    ReplaceEnvDirective,
    StandardConfigMerger,
    StandardConfigProcessor,
)
from fairseq2.utils.env import Environment, StandardEnvironment, maybe_get_rank
from fairseq2.utils.process import ProcessRunner, StandardProcessRunner
from fairseq2.utils.progress import NOOP_PROGRESS_REPORTER, ProgressReporter
from fairseq2.utils.rich import (
    RichProgressReporter,
    _create_rich_download_progress_columns,
    get_error_console,
)
from fairseq2.utils.structured import StandardValueConverter, ValueConverter
from fairseq2.utils.threading import ThreadLocalStorage, _StandardThreadLocalStorage
from fairseq2.utils.validation import ObjectValidator, StandardObjectValidator
from fairseq2.utils.yaml import (
    RuamelYamlDumper,
    RuamelYamlLoader,
    YamlDumper,
    YamlLoader,
)


def _register_library(
    container: DependencyContainer, *, no_progress: bool | None = None
) -> None:
    # Environment
    env = StandardEnvironment()

    container.register_instance(Environment, env)

    # Console
    error_console = get_error_console()

    container.register_instance(Console, error_console)

    # ProgressReporter
    if no_progress is None:
        no_progress = env.has("FAIRSEQ2_NO_PROGRESS")

    if not no_progress:
        rank = maybe_get_rank(env)
        if rank is not None and rank != 0:
            no_progress = True

    if no_progress:
        container.register_instance(ProgressReporter, NOOP_PROGRESS_REPORTER)

        container.register_instance(
            ProgressReporter, NOOP_PROGRESS_REPORTER, key="download_reporter"
        )
    else:
        container.register_type(ProgressReporter, RichProgressReporter, singleton=True)

        def create_download_progress_reporter(
            resolver: DependencyResolver,
        ) -> ProgressReporter:
            columns = _create_rich_download_progress_columns()

            return wire_object(resolver, RichProgressReporter, columns=columns)

        container.register(
            ProgressReporter,
            create_download_progress_reporter,
            key="download_reporter",
            singleton=True,
        )

    # DataTypeModeStack
    def create_dtype_mode_stack(resolver: DependencyResolver) -> _DataTypeModeStack:
        constructors = _tensor_constructors()

        return wire_object(resolver, _DataTypeModeStack, constructors=constructors)

    container.register(_DataTypeModeStack, create_dtype_mode_stack)

    container.register_type(AssetConfigLoader, StandardAssetConfigLoader)
    container.register_type(_ClusterResolver, _StandardClusterResolver)
    container.register_type(ConfigMerger, StandardConfigMerger)
    container.register_type(ConfigProcessor, StandardConfigProcessor, singleton=True)
    container.register_type(DataTypeContext, _StandardDataTypeContext, singleton=True)
    container.register_type(DeviceContext, _StandardDeviceContext, singleton=True)
    container.register_type(FileSystem, LocalFileSystem, singleton=True)
    container.register_type(GangContext, _StandardGangContext, singleton=True)
    container.register_type(_GangManager, singleton=True)
    container.register_type(GlobalModelLoader, singleton=True)
    container.register_type(GlobalTokenizerLoader, singleton=True)
    container.register_type(ModelCheckpointLoader, _DelegatingModelCheckpointLoader, singleton=True)  # fmt: skip
    container.register_type(ModelSharder, StandardModelSharder, singleton=True)
    container.register_type(ObjectValidator, StandardObjectValidator, singleton=True)
    container.register_type(ProcessRunner, StandardProcessRunner)
    container.register_type(SafetensorsLoader, _HuggingFaceSafetensorsLoader)
    container.register_type(TensorFileDumper, _TorchTensorFileDumper, singleton=True)
    container.register_type(TensorFileLoader, _TorchTensorFileLoader, singleton=True)
    container.register_type(ThreadLocalStorage, _StandardThreadLocalStorage)
    container.register_type(ValueConverter, StandardValueConverter, singleton=True)
    container.register_type(YamlDumper, RuamelYamlDumper, singleton=True)
    container.register_type(YamlLoader, RuamelYamlLoader, singleton=True)

    container.collection.register_type(ClusterHandler, SlurmHandler)

    container.collection.register_type(ModuleSharder, EmbeddingSharder)
    container.collection.register_type(ModuleSharder, LinearSharder)
    container.collection.register_type(ModuleSharder, MoESharder)

    container.collection.register_type(
        ModelCheckpointLoader, _BasicModelCheckpointLoader
    )
    container.collection.register_type(
        ModelCheckpointLoader, _NativeModelCheckpointLoader
    )
    container.collection.register_type(
        ModelCheckpointLoader, _SafetensorsCheckpointLoader
    )

    container.collection.register_type(ConfigDirective, ReplaceEnvDirective)

    _register_asset(container)

    # Asset Families
    _register_model_families(container)
    _register_dataset_families(container)
    _register_tokenizer_families(container)

    # Extensions
    _register_extensions(container)
