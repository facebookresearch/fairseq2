# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from typing import Mapping

from fairseq2.assets import (
    AssetDirectories,
    AssetDownloadManager,
    CompositeAssetDownloadManager,
    HuggingFaceHub,
    InProcAssetDownloadManager,
    NoopAssetDownloadManager,
    StandardAssetStore,
)
from fairseq2.context import RuntimeContext
from fairseq2.extensions import run_extensions
from fairseq2.file_system import FileSystem, LocalFileSystem
from fairseq2.utils.progress import NoopProgressReporter, ProgressReporter

# isort: split

from fairseq2.setup._asset import _register_assets
from fairseq2.setup._chatbots import _register_chatbots
from fairseq2.setup._cluster import _register_clusters
from fairseq2.setup._datasets import _register_dataset_families
from fairseq2.setup._generation import (
    _register_beam_search_algorithms,
    _register_samplers,
    _register_seq2seq_generators,
    _register_seq_generators,
)
from fairseq2.setup._lr_schedulers import _register_lr_schedulers
from fairseq2.setup._metric_recorders import _register_metric_recorders
from fairseq2.setup._metrics import _register_metric_descriptors
from fairseq2.setup._models import _register_model_families
from fairseq2.setup._optim import _register_optimizers
from fairseq2.setup._po_finetune_units import (
    _register_online_finetune_units,
    _register_po_finetune_units,
)
from fairseq2.setup._profilers import _register_profilers
from fairseq2.setup._recipes import _register_recipes
from fairseq2.setup._text_tokenizers import _register_text_tokenizer_families


def setup_library(progress_reporter: ProgressReporter | None = None) -> RuntimeContext:
    env = os.environ

    file_system = LocalFileSystem()

    asset_store = StandardAssetStore()

    asset_download_manager = _create_asset_download_manager(env, file_system)

    if progress_reporter is None:
        progress_reporter = NoopProgressReporter()

    context = RuntimeContext(
        env, asset_store, asset_download_manager, file_system, progress_reporter
    )

    context.wall_watch.start()

    _register_assets(context)
    _register_beam_search_algorithms(context)
    _register_chatbots(context)
    _register_clusters(context)
    _register_dataset_families(context)
    _register_lr_schedulers(context)
    _register_metric_descriptors(context)
    _register_metric_recorders(context)
    _register_model_families(context)
    _register_optimizers(context)
    _register_po_finetune_units(context)
    _register_online_finetune_units(context)
    _register_profilers(context)
    _register_recipes(context)
    _register_samplers(context)
    _register_seq2seq_generators(context)
    _register_seq_generators(context)
    _register_text_tokenizer_families(context)

    signature = "extension_function(context: RuntimeContext) -> None"

    run_extensions("fairseq2.extension", signature, context)

    return context


def _create_asset_download_manager(
    env: Mapping[str, str], file_system: FileSystem
) -> AssetDownloadManager:
    asset_dirs = AssetDirectories(env, file_system)

    asset_cache_dir = asset_dirs.get_cache_dir()

    noop_download_manager = NoopAssetDownloadManager()

    in_proc_download_manager = InProcAssetDownloadManager(asset_cache_dir)

    hg_download_manager = HuggingFaceHub()

    return CompositeAssetDownloadManager(
        {
            "file": noop_download_manager,
            "https": in_proc_download_manager,
            "hg": hg_download_manager,
        }
    )
