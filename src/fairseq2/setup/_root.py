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
    InProcAssetDownloadManager,
    StandardAssetStore,
)
from fairseq2.context import RuntimeContext
from fairseq2.extensions import run_extensions
from fairseq2.file_system import FileSystem, GlobalFileSystem, register_filesystems
from fairseq2.utils.progress import NoopProgressReporter, ProgressReporter

# isort: split

from fairseq2.setup._asset import register_assets
from fairseq2.setup._chatbots import register_chatbots
from fairseq2.setup._cluster import register_clusters
from fairseq2.setup._datasets import register_dataset_families
from fairseq2.setup._generation import (
    register_beam_search_algorithms,
    register_samplers,
    register_seq2seq_generators,
    register_seq_generators,
)
from fairseq2.setup._lr_schedulers import register_lr_schedulers
from fairseq2.setup._metric_recorders import register_metric_recorders
from fairseq2.setup._metrics import register_metric_descriptors
from fairseq2.setup._models import register_model_families
from fairseq2.setup._optim import register_optimizers
from fairseq2.setup._po_finetune_units import register_po_finetune_units
from fairseq2.setup._profilers import register_profilers
from fairseq2.setup._recipes import register_recipes
from fairseq2.setup._text_tokenizers import register_text_tokenizer_families


def setup_library(progress_reporter: ProgressReporter | None = None) -> RuntimeContext:
    env = os.environ

    file_system = GlobalFileSystem()

    asset_store = StandardAssetStore()

    asset_download_manager = _create_asset_download_manager(env, file_system)

    if progress_reporter is None:
        progress_reporter = NoopProgressReporter()

    context = RuntimeContext(
        env, asset_store, asset_download_manager, file_system, progress_reporter
    )

    context.wall_watch.start()

    register_assets(context)
    register_beam_search_algorithms(context)
    register_chatbots(context)
    register_clusters(context)
    register_dataset_families(context)
    register_lr_schedulers(context)
    register_metric_descriptors(context)
    register_metric_recorders(context)
    register_model_families(context)
    register_optimizers(context)
    register_po_finetune_units(context)
    register_profilers(context)
    register_recipes(context)
    register_samplers(context)
    register_seq2seq_generators(context)
    register_seq_generators(context)
    register_text_tokenizer_families(context)
    register_filesystems(context)

    signature = "extension_function(context: RuntimeContext) -> None"

    run_extensions("fairseq2.extension", signature, context)

    return context


def _create_asset_download_manager(
    env: Mapping[str, str], file_system: FileSystem
) -> AssetDownloadManager:
    asset_dirs = AssetDirectories(env, file_system)

    asset_cache_dir = asset_dirs.get_cache_dir()

    return InProcAssetDownloadManager(asset_cache_dir)
