# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from enum import Enum
from typing import Mapping

from fairseq2.assets import (
    AssetDirectories,
    AssetDownloadManager,
    InProcAssetDownloadManager,
    StandardAssetStore,
)
from fairseq2.context import RuntimeContext, set_runtime_context
from fairseq2.extensions import run_extensions
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
from fairseq2.utils.file import FileSystem, LocalFileSystem


class _SetupState(Enum):
    NOT_CALLED = 0
    IN_CALL = 1
    CALLED = 2


_setup_state: _SetupState = _SetupState.NOT_CALLED


def setup_fairseq2() -> None:
    """
    Sets up fairseq2.

    As part of the initialization, this function also registers extensions
    with via setuptools' `entry-point`__ mechanism. See
    :doc:`/basics/runtime_extensions` for more information.

    .. important::

        This function must be called before using any of the fairseq2 APIs.

    .. __: https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    """
    global _setup_state

    if _setup_state == _SetupState.CALLED:
        return

    if _setup_state == _SetupState.IN_CALL:
        raise RuntimeError("`setup_fairseq2()` cannot be called recursively.")

    _setup_state = _SetupState.IN_CALL

    try:
        context = setup_library()
    except Exception:
        _setup_state = _SetupState.NOT_CALLED

        raise

    set_runtime_context(context)

    _setup_state = _SetupState.CALLED


def setup_library() -> RuntimeContext:
    env = os.environ

    file_system = LocalFileSystem()

    asset_store = StandardAssetStore()

    asset_download_manager = _create_asset_download_manager(env, file_system)

    context = RuntimeContext(env, asset_store, asset_download_manager, file_system)

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

    run_extensions("fairseq2.extension", context)

    return context


def _create_asset_download_manager(
    env: Mapping[str, str], file_system: FileSystem
) -> AssetDownloadManager:
    asset_dirs = AssetDirectories(env, file_system)

    asset_cache_dir = asset_dirs.get_cache_dir()

    return InProcAssetDownloadManager(asset_cache_dir)
