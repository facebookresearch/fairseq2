# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from enum import Enum

from fairseq2.assets import (
    AssetDirectories,
    InProcAssetDownloadManager,
    StandardAssetStore,
)
from fairseq2.assets.setup import register_assets
from fairseq2.chatbots.setup import register_chatbots
from fairseq2.cluster import register_clusters
from fairseq2.context import RuntimeContext, set_runtime_context
from fairseq2.data.text.tokenizers.setup import register_text_tokenizer_families
from fairseq2.datasets.setup import register_dataset_families
from fairseq2.extensions import run_extensions
from fairseq2.generation import (
    register_beam_search_algorithms,
    register_beam_search_seq_generators,
    register_samplers,
    register_sampling_seq_generators,
)
from fairseq2.metrics import register_metric_descriptors
from fairseq2.metrics.recorders import register_metric_recorders
from fairseq2.models.setup import register_model_families
from fairseq2.optim import register_optimizers
from fairseq2.optim.lr_scheduler import register_lr_schedulers
from fairseq2.profilers import register_profilers
from fairseq2.recipes.setup import register_recipes
from fairseq2.utils.file import LocalFileSystem


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
    asset_store = StandardAssetStore()

    file_system = LocalFileSystem()

    asset_dirs = AssetDirectories(os.environ, file_system)

    asset_cache_dir = asset_dirs.get_cache_dir()

    asset_download_manager = InProcAssetDownloadManager(asset_cache_dir)

    context = RuntimeContext(asset_store, asset_download_manager, file_system)

    register_assets(context)
    register_beam_search_algorithms(context)
    register_beam_search_seq_generators(context)
    register_chatbots(context)
    register_clusters(context)
    register_dataset_families(context)
    register_lr_schedulers(context)
    register_metric_descriptors(context)
    register_metric_recorders(context)
    register_model_families(context)
    register_optimizers(context)
    register_profilers(context)
    register_recipes(context)
    register_samplers(context)
    register_sampling_seq_generators(context)
    register_text_tokenizer_families(context)

    run_extensions("fairseq2.extension", context)

    return context
