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
from fairseq2.context import RuntimeContext, set_runtime_context
from fairseq2.extensions import run_extensions
from fairseq2.setup._assets import _register_assets
from fairseq2.setup._chatbots import _register_chatbots
from fairseq2.setup._clusters import _register_clusters
from fairseq2.setup._datasets import _register_datasets
from fairseq2.setup._generation import (
    _register_beam_search_algorithms,
    _register_samplers,
    _register_seq2seq_generators,
    _register_seq_generators,
)
from fairseq2.setup._metrics import (
    _register_metric_descriptors,
    _register_metric_recorders,
)
from fairseq2.setup._models import _register_models
from fairseq2.setup._optim import _register_lr_schedulers, _register_optimizers
from fairseq2.setup._profilers import _register_profilers
from fairseq2.setup._recipes import _register_recipes
from fairseq2.setup._text_tokenizers import _register_text_tokenizers
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

    _register_assets(context)
    _register_beam_search_algorithms(context)
    _register_chatbots(context)
    _register_clusters(context)
    _register_datasets(context)
    _register_lr_schedulers(context)
    _register_metric_descriptors(context)
    _register_metric_recorders(context)
    _register_models(context)
    _register_optimizers(context)
    _register_profilers(context)
    _register_recipes(context)
    _register_samplers(context)
    _register_seq2seq_generators(context)
    _register_seq_generators(context)
    _register_text_tokenizers(context)

    run_extensions("fairseq2.extension", context)

    return context
