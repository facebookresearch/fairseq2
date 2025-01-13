# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.assets import InProcAssetDownloadManager, default_asset_store
from fairseq2.context import RuntimeContext, set_runtime_context
from fairseq2.extensions import run_extensions
from fairseq2.setup.assets import _register_assets
from fairseq2.setup.chatbots import _register_chatbots
from fairseq2.setup.clusters import _register_clusters
from fairseq2.setup.config import _register_config_sections
from fairseq2.setup.datasets import _register_datasets
from fairseq2.setup.generation import (
    _register_beam_search_algorithms,
    _register_samplers,
    _register_seq2seq_generators,
    _register_seq_generators,
)
from fairseq2.setup.optim import _register_lr_schedulers, _register_optimizers
from fairseq2.setup.text_tokenizers import _register_text_tokenizers

_setup_called: bool = False


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
    global _setup_called

    if _setup_called:
        return

    _setup_called = True  # Mark as called to avoid recursive calls.

    context = setup_runtime_context()

    set_runtime_context(context)

    run_extensions("fairseq2")  # compat


def setup_runtime_context() -> RuntimeContext:
    asset_download_manager = InProcAssetDownloadManager()

    context = RuntimeContext(default_asset_store, asset_download_manager)

    _register_assets(context)
    _register_beam_search_algorithms(context)
    _register_chatbots(context)
    _register_clusters(context)
    _register_config_sections(context)
    _register_datasets(context)
    _register_lr_schedulers(context)
    _register_optimizers(context)
    _register_samplers(context)
    _register_seq2seq_generators(context)
    _register_seq_generators(context)
    _register_text_tokenizers(context)

    run_extensions("fairseq2.extension", context)

    return context
