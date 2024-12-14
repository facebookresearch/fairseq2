# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

from fairseq2.assets.store import StandardAssetStore
from fairseq2.extensions import run_extensions
from fairseq2.utils.env import get_path_from_env


def register_assets(store: StandardAssetStore) -> None:
    store.add_package_metadata_provider("fairseq2.assets.cards")

    _register_etc_dir_metadata_provider(store)

    _register_home_config_dir_metadata_provider(store)

    # Extensions
    run_extensions("register_fairseq2_assets", store)


def _register_etc_dir_metadata_provider(store: StandardAssetStore) -> None:
    asset_dir = get_path_from_env("FAIRSEQ2_ASSET_DIR")
    if asset_dir is None:
        asset_dir = Path("/etc/fairseq2/assets").resolve()
        if not asset_dir.exists():
            return

    store.add_file_metadata_provider(asset_dir)


def _register_home_config_dir_metadata_provider(store: StandardAssetStore) -> None:
    asset_dir = get_path_from_env("FAIRSEQ2_USER_ASSET_DIR")
    if asset_dir is None:
        asset_dir = get_path_from_env("XDG_CONFIG_HOME")
        if asset_dir is None:
            asset_dir = Path("~/.config").expanduser()

        asset_dir = asset_dir.joinpath("fairseq2/assets").resolve()
        if not asset_dir.exists():
            return

    store.add_file_metadata_provider(asset_dir, user=True)
