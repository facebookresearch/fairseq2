# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import final

from fairseq2.assets._metadata_provider import AssetMetadataLoadError
from fairseq2.logging import log
from fairseq2.utils.env import InvalidEnvironmentVariableError, get_path_from_env
from fairseq2.utils.file import FileSystem


@final
class AssetDirectories:
    _env: Mapping[str, str]
    _file_system: FileSystem

    def __init__(self, env: Mapping[str, str], file_system: FileSystem) -> None:
        self._env = env
        self._file_system = file_system

    def get_system_dir(self) -> Path | None:
        var_name = "FAIRSEQ2_ASSET_DIR"

        asset_dir = self._get_path_from_env(var_name)
        if asset_dir is not None:
            dir_exists = self._exists(asset_dir)
            if not dir_exists:
                self._warn_missing(var_name, asset_dir)

                asset_dir = None

        if asset_dir is None:
            asset_dir = self._get_path("/etc/fairseq2/assets")

        dir_exists = self._exists(asset_dir)
        if not dir_exists:
            return None

        return asset_dir

    def get_user_dir(self) -> Path | None:
        var_name = "FAIRSEQ2_USER_ASSET_DIR"

        asset_dir = self._get_path_from_env(var_name)
        if asset_dir is not None:
            dir_exists = self._exists(asset_dir)
            if not dir_exists:
                self._warn_missing(var_name, asset_dir)

                asset_dir = None

        if asset_dir is None:
            asset_dir = self._get_path_from_env(
                "XDG_CONFIG_HOME", sub_pathname="fairseq2/assets"
            )

            if asset_dir is None:
                asset_dir = self._get_path("~/.config/fairseq2/assets")

        dir_exists = self._exists(asset_dir)
        if not dir_exists:
            return None

        return asset_dir

    def get_cache_dir(self) -> Path:
        cache_dir = self._get_path_from_env("FAIRSEQ2_CACHE_DIR")
        if cache_dir is None:
            cache_dir = self._get_path_from_env(
                "XDG_CACHE_HOME", sub_pathname="fairseq2/assets"
            )

            if cache_dir is None:
                cache_dir = self._get_path("~/.cache/fairseq2/assets")

        return cache_dir

    def _get_path(self, pathname: str) -> Path:
        path = Path(pathname)

        try:
            return self._file_system.resolve(path)
        except OSError as ex:
            raise AssetMetadataLoadError(
                f"The '{path}' path cannot be accessed. See the nested exception for details."
            ) from ex

    def _get_path_from_env(
        self, var_name: str, sub_pathname: str | None = None
    ) -> Path | None:
        try:
            path = get_path_from_env(self._env, var_name)
        except InvalidEnvironmentVariableError as ex:
            raise AssetMetadataLoadError(
                f"The `{var_name}` environment variable cannot be read as a pathname. See the nested exception for details."
            ) from ex

        if path is None:
            return None

        if sub_pathname is not None:
            path = path.joinpath(sub_pathname)

        try:
            return self._file_system.resolve(path)
        except OSError as ex:
            raise AssetMetadataLoadError(
                f"The '{path}' path cannot be accessed. See the nested exception for details."
            ) from ex

    def _exists(self, path: Path) -> bool:
        try:
            return self._file_system.exists(path)
        except OSError as ex:
            raise AssetMetadataLoadError(
                f"The '{path}' path cannot be accessed. See the nested exception for details."
            ) from ex

    @staticmethod
    def _warn_missing(var_name: str, path: Path) -> None:
        log.warning("The '{}' path pointed to by the `{}` environment variable does not exist.", path, var_name)  # fmt: skip
