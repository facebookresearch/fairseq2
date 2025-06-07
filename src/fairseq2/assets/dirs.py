# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import final

from fairseq2.error import InfraError
from fairseq2.file_system import FileSystem
from fairseq2.logging import log
from fairseq2.utils.env import get_path_from_env


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

    def get_download_dir(self) -> Path:
        download_dir = self._get_path_from_env("FAIRSEQ2_DOWNLOAD_DIR")
        if download_dir is None:
            download_dir = self._get_path_from_env(
                "XDG_CACHE_HOME", sub_pathname="fairseq2/assets"
            )

            if download_dir is None:
                download_dir = self._get_path("~/.cache/fairseq2/assets")

        return download_dir

    def _get_path(self, pathname: str) -> Path:
        path = Path(pathname)

        try:
            return self._file_system.resolve(path)
        except OSError as ex:
            raise InfraError(
                f"A system error has occurred while resolving the '{path}' path. See the nested exception for details."
            ) from ex

    def _get_path_from_env(
        self, var_name: str, sub_pathname: str | None = None
    ) -> Path | None:
        path = get_path_from_env(self._env, var_name)
        if path is None:
            return None

        if sub_pathname is not None:
            path = path.joinpath(sub_pathname)

        try:
            return self._file_system.resolve(path)
        except OSError as ex:
            raise InfraError(
                f"A system error has occurred while resolving the '{path}' path. See the nested exception for details."
            ) from ex

    def _exists(self, path: Path) -> bool:
        try:
            return self._file_system.exists(path)
        except OSError as ex:
            raise InfraError(
                f"A system error has occurred while accessing the '{path}' path. See the nested exception for details."
            ) from ex

    def _warn_missing(self, var_name: str, path: Path) -> None:
        log.warning("The '{}' path pointed to by the `{}` environment variable does not exist.", path, var_name)  # fmt: skip
