# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import final

from typing_extensions import override

from fairseq2.file_system import FileSystem
from fairseq2.logging import log
from fairseq2.utils.env import Environment


class _AssetDirectoryAccessor(ABC):
    @abstractmethod
    def maybe_get_system_dir(self) -> Path | None:
        """
        :raises AssetPathVariableError:
        :raises AssetDirectoryError:
        """

    @abstractmethod
    def maybe_get_user_dir(self) -> Path | None:
        """
        :raises AssetPathVariableError:
        :raises AssetDirectoryError:
        """

    @abstractmethod
    def get_cache_dir(self) -> Path:
        """
        :raises AssetPathVariableError:
        :raises AssetDirectoryError:
        """


class AssetDirectoryError(Exception):
    pass


class AssetPathVariableError(AssetDirectoryError):
    def __init__(self, var_name: str, value: str) -> None:
        super().__init__(
            f"`{var_name}` environment variable is expected to be a pathname, but is '{value}' instead"
        )

        self.var_name = var_name
        self.value = value


@final
class _StandardAssetDirectoryAccessor(_AssetDirectoryAccessor):
    def __init__(self, env: Environment, file_system: FileSystem) -> None:
        self._env = env
        self._file_system = file_system

    @override
    def maybe_get_system_dir(self) -> Path | None:
        env_var_name = "FAIRSEQ2_ASSET_DIR"

        asset_dir = self._maybe_get_path_from_env(env_var_name)
        if asset_dir is not None:
            if not self._path_exists(asset_dir):
                log.warning("{} path pointed to by the `{}` environment variable is not found.", asset_dir, env_var_name)  # fmt: skip

                asset_dir = None

        if asset_dir is None:
            asset_dir = self._get_path("/etc/fairseq2/assets")

        if not self._path_exists(asset_dir):
            return None

        return asset_dir

    @override
    def maybe_get_user_dir(self) -> Path | None:
        env_var_name = "FAIRSEQ2_USER_ASSET_DIR"

        asset_dir = self._maybe_get_path_from_env(env_var_name)
        if asset_dir is not None:
            if not self._path_exists(asset_dir):
                log.warning("{} path pointed to by the `{}` environment variable is not found.", asset_dir, env_var_name)  # fmt: skip

                asset_dir = None

        if asset_dir is None:
            asset_dir = self._maybe_get_path_from_env(
                "XDG_CONFIG_HOME", sub_pathname="fairseq2/assets"
            )

            if asset_dir is None:
                asset_dir = self._get_path("~/.config/fairseq2/assets")

        if not self._path_exists(asset_dir):
            return None

        return asset_dir

    @override
    def get_cache_dir(self) -> Path:
        cache_dir = self._maybe_get_path_from_env("FAIRSEQ2_CACHE_DIR")
        if cache_dir is None:
            cache_dir = self._maybe_get_path_from_env(
                "XDG_CACHE_HOME", sub_pathname="fairseq2/assets"
            )

            if cache_dir is None:
                cache_dir = self._get_path("~/.cache/fairseq2/assets")

        return cache_dir

    def _maybe_get_path_from_env(
        self, var_name: str, sub_pathname: str | None = None
    ) -> Path | None:
        pathname = self._env.maybe_get(var_name)
        if pathname is None:
            return None

        try:
            path = Path(pathname)
        except ValueError:
            raise AssetPathVariableError(var_name, pathname) from None

        if sub_pathname is not None:
            path = path.joinpath(sub_pathname)

        return self._resolve_path(path)

    def _get_path(self, pathname: str) -> Path:
        path = Path(pathname)

        return self._resolve_path(path)

    def _path_exists(self, path: Path) -> bool:
        try:
            return self._file_system.exists(path)
        except OSError as ex:
            raise AssetDirectoryError(f"failed to access path '{path}'") from ex

    def _resolve_path(self, path: Path) -> Path:
        try:
            return self._file_system.resolve(path)
        except RuntimeError as ex:
            raise AssetDirectoryError(f"failed to access path '{path}'") from ex
