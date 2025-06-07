# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from errno import ENOENT
from os import strerror
from pathlib import Path
from typing import Any, Protocol, TypeVar, final

from typing_extensions import override

from fairseq2.assets import AssetCard, AssetConfigLoader, AssetDownloadManager
from fairseq2.data.tokenizers.error import TokenizerLoadError
from fairseq2.data.tokenizers.tokenizer import Tokenizer
from fairseq2.error import ContractError, InfraError
from fairseq2.file_system import FileSystem
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver
from fairseq2.utils.structured import StructureError, ValueConverter


class TokenizerFamilyHandler(ABC):
    @abstractmethod
    def load_tokenizer_config(self, card: AssetCard) -> object: ...

    @abstractmethod
    def load_tokenizer(
        self, resolver: DependencyResolver, card: AssetCard, *, config: object = None
    ) -> Tokenizer: ...

    @abstractmethod
    def load_tokenizer_from_path(
        self, resolver: DependencyResolver, path: Path, config: object
    ) -> Tokenizer: ...

    @property
    @abstractmethod
    def family(self) -> str: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


TokenizerConfigT_contra = TypeVar("TokenizerConfigT_contra", contravariant=True)


class TokenizerLoader(Protocol[TokenizerConfigT_contra]):
    def __call__(
        self,
        resolver: DependencyResolver,
        path: Path,
        name: str,
        config: TokenizerConfigT_contra,
    ) -> Tokenizer: ...


TokenizerConfigT = TypeVar("TokenizerConfigT")


@final
class StandardTokenizerFamilyHandler(TokenizerFamilyHandler):
    _family: str
    _config_kls: type[object]
    _loader: TokenizerLoader[Any]
    _file_system: FileSystem
    _asset_download_manager: AssetDownloadManager
    _config_loader: AssetConfigLoader

    def __init__(
        self,
        family: str,
        config_kls: type[TokenizerConfigT],
        loader: TokenizerLoader[TokenizerConfigT],
        file_system: FileSystem,
        asset_download_manager: AssetDownloadManager,
        config_loader: AssetConfigLoader,
    ) -> None:
        self._family = family
        self._config_kls = config_kls
        self._loader = loader
        self._file_system = file_system
        self._asset_download_manager = asset_download_manager
        self._config_loader = config_loader

    @override
    def load_tokenizer_config(self, card: AssetCard) -> object:
        try:
            default_config = self._config_kls()
        except TypeError as ex:
            raise ContractError(
                f"The default configuration of the '{self._family}' tokenizer family cannot be constructed. See the nested exception for details."
            ) from ex

        # Override the default configuration if the asset card or its bases have
        # a 'tokenizer_config' field.
        try:
            return self._config_loader.load(
                card, default_config, config_key="tokenizer_config"
            )
        except StructureError as ex:
            raise ContractError(
                f"The configuration class of the '{self._family}' tokenizer family cannot be unstructured. See the nested exception for details."
            ) from ex

    @override
    def load_tokenizer(
        self, resolver: DependencyResolver, card: AssetCard, *, config: object = None
    ) -> Tokenizer:
        name = card.name

        uri = card.field("tokenizer").as_uri()

        path = self._asset_download_manager.download_tokenizer(uri, name)

        # Load the configuration.
        if config is None:
            config = self.load_tokenizer_config(card)

            has_custom_config = False
        else:
            if not isinstance(config, self._config_kls):
                raise TypeError(
                    f"`config` is expected to be of type `{self._config_kls}`, but is of type `{type(config)}` instead."
                )

            has_custom_config = True

        try:
            return self._do_load_tokenizer(resolver, path, name, config)
        except ValueError as ex:
            if has_custom_config:
                raise

            raise TokenizerLoadError(
                name, f"The '{name}' tokenizer does not have a valid configuration. See the nested exception for details."  # fmt: skip
            ) from ex
        except FileNotFoundError as ex:
            raise TokenizerLoadError(
                name, f"The '{name}' tokenizer cannot be found at the '{path}' path."  # fmt: skip
            ) from ex

    @override
    def load_tokenizer_from_path(
        self, resolver: DependencyResolver, path: Path, config: object
    ) -> Tokenizer:
        if not isinstance(config, self._config_kls):
            raise TypeError(
                f"`config` is expected to be of type `{self._config_kls}`, but is of type `{type(config)}` instead."
            )

        name = str(path)

        return self._do_load_tokenizer(resolver, path, name, config)

    def _do_load_tokenizer(
        self, resolver: DependencyResolver, path: Path, name: str, config: object
    ) -> Tokenizer:
        try:
            path_exists = self._file_system.exists(path)
        except OSError as ex:
            raise InfraError(
                f"A system error has occurred while accessing the '{path}' path of the dataset. See the nested exception for details."
            ) from ex

        if not path_exists:
            raise FileNotFoundError(ENOENT, strerror(ENOENT), path)

        return self._loader(resolver, path, name, config)

    @property
    @override
    def family(self) -> str:
        return self._family

    @property
    @override
    def config_kls(self) -> type[object]:
        return self._config_kls


def register_tokenizer_family(
    container: DependencyContainer,
    family: str,
    config_kls: type[TokenizerConfigT],
    loader: TokenizerLoader[TokenizerConfigT],
) -> None:
    def create_handler(resolver: DependencyResolver) -> TokenizerFamilyHandler:
        value_converter = resolver.resolve(ValueConverter)

        file_system = resolver.resolve(FileSystem)

        asset_download_manager = resolver.resolve(AssetDownloadManager)

        config_loader = AssetConfigLoader(value_converter)

        return StandardTokenizerFamilyHandler(
            family,
            config_kls,
            loader,
            file_system,
            asset_download_manager,
            config_loader,
        )

    container.register(TokenizerFamilyHandler, create_handler, key=family)
