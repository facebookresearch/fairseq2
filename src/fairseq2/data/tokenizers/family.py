# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Final, Protocol, TypeVar, final

from typing_extensions import override

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetConfigLoader,
    AssetDownloadManager,
)
from fairseq2.data.tokenizers.tokenizer import Tokenizer
from fairseq2.error import InternalError, raise_operational_system_error
from fairseq2.file_system import FileSystem
from fairseq2.gang import GangError, Gangs, raise_operational_gang_error
from fairseq2.runtime.lookup import Lookup
from fairseq2.utils.validation import ObjectValidator, ValidationError


class TokenizerFamily(ABC):
    @abstractmethod
    def get_tokenizer_config(self, card: AssetCard) -> object: ...

    @abstractmethod
    def load_tokenizer(
        self, card: AssetCard, gangs: Gangs, config: object | None
    ) -> Tokenizer: ...

    @abstractmethod
    def load_custom_tokenizer(
        self, path: Path, config: object, gangs: Gangs
    ) -> Tokenizer: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def kls(self) -> type[object]: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


class TokenizerGatedError(Exception):
    def __init__(self, name: str, url: str | None) -> None:
        super().__init__(f"{name} is a gated tokenizer.")

        self.name = name
        self.url = url


class TokenizerModelError(Exception):
    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)

        self.path = path


def get_tokenizer_family(
    card: AssetCard, families: Lookup[TokenizerFamily]
) -> TokenizerFamily:
    family_name = card.field("tokenizer_family").as_(str)

    family = families.maybe_get(family_name)
    if family is None:
        msg = f"family field of the {card.name} asset card is expected to be a supported tokenizer family, but is {family_name} instead."

        raise AssetCardError(card.name, msg)

    return family


TokenizerConfigT_contra = TypeVar("TokenizerConfigT_contra", contravariant=True)


class TokenizerLoader(Protocol[TokenizerConfigT_contra]):
    def __call__(self, path: Path, config: TokenizerConfigT_contra) -> Tokenizer: ...


TokenizerT = TypeVar("TokenizerT", bound=Tokenizer)

TokenizerConfigT = TypeVar("TokenizerConfigT")


@final
class StandardTokenizerFamily(TokenizerFamily):
    _CONFIG_KEYS: Final = (
        "tokenizer_config_overrides",
        "tokenizer_config_override",
        "tokenizer_config",
    )

    def __init__(
        self,
        name: str,
        kls: type[TokenizerT],
        config_kls: type[TokenizerConfigT],
        loader: TokenizerLoader[TokenizerConfigT],
        file_system: FileSystem,
        validator: ObjectValidator,
        asset_download_manager: AssetDownloadManager,
        asset_config_loader: AssetConfigLoader,
    ) -> None:
        self._name = name
        self._kls: type[Tokenizer] = kls
        self._config_kls: type[object] = config_kls
        self._loader: TokenizerLoader[Any] = loader
        self._file_system = file_system
        self._validator = validator
        self._asset_download_manager = asset_download_manager
        self._asset_config_loader = asset_config_loader

    @override
    def get_tokenizer_config(self, card: AssetCard) -> object:
        try:
            base_config = self._config_kls()
        except TypeError as ex:
            raise InternalError(
                f"Default configuration of the {self._name} tokenizer family cannot be constructed."
            ) from ex

        name = card.name

        for key in self._CONFIG_KEYS:
            config = self._asset_config_loader.load(card, base_config, config_key=key)

            if config is not base_config:
                try:
                    self._validator.validate(config)
                except ValidationError as ex:
                    msg = f"{key} field of the {name} asset card is not a valid {self._name} tokenizer configuration."

                    raise AssetCardError(name, msg) from ex

                return config

        return base_config

    @override
    def load_tokenizer(
        self, card: AssetCard, gangs: Gangs, config: object | None
    ) -> Tokenizer:
        if config is None:
            config = self.get_tokenizer_config(card)
        else:
            if not isinstance(config, self._config_kls):
                raise TypeError(
                    f"`config` must be of type `{self._config_kls}`, but is of type `{type(config)}` instead."
                )

        name = card.name

        uri_field = card.maybe_get_field("tokenizer")
        if uri_field is None:
            uri_field = card.maybe_get_field("url")
            if uri_field is not None:
                url = uri_field.as_(str)
            else:
                url = None

            raise TokenizerGatedError(name, url)

        uri = uri_field.as_uri()

        if uri.scheme not in self._asset_download_manager.supported_schemes:
            msg = f"tokenizer URI scheme of the {name} asset card is expected to be a supported scheme, but is {uri.scheme} instead."

            raise AssetCardError(name, msg)

        try:
            if gangs.root.rank == 0:
                download_path = self._asset_download_manager.download_tokenizer(uri)

                gangs.root.barrier()
            else:
                gangs.root.barrier()

                download_path = self._asset_download_manager.download_tokenizer(
                    uri, local_only=True
                )
        except GangError as ex:
            raise_operational_gang_error(ex)

        sub_path_field = card.maybe_get_field("tokenizer_path")
        if sub_path_field is not None:
            sub_pathname = sub_path_field.as_(str)

            path = download_path.joinpath(sub_pathname)

            try:
                path = self._file_system.resolve(path)
            except OSError as ex:
                raise_operational_system_error(ex)

            if not path.is_relative_to(download_path):
                msg = f"tokenizer_path field of the {name} asset card points to a path that is not relative to the download directory."

                raise AssetCardError(name, msg)
        else:
            path = download_path

        try:
            return self._loader(path, config)
        except TokenizerModelError as ex:
            msg = f"Tokenizer model of the {name} asset card is erroneous."

            if uri.scheme != "file":
                msg = f"{msg} Make sure that it is downloaded correctly and, if not, delete your cached version at {path}."

            raise AssetCardError(name, msg) from ex
        except FileNotFoundError as ex:
            if uri.scheme != "file":
                raise_operational_system_error(ex)

            msg = f"{path} pointed to by the tokenizer field of the {name} asset card is not found."

            raise AssetCardError(name, msg)
        except OSError as ex:
            raise_operational_system_error(ex)

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

    @override
    def load_custom_tokenizer(
        self, path: Path, config: object, gangs: Gangs
    ) -> Tokenizer:
        if not isinstance(config, self._config_kls):
            raise TypeError(
                f"`config` must be of type `{self._config_kls}`, but is of type `{type(config)}` instead."
            )

        return self._loader(path, config)

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def kls(self) -> type[object]:
        return self._kls

    @property
    @override
    def config_kls(self) -> type[object]:
        return self._config_kls
