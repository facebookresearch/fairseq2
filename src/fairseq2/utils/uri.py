# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import final
from urllib.parse import ParseResult, unquote, urlparse, urlunparse

from fairseq2.error import InternalError, NotSupportedError


@final
class Uri:
    """Represents a Uniform Resource Identifier."""

    @classmethod
    def parse(cls, s: str) -> Uri:
        try:
            result = urlparse(s)
        except ValueError as ex:
            raise UriFormatError(f"{s} cannot be parsed as URI.") from ex

        if not result.scheme:
            raise UriFormatError(f"{s} does not have a URI scheme.")

        params = cls._parse_params(s, result.params)

        return Uri(result, params)

    @staticmethod
    def _parse_params(s: str, params: str) -> dict[str, str]:
        output: dict[str, str] = {}

        params = params.strip()
        if not params:
            return output

        def unquote_and_strip(idx: int, p: str) -> str:
            try:
                p = unquote(p)
            except (UnicodeEncodeError, ValueError) as ex:
                raise UriFormatError(
                    f"Path parameters of {s} are expected to be valid quoted URI strings, but parameter at index {idx} cannot be unquoted."
                ) from ex

            return p.strip()

        pairs = params.split(";")

        for idx, param in enumerate(pairs):
            kv = param.split("=")
            if len(kv) != 2:
                raise UriFormatError(
                    f"Path parameters of {s} are expected to be semi-colon separated key-value pairs, but parameter at index {idx} is {param}."
                )

            key, value = kv

            key = unquote_and_strip(idx, key)
            if not key:
                raise UriFormatError(
                    f"Path parameter keys of {s} are expected to be non-empty, but parameter key at index {idx} is empty."
                )

            output[key] = unquote_and_strip(idx, value)

        return output

    @staticmethod
    def maybe_parse(s: str) -> Uri | None:
        try:
            return Uri.parse(s)
        except UriFormatError:
            return None

    @staticmethod
    def from_path(path: Path) -> Uri:
        s = path.as_uri()

        uri = Uri.maybe_parse(s)
        if uri is None:
            raise InternalError("`path.as_uri()` cannot be parsed as URI.")

        return uri

    def __init__(self, _result: ParseResult, _params: dict[str, str]) -> None:
        self._result = _result
        self._params = _params

    @property
    def scheme(self) -> str:
        return self._result.scheme

    @property
    def netloc(self) -> str:
        return self._result.netloc

    @property
    def path(self) -> str:
        return self._result.path

    @property
    def params(self) -> str:
        return self._result.params

    @property
    def parsed_params(self) -> Mapping[str, str]:
        return self._params

    @property
    def query(self) -> str:
        return self._result.query

    @property
    def fragment(self) -> str:
        return self._result.fragment

    def strip_params(self) -> Uri:
        """Returns a copy of this URI with path parameters removed."""
        result = self._result._replace(params="")

        return Uri(result, {})

    def to_path(self) -> Path:
        if self.scheme != "file":
            raise NotSupportedError(
                "`to_path()` is only supported for URIs with file scheme."
            )

        s = str(self)

        return Path(s[7:])

    def __str__(self) -> str:
        return urlunparse(self._result)


class UriFormatError(Exception):
    pass
