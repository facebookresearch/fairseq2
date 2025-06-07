# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import final
from urllib.parse import ParseResult, urlparse, urlunparse

from fairseq2.error import InternalError


@final
class Uri:
    @staticmethod
    def maybe_parse(s: str) -> Uri | None:
        try:
            result = urlparse(s)
        except ValueError:
            return None

        if not result.scheme:
            return None

        return Uri(result)

    @staticmethod
    def from_path(path: Path) -> Uri:
        s = path.as_uri()

        uri = Uri.maybe_parse(s)
        if uri is None:
            raise InternalError("`path.as_uri()` cannot be parsed as URI.")

        return uri

    def __init__(self, _result: ParseResult) -> None:
        self._result = _result

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
    def query(self) -> str:
        return self._result.query

    @property
    def fragment(self) -> str:
        return self._result.fragment

    def __str__(self) -> str:
        return urlunparse(self._result)
