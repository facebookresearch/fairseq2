# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


class UnknownTextTokenizerError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known text tokenizer.")

        self.name = name


class UnknownTextTokenizerFamilyError(Exception):
    family: str
    tokenizer_name: str | None

    def __init__(self, family: str, tokenizer_name: str | None = None) -> None:
        super().__init__(f"'{family}' is not a known text tokenizer family.")

        self.family = family
        self.tokenizer_name = tokenizer_name


class TextTokenizerLoadError(Exception):
    name: str

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)

        self.name = name
