# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


class UnknownTokenizerError(Exception):
    tokenizer_name: str

    def __init__(self, tokenizer_name: str) -> None:
        super().__init__(f"'{tokenizer_name}' is not a known tokenizer.")

        self.tokenizer_name = tokenizer_name


class UnknownTokenizerFamilyError(Exception):
    family: str

    def __init__(self, family: str) -> None:
        super().__init__(f"'{family}' is not a know tokenizer family.")

        self.family = family


class TokenizerLoadError(Exception):
    tokenizer_name: str

    def __init__(self, tokenizer_name: str, message: str) -> None:
        super().__init__(message)

        self.tokenizer_name = tokenizer_name


class TokenizerConfigLoadError(Exception):
    tokenizer_name: str

    def __init__(self, tokenizer_name: str, message: str) -> None:
        super().__init__(message)

        self.tokenizer_name = tokenizer_name
