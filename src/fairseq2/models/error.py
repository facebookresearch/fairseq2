# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


class UnknownModelError(Exception):
    model_name: str

    def __init__(self, model_name: str) -> None:
        super().__init__(f"'{model_name}' is not a known model.")

        self.model_name = model_name


class UnknownModelFamilyError(Exception):
    family: str

    def __init__(self, family: str) -> None:
        super().__init__(f"'{family}' is not a known model family.")

        self.family = family


class UnknownModelArchitectureError(Exception):
    arch: str
    family: str

    def __init__(self, arch: str, family: str) -> None:
        super().__init__(
            f"The '{family}' model family does not have an architecture named '{arch}'."
        )

        self.arch = arch
        self.family = family


class ModelLoadError(Exception):
    model_name: str

    def __init__(self, model_name: str, message: str) -> None:
        super().__init__(message)

        self.model_name = model_name


class ModelConfigLoadError(Exception):
    model_name: str

    def __init__(self, model_name: str, message: str) -> None:
        super().__init__(message)

        self.model_name = model_name
