# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Module


class UnknownModelError(Exception):
    model_name: str

    def __init__(self, model_name: str) -> None:
        super().__init__(f"'{model_name}' is not a known model.")

        self.model_name = model_name


class UnknownModelFamilyError(Exception):
    family: str
    model_name: str

    def __init__(self, family: str, model_name: str) -> None:
        super().__init__(f"The '{model_name}' model has an unknown family '{family}'")

        self.family = family
        self.model_name = model_name


class UnknownModelArchitectureError(Exception):
    arch: str
    family: str
    model_name: str

    def __init__(self, arch: str, family: str, model_name: str) -> None:
        super().__init__(
            f"The '{family}' family of the '{model_name}' model does not have an architecture named '{arch}'."
        )

        self.arch = arch
        self.family = family
        self.model_name = model_name


class ModelLoadError(Exception):
    model_name: str

    def __init__(self, model_name: str, message: str) -> None:
        super().__init__(message)

        self.model_name = model_name


def model_asset_card_error(name: str) -> ModelLoadError:
    return ModelLoadError(
        name, f"The '{name}' asset card cannot be read. See the nested exception for details."  # fmt: skip
    )


class ShardedModelLoadError(ModelLoadError):
    num_shards: int
    tp_size: int

    def __init__(self, model_name: str, num_shards: int, tp_size: int) -> None:
        super().__init__(
            model_name, f"The number of checkpoint shards of the '{model_name}' model is expected to match the tensor parallel size ({tp_size}), but is {num_shards} instead."  # fmt: skip
        )

        self.num_shards = num_shards
        self.tp_size = tp_size


class ModelConfigLoadError(Exception):
    model_name: str

    def __init__(self, model_name: str, message: str) -> None:
        super().__init__(message)

        self.model_name = model_name


class InvalidModelTypeError(Exception):
    model_name: str
    kls: type[Module]
    expected_kls: type[Module]

    def __init__(
        self,
        model_name: str,
        kls: type[Module],
        expected_kls: type[Module],
    ) -> None:
        super().__init__(
            f"The '{model_name}' model is expected to be of type `{expected_kls}`, but is of type `{kls}` instead."
        )

        self.model_name = model_name
        self.kls = kls
        self.expected_kls = expected_kls


class InvalidModelConfigTypeError(Exception):
    model_name: str
    kls: type[object]
    expected_kls: type[object]

    def __init__(
        self, model_name: str, kls: type[object], expected_kls: type[object]
    ) -> None:
        super().__init__(
            f"The '{model_name}' model configuration is expected to be of type `{expected_kls}`, but is of type `{kls}` instead."
        )

        self.model_name = model_name
        self.kls = kls
        self.expected_kls = expected_kls
