# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

from torch.nn import Module

from fairseq2.error import NotSupportedError


class UnknownModelError(Exception):
    model_name: str

    def __init__(self, model_name: str) -> None:
        super().__init__(f"'{model_name}' is not a known model.")

        self.model_name = model_name


class UnknownModelFamilyError(Exception):
    family: str
    model_name: str | None

    def __init__(self, family: str, model_name: str | None = None) -> None:
        if model_name is None:
            message = f"'{family}' is not a known model family."
        else:
            message = f"The '{model_name}' model has an unknown family '{family}'"

        super().__init__(message)

        self.family = family
        self.model_name = model_name


class UnknownModelArchitectureError(Exception):
    family: str
    arch: str
    model_name: str | None

    def __init__(self, family: str, arch: str, model_name: str | None = None) -> None:
        if model_name is None:
            message = (
                f"'{arch}' is not a known architecture of the '{family}' model family."
            )
        else:
            message = f"The '{family}' family of the '{model_name}' model does not have an architecture named '{arch}'."

        super().__init__(message)

        self.family = family
        self.arch = arch


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
        message = f"The number of checkpoint shards of the '{model_name}' model is expected to match the tensor parallel size ({tp_size}), but is {num_shards} instead."

        super().__init__(model_name, message)

        self.num_shards = num_shards
        self.tp_size = tp_size


class ModelCheckpointNotFoundError(ModelLoadError):
    path: Path

    def __init__(self, model_name: str, path: Path) -> None:
        super().__init__(
            model_name,
            f"The '{model_name}' model has no checkpoint at the '{path}' path.",
        )

        self.path = path


class ModelConfigLoadError(Exception):
    model_name: str

    def __init__(self, model_name: str, message: str) -> None:
        super().__init__(message)

        self.model_name = model_name


class MetaDeviceNotSupportedError(NotSupportedError):
    family: str
    model_name: str | None

    def __init__(self, family: str, model_name: str | None = None) -> None:
        if model_name is None:
            message = f"The '{family}' model family does not support meta device initialization."
        else:
            message = f"The '{family}' family of the '{model_name}' model does not support meta device initialization."

        super().__init__(message)

        self.family = family
        self.model_name = model_name


class ModelParallelismNotSupportedError(NotSupportedError):
    family: str
    model_name: str | None

    def __init__(self, family: str, model_name: str | None = None) -> None:
        if model_name is None:
            message = (
                f"The '{family}' model family does not support non-data parallelism."
            )
        else:
            message = f"The '{family}' family of the '{model_name}' model does not support non-data parallelism."

        super().__init__(message)

        self.family = family
        self.model_name = model_name


class InvalidModelTypeError(Exception):
    kls: type[Module]
    expected_kls: type[Module]
    model_name: str | None

    def __init__(
        self,
        kls: type[Module],
        expected_kls: type[Module],
        model_name: str | None = None,
    ) -> None:
        if model_name is None:
            message = f"The model is expected to be of type `{expected_kls}`, but is of type `{kls}` instead."
        else:
            message = f"The '{model_name}' model is expected to be of type `{expected_kls}`, but is of type `{kls}` instead."

        super().__init__(message)

        self.kls = kls
        self.expected_kls = expected_kls
        self.model_name = model_name


class InvalidModelConfigTypeError(Exception):
    kls: type[object]
    expected_kls: type[object]
    model_name: str | None

    def __init__(
        self,
        kls: type[object],
        expected_kls: type[object],
        model_name: str | None = None,
    ) -> None:
        if model_name is None:
            message = f"The model configuration is expected to be of type `{expected_kls}`, but is of type `{kls}` instead."
        else:
            message = f"The '{model_name}' model configuration is expected to be of type `{expected_kls}`, but is of type `{kls}` instead."

        super().__init__(message)

        self.kls = kls
        self.expected_kls = expected_kls
        self.model_name = model_name
