# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from fairseq2.error import NotSupportedError


class UnknownModelError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known model.")

        self.name = name


class UnknownModelFamilyError(Exception):
    family: str
    model_name: str | None

    def __init__(self, family: str, model_name: str | None = None) -> None:
        super().__init__(f"'{family}' is not a known model family.")

        self.family = family
        self.model_name = model_name


class UnknownModelArchitectureError(Exception):
    family: str
    arch: str
    model_name: str | None

    def __init__(self, family: str, arch: str, model_name: str | None = None) -> None:
        super().__init__(
            f"'{arch}' is not a known architecture of the '{family}' model family."
        )

        self.family = family
        self.arch = arch
        self.model_name = model_name


class ModelLoadError(Exception):
    name: str

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)

        self.name = name


class MetaDeviceNotSupportedError(NotSupportedError):
    pass


class NonDataParallelismNotSupported(NotSupportedError):
    pass
