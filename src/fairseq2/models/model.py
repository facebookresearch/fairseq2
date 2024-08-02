# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from mypy_extensions import DefaultNamedArg
from torch.nn import Module
from typing_extensions import Self

from fairseq2.factory_registry import ConfigBoundFactoryRegistry
from fairseq2.typing import DataType, Device


class Model(Module):
    """Represents a machine learning model."""

    _family: Optional[str]

    def __init__(self) -> None:
        super().__init__()

        self._family = None

    def set_family(self, family: str) -> Self:
        """Set the family of the model."""
        if self._family is not None:
            raise ValueError(
                f"The model must not have a prior family, but has already '{self._family}'."
            )

        self._family = family

        return self

    @property
    def family(self) -> Optional[str]:
        """The family of the model."""
        return self._family


if TYPE_CHECKING:  # compat: remove when Python 3.9 support is dropped.
    model_factories = ConfigBoundFactoryRegistry[
        [DefaultNamedArg(Device, "device"), DefaultNamedArg(DataType, "dtype")], Module
    ]()
else:
    model_factories = ConfigBoundFactoryRegistry()
