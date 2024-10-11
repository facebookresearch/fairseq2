# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Module
from typing_extensions import Self


class Model(Module):
    """Represents a machine learning model."""

    _family: str | None

    def __init__(self) -> None:
        super().__init__()

        self._family = None

    def set_family(self, family: str) -> Self:
        """Set the family of the model."""
        self._family = family

        return self

    @property
    def family(self) -> str | None:
        """The family of the model."""
        return self._family
