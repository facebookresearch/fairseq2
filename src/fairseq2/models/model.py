# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from torch.nn import Module
from typing_extensions import Self


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
