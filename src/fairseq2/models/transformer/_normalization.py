# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Protocol

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn import LayerNorm, StandardLayerNorm


class LayerNormFactory(Protocol):
    """Creates :class:`LayerNorm` instances."""

    def __call__(
        self,
        model_dim: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> LayerNorm:
        """
        :param model_dim: The dimensionality of the model.
        :param device: The device on which to initialize the module.
        :param dtype: The data type of the module.
        """


def create_standard_layer_norm(
    model_dim: int, *, device: Device | None = None, dtype: DataType | None = None
) -> LayerNorm:
    """Creates a :class:`StandardLayerNorm` instance."""
    return StandardLayerNorm(model_dim, bias=True, device=device, dtype=dtype)
