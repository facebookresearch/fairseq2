# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Protocol

from fairseq2.nn.normalization import LayerNorm, StandardLayerNorm
from fairseq2.typing import DataType, Device


class LayerNormFactory(Protocol):
    """Creates instances of :class:`LayerNorm`."""

    def __call__(
        self, model_dim: int, device: Optional[Device], dtype: Optional[DataType]
    ) -> LayerNorm:
        """
        :param model_dim:
            The dimensionality of the model.
        :param device:
            The device on which to initialize the module.
        :param dtype:
            The data type of the module.
        """


def create_default_layer_norm(
    model_dim: int, device: Optional[Device] = None, dtype: Optional[DataType] = None
) -> LayerNorm:
    """Create a :class:`StandardLayerNorm` instance."""
    return StandardLayerNorm(model_dim, device=device, dtype=dtype)
