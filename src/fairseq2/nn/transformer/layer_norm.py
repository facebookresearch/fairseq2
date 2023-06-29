# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

from typing_extensions import TypeAlias

from fairseq2.nn.normalization import LayerNorm, StandardLayerNorm
from fairseq2.typing import DataType, Device

LayerNormFactory: TypeAlias = Callable[
    [int, Optional[Device], Optional[DataType]], LayerNorm
]


def create_default_layer_norm(
    model_dim: int, device: Optional[Device] = None, dtype: Optional[DataType] = None
) -> LayerNorm:
    """Constructs a standard Layer Normalization module."""
    return StandardLayerNorm(model_dim, device=device, dtype=dtype)
