# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class StaticBatching:
    """Specifies batching where each batch has the same number of examples."""

    batch_size: int
    """The number of examples in each batch."""


@dataclass
class LengthBatching:
    """Specifies batching where each batch has a maximum number of elements."""

    max_num_elements: int
    """The maximum number of elements (e.g. tokens) in each batch."""
