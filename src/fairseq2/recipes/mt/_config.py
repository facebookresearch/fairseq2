# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class MTLossSection:
    label_smoothing: float = 0.1
    """The amount of label smoothing to apply while computing the loss."""
