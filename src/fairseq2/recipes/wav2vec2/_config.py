# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class Wav2Vec2LossSection:
    diversity_weight: float = 0.1
    """The weight of the diversity loss."""

    features_penalty_weight: float = 10.0
    """The weight of the regularization penalty applied to the extracted features."""
