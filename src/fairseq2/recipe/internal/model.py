# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from torch.nn import Module

from fairseq2.models import ModelFamily


@dataclass
class _ModelHolder:
    model: Module
    family: ModelFamily
    config: object
    dp_model: Module = field(init=False)
    newly_initialized: bool = False

    def __post_init__(self) -> None:
        self.dp_model = self.model
