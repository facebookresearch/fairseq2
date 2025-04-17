# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence

from torch.nn import Module, ModuleList


class LayerStack(Module):
    layers: ModuleList

    def __init__(self, layers: Sequence[Module]) -> None:
        super().__init__()

        self.layers = ModuleList(layers)
