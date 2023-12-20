# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Iterator, Optional, final

import torch
from torch.nn import Module
from torch.nn import ModuleList as ModuleListBase

from fairseq2.typing import CPU


@final
class ModuleList(ModuleListBase):
    """Holds submodules in a list.

    This class extends :class:`torch.nn.ModuleList` with an extra feature that
    optionally drops a random number of submodules at every iteration during
    training.

    Usage:

    >>> from torch.nn import Module
    >>>
    >>> from fairseq2.nn import ModuleList
    >>>
    >>> layer1 = Module()
    >>> layer2 = Module()
    >>> layer3 = Module()
    >>>
    >>> layers = ModuleList([layer1, layer2, layer3], drop_p=0.5)
    >>>
    >>> for layer in layers.drop_iter():  # This might iterate over layers 1 and 3.
    ...    x = layer(x)
    >>> for layer in layers.drop_iter():  # This might iterate over all layers.
    ...    x = layer(x)
    >>> for layer in layers.drop_iter():  # This might not iterate over any layers.
    ...    x = layer(x)
    """

    drop_p: float

    def __init__(
        self, modules: Optional[Iterable[Module]] = None, *, drop_p: float = 0.0
    ) -> None:
        """
        :param modules:
            An iterable of modules to add.
        :param drop_p:
            The probability of dropping a submodule during training.
        """
        super().__init__(modules)

        self.drop_p = drop_p

    def drop_iter(self) -> Iterator[Module]:
        """Return an iterator that drops a random set of submodules."""
        if self.drop_p > 0.0 and self.training:
            prob_dist = torch.rand(len(self), device=CPU, dtype=torch.float32)
        else:
            prob_dist = None

        for idx, m in enumerate(super().__iter__()):
            if prob_dist is None or prob_dist[idx] > self.drop_p:
                yield m

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self.drop_p > 0.0:
            s = f"{s}, drop_p={self.drop_p}"

        return s
