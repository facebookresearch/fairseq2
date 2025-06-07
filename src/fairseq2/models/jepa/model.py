# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, final

from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.models.transformer import TransformerEncoder, TransformerFrontend
from fairseq2.nn import BatchLayout

# TODO(balioglu): This implementation is not complete. As of this commit, only
# the encoder and encoder-frontend are available for parity check purposes.


@final
class JepaModel(Module):
    """
    Represents a JEPA model as described in:
        * :cite:t:`https://doi.org/10.48550/arXiv.2301.08243`
        * :cite:t:`https://doi.org/10.48550/arXiv.2404.08471`
    """

    def __init__(
        self,
        model_dim: int,
        encoder_frontend: TransformerFrontend,
        encoder: TransformerEncoder,
    ) -> None:
        super().__init__()

        self.model_dim = model_dim
        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

    def forward(
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> tuple[Tensor, BatchLayout]:
        seqs, seqs_layout = self.encoder_frontend(seqs, seqs_layout)

        seqs = self.encoder(seqs, seqs_layout)

        return seqs, seqs_layout

    if TYPE_CHECKING:
        __call__ = forward

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"
