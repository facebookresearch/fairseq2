# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch.nn import Module

from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.nn.transformer import TransformerEncoder

# TODO(balioglu): This implementation is not complete. As of this commit, only
# the encoder and encoder-frontend are available for parity check purposes.


@final
class JepaModel(Module):
    """
    Represents a JEPA model as described in:
        * :cite:t:`https://doi.org/10.48550/arXiv.2301.08243`
        * :cite:t:`https://doi.org/10.48550/arXiv.2404.08471`
    """

    model_dim: int
    encoder_frontend: TransformerFrontend
    encoder: TransformerEncoder

    def __init__(
        self,
        encoder_frontend: TransformerFrontend,
        encoder: TransformerEncoder,
    ) -> None:
        super().__init__()

        self.model_dim = encoder.model_dim

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

    def forward(self, batch: SequenceBatch) -> SequenceBatch:
        seqs, padding_mask = self.encoder_frontend(batch.seqs, batch.padding_mask)

        seqs, padding_mask = self.encoder(seqs, padding_mask)

        return SequenceBatch(seqs, padding_mask)
