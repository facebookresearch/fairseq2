# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import final

from torch import Tensor
from torch.nn import Dropout
from typing_extensions import override

from fairseq2.error import NotSupportedError
from fairseq2.models.feature_extractor import SequenceFeatureExtractor
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.nn import BatchLayout, IncrementalStateBag, PositionEncoder, Projection


@final
class S2TTransformerFrontend(TransformerFrontend):
    """Represents a Transformer encoder front-end as described in Section 2.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`."""

    def __init__(
        self,
        model_dim: int,
        feature_extractor: SequenceFeatureExtractor | None,
        pos_encoder: PositionEncoder | None,
        proj: Projection | None,
        *,
        dropout_p: float = 0.0,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param feature_extractor:
            The feature extractor. If ``None``, features are assumed to be
            extracted externally before being fed to the model.
        :param pos_encoder:
            The position encoder.
        :param proj: The projection to extracted features before dropout as
            described in Section 2 of
            :cite:t:`https://doi.org/10.48550/arxiv.2005.08100`.
        :param dropout_p:
            The dropout probability on extracted features.
        """
        super().__init__()

        self.feature_extractor: SequenceFeatureExtractor | None

        self.register_module("feature_extractor", feature_extractor)

        self.scale = math.sqrt(model_dim)

        self.pos_encoder: PositionEncoder | None

        self.register_module("pos_encoder", pos_encoder)

        self.proj: Projection | None

        self.register_module("proj", proj)

        if dropout_p > 0.0:
            dropout = Dropout(dropout_p)
        else:
            dropout = None

        self.dropout: Dropout | None

        self.register_module("dropout", dropout)

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> tuple[Tensor, BatchLayout]:
        if state_bag is not None:
            raise NotSupportedError(
                f"`{S2TTransformerFrontend}` does not support incremental decoding."
            )

        if self.feature_extractor is not None:
            seqs, seqs_layout = self.feature_extractor(seqs, seqs_layout)

        if self.scale != 1.0:
            seqs = seqs * self.scale

        if self.pos_encoder is not None:
            seqs = self.pos_encoder(seqs, seqs_layout)

        if self.proj is not None:
            seqs = self.proj(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs, seqs_layout
