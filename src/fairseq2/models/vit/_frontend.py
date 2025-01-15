# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor
from torch.nn import Dropout
from typing_extensions import override

from fairseq2.models.transformer import TransformerFrontend
from fairseq2.models.vit._feature_extractor import PatchFeatureExtractor
from fairseq2.nn import IncrementalStateBag, InterpolatedPositionEncoder
from fairseq2.nn.padding import PaddingMask


@final
class StandardViTFrontend(TransformerFrontend):
    """Represents a standard Vision Transformer front-end as described in
    :cite:t:`https://doi.org/10.48550/arXiv.2010.11929`."""

    feature_extractor: PatchFeatureExtractor
    pos_encoder: InterpolatedPositionEncoder
    dropout: Dropout | None

    def __init__(
        self,
        feature_extractor: PatchFeatureExtractor,
        pos_encoder: InterpolatedPositionEncoder,
        *,
        dropout_p: float = 0.0,
    ) -> None:
        """
        :param feature_extractor: The feature extractor.
        :param pos_encoder: The interpolated position encoder.
        :param dropout_p: The dropout probability on extracted patch features.
        """
        feature_dim = feature_extractor.feature_dim

        super().__init__(feature_dim)

        self.feature_extractor = feature_extractor

        if pos_encoder.encoding_dim != feature_dim:
            raise ValueError(
                f"`pos_encoder.encoding_dim` must be equal to `feature_extractor.feature_dim` ({feature_dim}), but is {pos_encoder.encoding_dim} instead."
            )

        self.pos_encoder = pos_encoder

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

    @override
    def forward(
        self,
        seqs: Tensor,
        padding_mask: PaddingMask | None,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> tuple[Tensor, PaddingMask | None]:
        if padding_mask is not None:
            raise ValueError(f"`{type(self)}` does not support padding mask.")

        if state_bag is not None:
            raise ValueError(f"`{type(self)}` does not support incremental decoding.")

        seqs = self.feature_extractor(seqs)

        seqs = self.pos_encoder(seqs)

        # (N, *, E) -> (N, S, E)
        seqs = seqs.flatten(1, -2)

        return seqs, None
