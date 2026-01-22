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

from fairseq2.error import NotSupportedError
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.models.vit.feature_extractor import PatchFeatureExtractor
from fairseq2.nn import (
    BatchLayout,
    IncrementalStateBag,
    InterpolatedPositionEncoder,
)


@final
class StandardViTFrontend(TransformerFrontend):
    """Represents a standard Vision Transformer front-end as described in
    :cite:t:`https://doi.org/10.48550/arXiv.2010.11929`."""

    def __init__(
        self,
        patch_feature_extractor: PatchFeatureExtractor,
        pos_encoder: InterpolatedPositionEncoder,
        *,
        dropout_p: float = 0.0,
    ) -> None:
        """
        :param feature_extractor: The feature extractor.
        :param pos_encoder: The interpolated position encoder.
        :param dropout_p: The dropout probability on extracted patch features.
        """
        super().__init__()

        self.patch_feature_extractor = patch_feature_extractor

        self.pos_encoder = pos_encoder

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
                f"`{StandardViTFrontend}` does not support incremental decoding."
            )

        if seqs_layout.packed:
            raise ValueError("`seqs` must not be a packed batch.")

        if seqs_layout.padded:
            raise ValueError("`seqs` must not be a padded batch.")

        seqs = self.patch_feature_extractor(seqs)

        seqs = self.pos_encoder(seqs)

        # (N, *, E) -> (N, S, E)
        seqs = seqs.flatten(1, -2)

        return seqs, seqs_layout
