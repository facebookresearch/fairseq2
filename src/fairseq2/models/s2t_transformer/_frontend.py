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

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.error import NotSupportedError
from fairseq2.models.feature_extractor import SequenceFeatureExtractor
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.nn import (
    BatchLayout,
    IncrementalStateBag,
    Linear,
    PositionEncoder,
    Projection,
)


@final
class S2TTransformerFrontend(TransformerFrontend):
    """Represents a Transformer encoder front-end as described in Section 2.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`."""

    feature_extractor: SequenceFeatureExtractor | None
    scale: float
    pos_encoder: PositionEncoder | None
    proj: Projection | None
    dropout: Dropout | None

    def __init__(
        self,
        model_dim: int,
        feature_extractor: SequenceFeatureExtractor | None,
        pos_encoder: PositionEncoder | None,
        *,
        proj: bool = False,
        dropout_p: float = 0.0,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param feature_extractor:
            The feature extractor. If ``None``, features are assumed to be
            extracted externally before being fed to the model.
        :param pos_encoder:
            The position encoder.
        :param proj:
            If ``True``, applies projection to extracted features before dropout
            as described in Section 2 of
            :cite:t:`https://doi.org/10.48550/arxiv.2005.08100`.
        :param dropout_p:
            The dropout probability on extracted features.
        """
        super().__init__(model_dim)

        if feature_extractor is not None:
            if feature_extractor.feature_dim != model_dim:
                raise ValueError(
                    f"`feature_dim` of `feature_extractor` must be equal to `model_dim` ({model_dim}), but is {feature_extractor.feature_dim} instead."
                )

            self.feature_extractor = feature_extractor
        else:
            self.register_module("feature_extractor", None)

        self.scale = math.sqrt(model_dim)

        if pos_encoder is not None:
            if pos_encoder.encoding_dim != model_dim:
                raise ValueError(
                    f"`encoding_dim` of `pos_encoder` must be equal to `model_dim` ({model_dim}), but is {pos_encoder.encoding_dim} instead."
                )

            self.pos_encoder = pos_encoder
        else:
            self.register_module("pos_encoder", None)

        if proj:
            self.proj = Linear(
                model_dim, model_dim, bias=True, device=device, dtype=dtype
            )
        else:
            self.register_module("proj", None)

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

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

        seqs = seqs * self.scale

        if self.pos_encoder is not None:
            seqs = self.pos_encoder(seqs, seqs_layout)

        if self.proj is not None:
            seqs = self.proj(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs, seqs_layout
