# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, final

import torch
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Dropout

from fairseq2.models.feature_extractor import SequenceFeatureExtractor
from fairseq2.models.transformer import TransformerFrontend, TransformerFrontendOutput
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.position_encoder import PositionEncoder
from fairseq2.nn.projection import Linear, Projection
from fairseq2.nn.utils.mask import to_padding_mask


@final
class S2TTransformerFrontend(TransformerFrontend):
    """Represents a Transformer encoder front-end as described in Section 2.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`."""

    feature_extractor: Optional[SequenceFeatureExtractor]
    scale: float
    pos_encoder: Optional[PositionEncoder]
    proj: Optional[Projection]
    dropout: Optional[Dropout]

    def __init__(
        self,
        model_dim: int,
        feature_extractor: Optional[SequenceFeatureExtractor],
        pos_encoder: Optional[PositionEncoder],
        proj: bool = False,
        dropout_p: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
                    f"`feature_dim` of `feature_extractor` and `model_dim` must be equal, but are {feature_extractor.feature_dim} and {model_dim} instead."
                )

            self.feature_extractor = feature_extractor
        else:
            self.register_module("feature_extractor", None)

        self.scale = math.sqrt(model_dim)

        if pos_encoder is not None:
            if pos_encoder.encoding_dim != model_dim:
                raise ValueError(
                    f"`encoding_dim` of `pos_encoder` and `model_dim` must be equal, but are {pos_encoder.encoding_dim} and {model_dim} instead."
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

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> TransformerFrontendOutput:
        if state_bag is not None:
            raise ValueError(
                "`S2TTransformerFrontend` does not support incremental evaluation."
            )

        if self.feature_extractor is not None:
            seqs, seq_lens = self.feature_extractor(seqs, seq_lens)

        padding_mask = to_padding_mask(seqs, seq_lens)

        seqs = seqs * self.scale

        if self.pos_encoder is not None:
            seqs = self.pos_encoder(seqs, padding_mask)

        if self.proj is not None:
            seqs = self.proj(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return TransformerFrontendOutput(seqs, padding_mask)
