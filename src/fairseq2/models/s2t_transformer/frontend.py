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

from fairseq2.models.feature_extractor import FeatureExtractor
from fairseq2.models.transformer import TransformerFrontend, TransformerFrontendOutput
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.positional_encoder import PositionalEncoder
from fairseq2.nn.projection import Linear, Projection
from fairseq2.nn.utils.mask import to_padding_mask


@final
class S2TTransformerFrontend(TransformerFrontend):
    """Represents a Transformer model front-end as described in Section 2.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`."""

    feature_extractor: Optional[FeatureExtractor]
    scale: float
    pos_encoder: Optional[PositionalEncoder]
    proj: Optional[Projection]
    dropout: Optional[Dropout]

    def __init__(
        self,
        model_dim: int,
        feature_extractor: Optional[FeatureExtractor],
        pos_encoder: Optional[PositionalEncoder],
        apply_projection: bool = False,
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
            The positional encoder.
        :param apply_projection:
            If ``True``, applies projection to outputs before dropout as
            described in Section 2 of
            :cite:t:`https://doi.org/10.48550/arxiv.2005.08100`.
        :param dropout_p:
            The dropout probability on outputs.
        """
        super().__init__(model_dim)

        if feature_extractor is not None:
            if feature_extractor.out_dim != model_dim:
                raise ValueError(
                    f"`out_dim` of `feature_extractor` and `model_dim` must be equal, but are {feature_extractor.out_dim} and {model_dim} instead."
                )

            self.feature_extractor = feature_extractor
        else:
            self.register_module("feature_extractor", None)

        self.scale = math.sqrt(model_dim)

        if pos_encoder is not None:
            if pos_encoder.dim != model_dim:
                raise ValueError(
                    f"`dim` of `pos_encoder` and `model_dim` must be equal, but are {pos_encoder.dim} and {model_dim} instead."
                )

            self.pos_encoder = pos_encoder
        else:
            self.register_module("pos_encoder", None)

        if apply_projection:
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
        if self.feature_extractor is not None:
            seqs, seq_lens = self.feature_extractor(seqs, seq_lens)

        padding_mask = to_padding_mask(seqs, seq_lens)

        seqs = seqs * self.scale

        if self.pos_encoder is not None:
            seqs = self.pos_encoder(seqs, padding_mask, state_bag)

        if self.proj is not None:
            seqs = self.proj(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return TransformerFrontendOutput(seqs, padding_mask)

    def extra_repr(self) -> str:
        """:meta private:"""
        return "no_scale=False" if self.scale != 1.0 else ""
