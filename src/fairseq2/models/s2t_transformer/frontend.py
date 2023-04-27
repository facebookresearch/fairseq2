# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple, final

import torch
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Dropout

from fairseq2.models.encoder_decoder import EncoderDecoderFrontend
from fairseq2.models.feature_extractor import FeatureExtractor
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.positional_embedding import PositionalEmbedding
from fairseq2.nn.projection import Linear, Projection


@final
class S2TTransformerFrontend(EncoderDecoderFrontend):
    """Represents a Transformer model front-end as described in Section 2.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`."""

    feat_extract: FeatureExtractor
    scale: float
    pos_embed: Optional[PositionalEmbedding]
    proj: Optional[Projection]
    dropout: Optional[Dropout]

    def __init__(
        self,
        feat_extract: FeatureExtractor,
        pos_embed: Optional[PositionalEmbedding],
        apply_projection: bool = False,
        dropout_p: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param feature_extractor:
            The feature extractor.
        :param pos_embed:
            The positional embedding.
        :param apply_projection:
            If ``True``, applies projection to outputs before dropout as
            described in Section 2 of
            :cite:t:`https://doi.org/10.48550/arxiv.2005.08100`.
        :param dropout_p:
            The dropout probability on outputs.
        """
        model_dim = feat_extract.embed_dim

        super().__init__(model_dim)

        self.feat_extract = feat_extract

        self.scale = math.sqrt(model_dim)

        if pos_embed is not None:
            if pos_embed.embed_dim != model_dim:
                raise ValueError(
                    f"`embed_dim` of `pos_embed` and `embed_dim` of `subsampler` must be equal, but are {pos_embed.embed_dim} and {model_dim} instead."
                )

            self.pos_embed = pos_embed
        else:
            self.register_module("pos_embed", None)

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
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x, seq_lens = self.feat_extract(seqs, seq_lens)

        x = x * self.scale

        if self.pos_embed is not None:
            x = self.pos_embed(x, state_bag)

        if self.proj is not None:
            x = self.proj(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x, seq_lens

    def extra_repr(self) -> str:
        """:meta private:"""
        return "no_scale=False" if self.scale != 1.0 else ""
