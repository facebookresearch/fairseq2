# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

import torch
from overrides import override as finaloverride
from torch import Tensor
from torch.nn import Dropout, LayerNorm

from fairseq2.models.feature_extractor import SequenceFeatureExtractor
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.position_encoder import PositionEncoder
from fairseq2.nn.projection import Linear
from fairseq2.nn.transformer import TransformerNormOrder
from fairseq2.nn.utils.mask import to_padding_mask


@final
class PretrainedWav2Vec2Frontend(TransformerFrontend):
    """Represents a pretrained wav2vec 2.0 encoder front-end as described in
    :cite:t:`baevski2020wav2vec`."""

    feature_extractor: Optional[SequenceFeatureExtractor]
    feature_layer_norm: LayerNorm
    feature_proj: Optional[Linear]
    feature_dropout: Optional[Dropout]
    pos_encoder: Optional[PositionEncoder]
    layer_norm: Optional[LayerNorm]
    dropout: Optional[Dropout]

    def __init__(
        self,
        model_dim: int,
        feature_extractor: Optional[SequenceFeatureExtractor],
        pos_encoder: Optional[PositionEncoder],
        feature_dropout_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        dropout_p: float = 0.1,
        norm_eps: float = 1e-5,
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
        :param feature_dropout_p:
            The dropout probability on outputs of ``feature_extractor``.
        :param layer_norm:
            If ``True``, applies Layer Normalization to features before dropout.
        :param dropout_p:
            The dropout probability on features.
        :param norm_eps:
            The epsilon value to add to the denominator of the
            :class:`~torch.nn.LayerNorm` module for numerical stability.
        """
        super().__init__(model_dim)

        if feature_extractor is not None:
            feature_dim = feature_extractor.out_dim

            self.feature_extractor = feature_extractor
        else:
            feature_dim = model_dim

            self.register_module("feature_extractor", None)

        self.feature_layer_norm = LayerNorm(
            feature_dim, norm_eps, device=device, dtype=dtype
        )

        if feature_dim != model_dim:
            self.feature_proj = Linear(
                feature_dim, model_dim, bias=True, device=device, dtype=dtype
            )
        else:
            self.register_module("feature_proj", None)

        if feature_dropout_p > 0.0:
            self.feature_dropout = Dropout(feature_dropout_p)
        else:
            self.register_module("feature_dropout", None)

        if pos_encoder is not None:
            if pos_encoder.dim != model_dim:
                raise ValueError(
                    f"`dim` of `pos_encoder` and `model_dim` must be equal, but are {pos_encoder.dim} and {model_dim} instead."
                )

            self.pos_encoder = pos_encoder
        else:
            self.register_module("pos_encoder", None)

        if norm_order == TransformerNormOrder.POST:
            self.layer_norm = LayerNorm(model_dim, norm_eps, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

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
        if state_bag is not None:
            raise ValueError(
                "The wav2vec 2.0 encoder front-end does not support incremental evaluation."
            )

        if self.feature_extractor is not None:
            seqs, seq_lens = self.feature_extractor(seqs, seq_lens)

        padding_mask = to_padding_mask(seqs, seq_lens)

        seqs = self.feature_layer_norm(seqs)

        if self.feature_proj is not None:
            seqs = self.feature_proj(seqs)

        if self.feature_dropout is not None:
            seqs = self.feature_dropout(seqs)

        if self.pos_encoder is not None:
            seqs = self.pos_encoder(seqs, padding_mask)

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs, padding_mask
