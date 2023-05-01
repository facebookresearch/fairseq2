# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Module

from fairseq2.models.feature_extractor import FeatureExtractor
from fairseq2.models.wav2vec2.mask import Wav2Vec2Mask
from fairseq2.nn.positional_embedding import PositionalEmbedding
from fairseq2.nn.projection import Linear
from fairseq2.nn.transformer import TransformerNormOrder
from fairseq2.nn.utils.mask import to_padding_mask


class Wav2Vec2Frontend(Module):
    """Represents a wav2vec 2.0 model front-end as described in
    :cite:t:`baevski2020wav2vec`."""

    model_dim: int
    feat_extractor: Optional[FeatureExtractor]
    feat_layer_norm: LayerNorm
    feat_proj: Optional[Linear]
    feat_dropout: Optional[Dropout]
    mask: Wav2Vec2Mask
    pos_embed: Optional[PositionalEmbedding]
    layer_norm: Optional[LayerNorm]
    dropout: Optional[Dropout]

    def __init__(
        self,
        model_dim: int,
        feat_extractor: Optional[FeatureExtractor],
        mask: Wav2Vec2Mask,
        pos_embed: Optional[PositionalEmbedding],
        feat_dropout_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        dropout_p: float = 0.1,
        norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs).
        :param feat_extractor:
            The feature extractor. If ``None``, it is assumed that features are
            extracted externally before being fed to the model.
        :param mask:
            The mask to apply to extracted features.
        :param pos_embed:
            The positional embedding.
        :param feature_dropout_p:
            The dropout probability on outputs of ``feat_extractor`` before
            masking.
        :param layer_norm:
            If ``True``, applies Layer Normalization to outputs before dropout.
        :param dropout_p:
            The dropout probability on outputs.
        :param norm_eps:
            The epsilon value to add to the denominator of the
            :class:`~torch.nn.LayerNorm` module for numerical stability.
        """
        super().__init__()

        self.model_dim = model_dim

        if feat_extractor is not None:
            embed_dim = feat_extractor.embed_dim

            self.feat_extractor = feat_extractor
        else:
            embed_dim = model_dim

            self.register_module("feat_extractor", None)

        self.feat_layer_norm = LayerNorm(
            embed_dim, norm_eps, device=device, dtype=dtype
        )

        if feat_extractor is not None and embed_dim != model_dim:
            self.feat_proj = Linear(
                embed_dim, model_dim, bias=True, device=device, dtype=dtype
            )
        else:
            self.register_module("feat_proj", None)

        if feat_dropout_p > 0.0:
            self.feat_dropout = Dropout(feat_dropout_p)
        else:
            self.register_module("feat_dropout", None)

        self.mask = mask

        if pos_embed is not None:
            if pos_embed.embed_dim != model_dim:
                raise ValueError(
                    f"`embed_dim` of `pos_embed` and `model_dim` must be equal, but are {pos_embed.embed_dim} and {model_dim} instead."
                )

            self.pos_embed = pos_embed
        else:
            self.register_module("pos_embed", None)

        if norm_order == TransformerNormOrder.POST:
            self.layer_norm = LayerNorm(model_dim, norm_eps, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

    def forward(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        :param seqs:
            The sequences to process. *Shape:* :math:`(N,S,*)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is
            the batch size.

        :returns:
            - The processed sequences to pass to the encoder. *Shape:*
              :math:`(N,S,M)`, where :math:`N` is the batch size, :math:`(S)` is
              the sequence length, and :math:`M` is the dimensionality of the
              model.
            - The non-quantized context network target embeddings extracted
              from ``seqs``. *Shape:* :math:`(N,S,M)`, where :math:`N` is the
              batch size, :math:`(S)` is the sequence length, and :math:`M` is
              the dimensionality of the model.
            - The float padding mask of the processed sequences. *Shape:*
              :math:`(N,S)`, where :math:`N` is the batch size and :math:`S` is
              the sequence length.
            - The boolean temporal mask that has been applied to the processed
              sequences. *Shape:* :math:`(N,S)`, where :math:`N` is the batch
              size and :math`S` is the sequence length.
        """
        if self.feat_extractor is not None:
            seqs, seq_lens = self.feat_extractor(seqs, seq_lens)

        padding_mask = to_padding_mask(seqs, seq_lens)

        seqs = self.feat_layer_norm(seqs)

        # The feature extractor outputs will later be quantized and used as the
        # targets of the context network.
        tgts = seqs.clone()

        if self.feat_proj is not None:
            seqs = self.feat_proj(seqs)

        if self.feat_dropout is not None:
            seqs = self.feat_dropout(seqs)
            tgts = self.feat_dropout(tgts)

        seqs, temporal_mask = self.mask(seqs, seq_lens)

        if self.pos_embed is not None:
            seqs = self.pos_embed(seqs, padding_mask)

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs, tgts, padding_mask, temporal_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"
