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
from fairseq2.nn.utils.mask import to_padding_mask


class Wav2Vec2Frontend(Module):
    """Represents a wav2vec 2.0 model front-end as described in
    :cite:t:`baevski2020wav2vec`."""

    feat_extract: FeatureExtractor
    feat_layer_norm: LayerNorm
    feat_dropout: Optional[Dropout]
    mask: Wav2Vec2Mask
    pos_embed: Optional[PositionalEmbedding]
    layer_norm: Optional[LayerNorm]
    dropout: Optional[Dropout]

    def __init__(
        self,
        feat_extract: FeatureExtractor,
        mask: Wav2Vec2Mask,
        pos_embed: Optional[PositionalEmbedding],
        feat_dropout_p: float = 0.0,
        layer_norm: bool = False,
        dropout_p: float = 0.1,
        norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param feature_extractor:
            The feature extractor.
        :param mask:
            The mask to apply to extracted features.
        :param pos_embed:
            The positional embedding.
        :param feature_dropout_p:
            The dropout probability on outputs of ``feat_extract`` before
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

        model_dim = feat_extract.embed_dim

        self.feat_extract = feat_extract

        self.feat_layer_norm = LayerNorm(
            model_dim, norm_eps, device=device, dtype=dtype
        )

        if feat_dropout_p > 0.0:
            self.feat_dropout = Dropout(feat_dropout_p)
        else:
            self.register_module("feature_dropout", None)

        self.mask = mask

        if pos_embed is not None:
            if pos_embed.embed_dim != model_dim:
                raise ValueError(
                    f"`embed_dim` of `pos_embed` and `embed_dim` of `feat_extract` must be equal, but are {pos_embed.embed_dim} and {model_dim} instead."
                )

            self.pos_embed = pos_embed
        else:
            self.register_module("pos_embed", None)

        if layer_norm:
            self.layer_norm = LayerNorm(model_dim, norm_eps, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

    def forward(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        :param seqs:
            The sequences to process. *Shape:* :math:`(N,S,*)`, or :math:`(S,*)`
            when unbatched, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, :math:`(N,1)`, or
            :math:`()` when unbatched, where :math:`N` is the batch size.

        :returns:
            - The processed sequences to pass to the encoder. *Shape:*
              :math:`(N,S,M)`, or :math:`(S,M)` when unbatched, where :math:`N`
              is the batch size, :math:`(S)` is the sequence length, and
              :math:`M` is the dimensionality of the model.
            - The non-quantized context network target embeddings extracted
              from ``seqs``. *Shape:* :math:`(N,S,M)`, or :math:`(S,M)`
              when unbatched, where :math:`N` is the batch size, :math:`(S)` is
              the sequence length, and :math:`M` is the dimensionality of the
              model.
            - The temporal time-step mask that has been applied to the returned
              sequences.
            - The boolean padding mask indicating which key positions to ignore
              for the purpose of self attention. *Shape:* :math:`(N,S)`, or
              :math:`(S)` when unbatched, where :math:`N` is the batch size and
              :math:`S` is the sequence length.
        """
        x, seq_lens = self.feat_extract(seqs, seq_lens)

        x = self.feat_layer_norm(x)

        # The feature extractor outputs will later be quantized and used as the
        # targets of the context network.
        y = x.clone()

        if self.feat_dropout is not None:
            x = self.feat_dropout(x)
            y = self.feat_dropout(y)

        x, mask = self.mask(x, seq_lens)

        # The remaining steps are identical to a regular Transformer front-end.
        if self.pos_embed is not None:
            x = self.pos_embed(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x, y, mask, self._get_padding_mask(x, seq_lens)

    def _get_padding_mask(
        self, x: Tensor, seq_lens: Optional[Tensor]
    ) -> Optional[Tensor]:
        if seq_lens is not None:
            padding_mask = to_padding_mask(seq_lens, mask_seq_len=x.size(-2))

            # Return only if we mask at least one element.
            if padding_mask.any():
                return padding_mask

        return None
