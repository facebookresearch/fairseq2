# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Dropout, LayerNorm

from fairseq2.models.feature_extractor import SequenceFeatureExtractor
from fairseq2.models.transformer import TransformerFrontend, TransformerFrontendOutput
from fairseq2.models.wav2vec2.masker import Wav2Vec2Masker, apply_temporal_mask
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.position_encoder import PositionEncoder
from fairseq2.nn.projection import Linear
from fairseq2.nn.utils.mask import to_padding_mask


class Wav2Vec2Frontend(TransformerFrontend):
    """Represents a Transformer encoder front-end as described in
    :cite:t:`baevski2020wav2vec`."""

    pretraining: bool
    feature_dim: int
    feature_extractor: Optional[SequenceFeatureExtractor]
    post_extract_layer_norm: LayerNorm
    post_extract_proj: Optional[Linear]
    post_extract_dropout_p: Optional[Dropout]
    masker: Optional[Wav2Vec2Masker]
    pos_encoder: Optional[PositionEncoder]
    layer_norm: Optional[LayerNorm]
    dropout: Optional[Dropout]

    def __init__(
        self,
        model_dim: int,
        feature_dim: int,
        feature_extractor: Optional[SequenceFeatureExtractor],
        pos_encoder: Optional[PositionEncoder],
        pretrain: bool = False,
        post_extract_dropout_p: float = 0.0,
        masker: Optional[Wav2Vec2Masker] = None,
        layer_norm: bool = False,
        dropout_p: float = 0.1,
        norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param feature_dim:
            The dimensionality of extracted features.
        :param feature_extractor:
            The feature extractor. If ``None``, features are assumed to be
            extracted externally before being fed to the model.
        :param pos_encoder:
            The position encoder.
        :param pretrain:
            If ``True``, applies masking and returns non-quantized context
            network targets.
        :param post_extract_dropout_p:
            The dropout probability on extracted features before masking and
            positional encoding.
        :param masker:
            The temporal/spatial feature masker.
        :param layer_norm:
            If ``True``, applies Layer Normalization to extracted features
            before dropout.
        :param dropout_p:
            The dropout probability on extracted features.
        :param norm_eps:
            The epsilon value to add to the denominator of the
            :class:`~torch.nn.LayerNorm` module for numerical stability.
        """
        super().__init__(model_dim)

        self.pretraining = pretrain

        self.feature_dim = feature_dim

        if feature_extractor is not None:
            if feature_dim != feature_extractor.feature_dim:
                raise ValueError(
                    f"`feature_dim` of `feature_extractor` and `feature_dim` must be equal, but are {feature_extractor.feature_dim} and {feature_dim} instead."
                )

            self.feature_extractor = feature_extractor
        else:
            self.register_module("feature_extractor", None)

        self.post_extract_layer_norm = LayerNorm(
            feature_dim, norm_eps, device=device, dtype=dtype
        )

        if feature_dim != model_dim:
            self.post_extract_proj = Linear(
                feature_dim, model_dim, bias=True, device=device, dtype=dtype
            )
        else:
            self.register_module("post_extract_proj", None)

        if post_extract_dropout_p > 0.0:
            self.post_extract_dropout = Dropout(post_extract_dropout_p)
        else:
            self.register_module("post_extract_dropout", None)

        if pretrain:
            if not masker:
                raise ValueError(
                    "`masker` must be specified when `pretrain` is `True`."
                )

            self.masker = masker
        else:
            self.register_module("masker", None)

        if pos_encoder is not None:
            if pos_encoder.encoding_dim != model_dim:
                raise ValueError(
                    f"`encoding_dim` of `pos_encoder` and `model_dim` must be equal, but are {pos_encoder.encoding_dim} and {model_dim} instead."
                )

            self.pos_encoder = pos_encoder
        else:
            self.register_module("pos_encoder", None)

        if layer_norm:
            self.layer_norm = LayerNorm(model_dim, norm_eps, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

    def forward(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> "Wav2Vec2FrontendOutput":
        """
        :param seqs:
            The source sequences to process. *Shape:* :math:`(N,S,*)`, where
            :math:`N` is the batch size, :math:`S` is the source sequence
            length, and :math:`*` is any number of sequence-specific dimensions
            including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is
            the batch size.
        """
        if state_bag is not None:
            raise ValueError(
                "`Wav2Vec2Frontend` does not support incremental evaluation."
            )

        if self.feature_extractor is not None:
            seqs, seq_lens = self.feature_extractor(seqs, seq_lens)

        padding_mask = to_padding_mask(seqs, seq_lens)

        seqs = self.post_extract_layer_norm(seqs)

        if self.pretraining:
            targets = seqs.clone().detach()

            if self.post_extract_dropout is not None:
                targets = self.post_extract_dropout(targets)
        else:
            targets = None

        if self.post_extract_proj is not None:
            seqs = self.post_extract_proj(seqs)

        if self.post_extract_dropout is not None:
            seqs = self.post_extract_dropout(seqs)

        if self.pretraining:
            seqs, temporal_mask = self.masker(seqs, seq_lens)  # type: ignore[misc]

            targets = apply_temporal_mask(targets, temporal_mask)
        else:
            temporal_mask = None

        if self.pos_encoder is not None:
            seqs = self.pos_encoder(seqs, padding_mask)

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return Wav2Vec2FrontendOutput(seqs, padding_mask, targets, temporal_mask)

    def pretrained(self) -> None:
        """Mark the frontend as pretrained."""
        self.pretraining = False

        self.register_module("masker", None)

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self.pretrain:
            s += ", pretrain=True"

        return s + f", feature_dim={self.feature_dim}"


@dataclass
class Wav2Vec2FrontendOutput(TransformerFrontendOutput):
    targets: Optional[Tensor]
    """The non-quantized context network targets extracted from ``seqs``.
    *Shape:* :math:`(N,S_{msk},M)`, where :math:`N` is the batch size,
    :math:`S_{msk}` is the masked sequence length, and :math:`M` is the
    dimensionality of the model."""

    temporal_mask: Optional[Tensor]
    """The boolean temporal mask that has been applied to the processed source
    sequences. *Shape:* :math:`(N,S_{out})`, where :math:`N` is the batch size
    and :math`S_{out}` is the output sequence length."""
