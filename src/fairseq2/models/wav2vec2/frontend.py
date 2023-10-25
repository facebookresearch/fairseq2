# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

from torch import Tensor
from torch.nn import Dropout

from fairseq2.models.feature_extractor import SequenceFeatureExtractor
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.models.wav2vec2.masker import Wav2Vec2Masker
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.normalization import LayerNorm, StandardLayerNorm
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.position_encoder import PositionEncoder
from fairseq2.nn.projection import Linear
from fairseq2.typing import DataType, Device, finaloverride


@final
class Wav2Vec2Frontend(TransformerFrontend):
    """Represents a Transformer encoder front-end as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    feature_dim: int
    feature_extractor: Optional[SequenceFeatureExtractor]
    post_extract_layer_norm: LayerNorm
    model_dim_proj: Optional[Linear]
    first_pass_dropout: Optional[Dropout]
    pos_encoder: Optional[PositionEncoder]
    layer_norm: Optional[LayerNorm]
    dropout: Optional[Dropout]

    def __init__(
        self,
        model_dim: int,
        feature_dim: int,
        feature_extractor: Optional[SequenceFeatureExtractor],
        pos_encoder: Optional[PositionEncoder],
        *,
        first_pass_dropout_p: float = 0.0,
        layer_norm: bool = False,
        dropout_p: float = 0.1,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
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
        :param first_pass_dropout_p:
            The dropout probability on extracted features before masking and
            positional encoding.
        :param layer_norm:
            If ``True``, applies Layer Normalization to extracted features
            before dropout.
        :param dropout_p:
            The dropout probability on extracted features.
        """
        super().__init__(model_dim)

        self.feature_dim = feature_dim

        if feature_extractor is not None:
            if feature_dim != feature_extractor.feature_dim:
                raise ValueError(
                    f"`feature_dim` of `feature_extractor` must be equal to `feature_dim` ({feature_dim}), but is {feature_extractor.feature_dim} instead."
                )

            self.feature_extractor = feature_extractor
        else:
            self.register_module("feature_extractor", None)

        self.post_extract_layer_norm = StandardLayerNorm(
            feature_dim, bias=True, device=device, dtype=dtype
        )

        if feature_dim != model_dim:
            self.model_dim_proj = Linear(
                feature_dim, model_dim, bias=True, device=device, dtype=dtype
            )
        else:
            self.register_module("model_dim_proj", None)

        if first_pass_dropout_p > 0.0:
            self.first_pass_dropout = Dropout(first_pass_dropout_p)
        else:
            self.register_module("first_pass_dropout", None)

        if pos_encoder is not None:
            if pos_encoder.encoding_dim != model_dim:
                raise ValueError(
                    f"`encoding_dim` of `pos_encoder` must be equal to `model_dim` ({model_dim}), but is {pos_encoder.encoding_dim} instead."
                )

            self.pos_encoder = pos_encoder
        else:
            self.register_module("pos_encoder", None)

        if layer_norm:
            self.layer_norm = StandardLayerNorm(
                model_dim, bias=True, device=device, dtype=dtype
            )
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
        padding_mask: Optional[PaddingMask],
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        if state_bag is not None:
            raise ValueError(
                "`Wav2Vec2Frontend` does not support incremental decoding."
            )

        seqs, padding_mask = self.extract_features(seqs, padding_mask)

        seqs, padding_mask, _ = self.process_features(seqs, padding_mask)

        return seqs, padding_mask

    def extract_features(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """Extract features from the specified sequences.

        :param seqs:
            The sequences from which to extract features. *Shape:*
            :math:`(N,S,*)`, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.

        :returns:
            - The extracted features. *Shape:* :math:`(N,S_{out},F)`, where
              :math:`N` is the batch size, :math:`S_{out}` is the output
              sequence length, and :math:`F` is the dimensionality of the
              extracted features.
            - The padding mask of the extracted features. *Shape:*
              :math:`(N,S_{out})`, where :math:`N` is the batch size and
              :math:`S_{out}` is the output sequence length.
        """
        if self.feature_extractor is not None:
            seqs, padding_mask = self.feature_extractor(seqs, padding_mask)

        seqs = self.post_extract_layer_norm(seqs)

        return seqs, padding_mask

    def process_features(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        masker: Optional[Wav2Vec2Masker] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask], Tensor]:
        """Process extracted features.

        :param seqs:
            The features to process. *Shape:* :math:`(N,S,F)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`F`
            is the dimensionality of the features.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param masker:
            If not ``None``, the features will be masked and the applied
            temporal mask will be returned as the third tuple element.

        :returns:
            - The processed sequences to pass to a Transformer encoder. *Shape:*
              :math:`(N,S,M)`, where :math:`N` is the batch size, :math:`S` is
              the sequence length, and :math:`M` is the dimensionality of the
              model.
            - The padding mask of the processed sequences. *Shape:* :math:`(N,S)`,
              where :math:`N` is the batch size and :math:`S` is the output
              sequence length.
            - The temporal mask that has been applied to the processed sequences.
              *Shape:* :math:`(N,S)`, where :math:`N` is the batch size and
              :math`S` is the sequence length.
        """
        if self.model_dim_proj is not None:
            seqs = self.model_dim_proj(seqs)

        if self.first_pass_dropout is not None:
            seqs = self.first_pass_dropout(seqs)

        if masker is not None:
            seqs, temporal_mask = masker(seqs, padding_mask)
        else:
            temporal_mask = None

        if self.pos_encoder is not None:
            seqs = self.pos_encoder(seqs, padding_mask)

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs, padding_mask, temporal_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, feature_dim={self.feature_dim}"
