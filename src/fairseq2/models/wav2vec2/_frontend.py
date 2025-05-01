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

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.error import NotSupportedError
from fairseq2.models.feature_extractor import SequenceFeatureExtractor
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.nn import (
    BatchLayout,
    IncrementalStateBag,
    LayerNorm,
    Linear,
    PositionEncoder,
    StandardLayerNorm,
)

# isort: split

from fairseq2.models.wav2vec2._masker import Wav2Vec2Masker


@final
class Wav2Vec2Frontend(TransformerFrontend):
    """Represents a Transformer encoder front-end as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    feature_dim: int
    feature_extractor: SequenceFeatureExtractor | None
    post_extract_layer_norm: LayerNorm
    model_dim_proj: Linear | None
    first_pass_dropout: Dropout | None
    pos_encoder: PositionEncoder | None
    layer_norm: LayerNorm | None
    dropout: Dropout | None

    def __init__(
        self,
        model_dim: int,
        feature_dim: int,
        feature_extractor: SequenceFeatureExtractor | None,
        pos_encoder: PositionEncoder | None,
        *,
        first_pass_dropout_p: float = 0.0,
        layer_norm: bool = False,
        dropout_p: float = 0.0,
        device: Device | None = None,
        dtype: DataType | None = None,
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
            if feature_extractor.feature_dim != feature_dim:
                raise ValueError(
                    f"`feature_extractor.feature_dim` must be equal to `feature_dim` ({feature_dim}), but is {feature_extractor.feature_dim} instead."
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
                    f"`pos_encoder.encoding_dim` must be equal to `model_dim` ({model_dim}), but is {pos_encoder.encoding_dim} instead."
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
                f"`{Wav2Vec2Frontend}` does not support incremental decoding."
            )

        seqs, seqs_layout, _ = self.extract_features(seqs, seqs_layout)

        seqs, _ = self.process_features(seqs, seqs_layout)

        return seqs, seqs_layout

    def extract_features(
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> tuple[Tensor, BatchLayout, Tensor]:
        """Extract features from the specified sequences.

        :param seqs:
            The sequences from which to extract features. *Shape:*
            :math:`(N,S,*)`, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.

        :returns:
            - The normalized features. *Shape:* :math:`(N,S_{out},E)`, where
              :math:`N` is the batch size, :math:`S_{out}` is the output
              sequence length, and :math:`E` is the dimensionality of the
              extracted features.
            - The raw features. *Shape*: Same as the normalized features (i.e.
              first element of the returned tuple).
        """
        if self.feature_extractor is not None:
            seqs, seqs_layout = self.feature_extractor(seqs, seqs_layout)

        raw_features = seqs.clone()

        seqs = self.post_extract_layer_norm(seqs)

        return seqs, seqs_layout, raw_features

    def process_features(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        masker: Wav2Vec2Masker | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Process extracted features.

        :param seqs:
            The features to process. *Shape:* :math:`(N,S,E)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`E`
            is the dimensionality of the features.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param masker:
            If not ``None``, the features will be masked and the applied
            temporal mask will be returned as the third element of the tuple.

        :returns:
            - The processed features to pass to the context network. *Shape:*
              :math:`(N,S,M)`, where :math:`N` is the batch size, :math:`S` is
              the sequence length, and :math:`M` is the dimensionality of the
              model.
            - The temporal mask that has been applied to the processed features.
              *Shape:* :math:`(N,S)`, where :math:`N` is the batch size and
              :math`S` is the sequence length.
        """
        if self.model_dim_proj is not None:
            seqs = self.model_dim_proj(seqs)

        if self.first_pass_dropout is not None:
            seqs = self.first_pass_dropout(seqs)

        if masker is not None:
            seqs, temporal_mask = masker(seqs, seqs_layout)
        else:
            temporal_mask = None

        if self.pos_encoder is not None:
            seqs = self.pos_encoder(seqs, seqs_layout)

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs, temporal_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, feature_dim={self.feature_dim}"
