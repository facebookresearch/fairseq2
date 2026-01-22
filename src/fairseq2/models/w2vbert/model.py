# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.wav2vec2 import (
    Wav2Vec2Loss,
    Wav2Vec2Masker,
    Wav2Vec2Model,
    Wav2Vec2Output,
    Wav2Vec2VectorQuantizerOutput,
)
from fairseq2.nn import BatchLayout, Linear
from fairseq2.nn.functional import cross_entropy


@final
class W2VBertModel(Module):
    """Represents a w2v-BERT model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2108.06209`."""

    def __init__(
        self,
        w2v2_model: Wav2Vec2Model,
        num_bert_encoder_layers: int,
        *,
        num_target_codebooks: int = 1,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param w2v2_model:
            The wav2vec 2.0 model.
        :param num_bert_encoder_layers:
            The number of Transformer encoder layers to use for masked
            prediction.
        :param num_target_codebooks:
            The number of consecutive groups of latent variables to use as
            masked prediction targets.
        """
        super().__init__()

        self.model_dim = w2v2_model.model_dim

        self.w2v2_model = w2v2_model

        self.num_bert_encoder_layers = num_bert_encoder_layers

        self.final_bert_proj = Linear(
            w2v2_model.model_dim,
            w2v2_model.quantizer.num_codebook_entries * num_target_codebooks,
            bias=True,
            device=device,
            dtype=dtype,
        )

        self.num_target_codebooks = num_target_codebooks

    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        bert_weight: float = 1.0,
        bert_label_smoothing: float = 0.0,
        w2v2_weight: float = 1.0,
        w2v2_diversity_weight: float = 0.1,
        w2v2_features_penalty_weight: float = 10.0,
    ) -> tuple[W2VBertLoss, W2VBertOutput]:
        w2v2_features = self.w2v2_model.run_frontend(seqs, seqs_layout)

        def hook(
            layer_idx: int,
            layer_output: Tensor,
            layer_output_layout: BatchLayout,
            num_layers: int,
        ) -> bool:
            if layer_idx == num_layers - self.num_bert_encoder_layers - 1:
                w2v2_features.seqs = layer_output

            return True

        with self.w2v2_model.encoder.register_layer_hook(hook):
            encoder_output = self.w2v2_model.encoder(
                w2v2_features.seqs, w2v2_features.seqs_layout
            )

        w2v2_output = self.w2v2_model.quantize_and_contrast(w2v2_features)

        seqs = Wav2Vec2Masker.extract_masked_elements(
            encoder_output, w2v2_features.temporal_mask
        )

        bert_logits = self.final_bert_proj(seqs)

        # (N, S_msk, V x G) -> (N x S_msk, V, G)
        bert_logits = bert_logits.view(
            -1,
            self.w2v2_model.quantizer.num_codebook_entries,
            self.num_target_codebooks,
        )

        bert_targets = self._get_target_indices(w2v2_output.quantizer_output)

        output = W2VBertOutput(w2v2_output, bert_logits, bert_targets)

        loss = self.compute_loss(
            output,
            bert_weight=bert_weight,
            bert_label_smoothing=bert_label_smoothing,
            w2v2_weight=w2v2_weight,
            w2v2_diversity_weight=w2v2_diversity_weight,
            w2v2_features_penalty_weight=w2v2_features_penalty_weight,
        )

        return loss, output

    if TYPE_CHECKING:
        __call__ = forward

    def _get_target_indices(
        self, quantizer_output: Wav2Vec2VectorQuantizerOutput
    ) -> Tensor:
        num_codebooks = self.w2v2_model.quantizer.num_codebooks

        batch_size, seq_len = quantizer_output.quantized_vectors.shape[:2]

        cb = quantizer_output.cb.view(batch_size * seq_len * num_codebooks, -1)

        indices = cb.argmax(dim=-1).view(-1, num_codebooks)

        indices = indices[..., :num_codebooks]

        return indices.detach()

    def compute_loss(
        self,
        output: W2VBertOutput,
        *,
        bert_weight: float = 1.0,
        bert_label_smoothing: float = 0.0,
        w2v2_weight: float = 1.0,
        w2v2_diversity_weight: float = 0.1,
        w2v2_features_penalty_weight: float = 10.0,
    ) -> W2VBertLoss:
        bert_loss = cross_entropy(
            output.bert_logits,
            output.bert_targets,
            pad_idx=None,
            reduction="sum",
            label_smoothing=bert_label_smoothing,
        )

        w2v2_loss = self.w2v2_model.compute_loss(
            output.w2v2_output,
            diversity_weight=w2v2_diversity_weight,
            features_penalty_weight=w2v2_features_penalty_weight,
        )

        weighted_bert_loss = bert_weight * bert_loss
        weighted_w2v2_loss = w2v2_weight * w2v2_loss.aggregate

        return W2VBertLoss(
            weighted_bert_loss + weighted_w2v2_loss, bert_loss, w2v2_loss
        )

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
            f"num_bert_encoder_layers={self.num_bert_encoder_layers}, "
            f"num_target_codebooks={self.num_target_codebooks}"
        )


@dataclass
class W2VBertOutput:
    """Holds the output of a w2v-BERT model."""

    w2v2_output: Wav2Vec2Output
    """The output of the wav2vec 2.0 model."""

    bert_logits: Tensor
    """The logits for masked feature prediction. *Shape:*
    :math:`(NxS_{msk},V,G_{tgt})`, where :math:`N` is the batch size,
    :math:`S_{msk}` is the masked sequence length, :math:`V` is the number of
    entries per codebook, and :math:`G_{tgt}` is the number of target
    codebooks."""

    bert_targets: Tensor
    """The target entry index per target codebook. *Shape:*
    :math:`(NxS_{msk},G_{tgt})`, where :math:`N` is the batch size,
    :math:`S_{msk}` is the masked sequence length, and :math:`G_{tgt}` is the
    number of target codebooks."""


@dataclass
class W2VBertLoss:
    aggregate: Tensor
    bert: Tensor
    w2v2: Wav2Vec2Loss
