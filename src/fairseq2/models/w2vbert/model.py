# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, final

from torch import Tensor
from torch.nn.functional import cross_entropy

from fairseq2.models.model import Model
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import (
    Wav2Vec2Features,
    Wav2Vec2Loss,
    Wav2Vec2Model,
    Wav2Vec2Output,
)
from fairseq2.models.wav2vec2.masker import extract_masked_elements
from fairseq2.nn import Linear
from fairseq2.nn.padding import PaddingMask
from fairseq2.typing import DataType, Device


@final
class W2VBertModel(Model):
    """Represents a w2v-BERT model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2108.06209`."""

    model_dim: int
    w2v2_model: Wav2Vec2Model
    num_bert_encoder_layers: int
    num_target_codebooks: int

    def __init__(
        self,
        w2v2_model: Wav2Vec2Model,
        num_bert_encoder_layers: int,
        *,
        num_target_codebooks: int = 1,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
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
            self.model_dim,
            w2v2_model.quantizer.num_codebook_entries * num_target_codebooks,
            bias=True,
            device=device,
            dtype=dtype,
        )

        self.num_target_codebooks = num_target_codebooks

    def forward(self, batch: SequenceBatch) -> W2VBertOutput:
        """
        :param batch:
            The batch of sequences to process.
        """
        seqs, padding_mask, targets, temporal_mask = self.w2v2_model.run_frontend(
            batch.seqs, batch.padding_mask
        )

        w2v2_layer_output = None
        w2v2_layer_padding_mask = None

        def hook(
            layer_idx: int,
            layer_output: Tensor,
            layer_padding_mask: Optional[PaddingMask],
            num_layers: int,
        ) -> bool:
            nonlocal w2v2_layer_output
            nonlocal w2v2_layer_padding_mask

            if layer_idx == num_layers - self.num_bert_encoder_layers - 1:
                w2v2_layer_output = layer_output
                w2v2_layer_padding_mask = layer_padding_mask

            return True

        with self.w2v2_model.encoder.register_layer_output_hook(hook):
            encoder_output, _ = self.w2v2_model.encoder(seqs, padding_mask)

        assert w2v2_layer_output is not None

        features = Wav2Vec2Features(
            w2v2_layer_output, w2v2_layer_padding_mask, targets, temporal_mask
        )

        w2v2_output = self.w2v2_model.quantize_and_contrast(features)

        seqs = extract_masked_elements(encoder_output, temporal_mask)

        bert_logits = self.final_bert_proj(seqs)

        # (N, S_msk, V x G) -> (N x S_msk, V, G)
        bert_logits = bert_logits.view(
            -1,
            self.w2v2_model.quantizer.num_codebook_entries,
            self.num_target_codebooks,
        )

        bert_targets = w2v2_output.quantizer_output.get_target_indices(
            self.num_target_codebooks
        )

        return W2VBertOutput(w2v2_output, bert_logits, bert_targets)

    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
            f"num_bert_encoder_layers={self.num_bert_encoder_layers}, "
            f"num_target_codebooks={self.num_target_codebooks}"
        )


@final
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

    def compute_loss(
        self,
        *,
        w2v2_loss_weight: float = 1.0,
        bert_loss_weight: float = 1.0,
        bert_label_smoothing: float = 0.0,
    ) -> W2VBertLoss:
        """Compute the loss.

        :param w2v2_loss_weight:
            The weight of wav2vec 2.0 loss in loss computation.
        :param bert_loss_weight:
            The weight of masked prediction loss in loss computation.
        :param bert_label_smoothing:
            The amount of label smoothing when computing masked prediction loss.
        """
        bert_loss = self.compute_bert_loss(label_smoothing=bert_label_smoothing)

        w2v2_loss = self.w2v2_output.compute_loss()

        l1 = bert_loss_weight * bert_loss
        l2 = w2v2_loss_weight * w2v2_loss.total

        return W2VBertLoss(l1 + l2, bert_loss, w2v2_loss)

    def compute_bert_loss(self, *, label_smoothing: float = 0.0) -> Tensor:
        """Compute the masked prediction loss.

        :param label_smoothing:
            The amount of label smoothing when computing masked prediction loss.
        """
        return cross_entropy(
            self.bert_logits,
            self.bert_targets,
            reduction="sum",
            label_smoothing=label_smoothing,
        )


@final
@dataclass
class W2VBertLoss:
    """Holds the loss of a w2v-BERT model."""

    total: Tensor
    """The total loss. *Shape:* :math:`()`."""

    bert: Tensor
    """The masked prediction loss. *Shape:* :math:`()`."""

    w2v2: Wav2Vec2Loss
    """The loss of the wav2vec 2.0 model."""

    def detach(self) -> W2VBertLoss:
        """Return a copy detached from the autograd graph."""
        return W2VBertLoss(self.total.detach(), self.bert.detach(), self.w2v2.detach())
