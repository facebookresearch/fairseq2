# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cross_entropy

from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Loss, Wav2Vec2Model, Wav2Vec2Output
from fairseq2.models.wav2vec2.masker import extract_masked_elements
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.projection import Linear
from fairseq2.typing import DataType, Device


class W2VBertModel(Module):
    """Represents a w2v-BERT model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2108.06209`."""

    model_dim: int
    w2v2_model: Wav2Vec2Model
    num_bert_encoder_layers: int
    num_target_codebooks: int
    w2v2_loss_weight: float
    bert_loss_weight: float
    bert_label_smoothing: float

    def __init__(
        self,
        w2v2_model: Wav2Vec2Model,
        num_bert_encoder_layers: int,
        *,
        num_target_codebooks: int = 1,
        w2v2_loss_weight: float = 1.0,
        bert_loss_weight: float = 1.0,
        bert_label_smoothing: float = 0.0,
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
        :param w2v2_loss_config:
            The weight of wav2vec 2.0 loss in loss computation.
        :param bert_loss_config:
            The weight of masked prediction loss in loss computation.
        :param bert_label_smoothing:
            The amount of label smoothing when computing masked prediction loss.
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

        self.w2v2_loss_weight = w2v2_loss_weight
        self.bert_loss_weight = bert_loss_weight
        self.bert_label_smoothing = bert_label_smoothing

    def forward(self, batch: SequenceBatch) -> W2VBertOutput:
        """
        :param batch:
            The batch of sequences to process.
        """
        seqs, padding_mask, targets, temporal_mask = self.w2v2_model.run_frontend(
            batch.seqs, batch.padding_mask
        )

        w2v2_layer_output = None

        def hook(
            layer_idx: int,
            layer_output: Tensor,
            layer_padding_mask: Optional[PaddingMask],
            num_layers: int,
        ) -> bool:
            nonlocal w2v2_layer_output

            if layer_idx == num_layers - self.num_bert_encoder_layers - 1:
                w2v2_layer_output = layer_output

            return True

        with self.w2v2_model.encoder.register_layer_output_hook(hook):
            encoder_output, _ = self.w2v2_model.encoder(seqs, padding_mask)

        assert w2v2_layer_output is not None

        w2v2_output = self.w2v2_model.quantize_and_contrast(
            w2v2_layer_output, targets, temporal_mask
        )

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

        return W2VBertOutput(
            w2v2_output,
            bert_logits,
            bert_targets,
            self.w2v2_loss_weight,
            self.bert_loss_weight,
            self.bert_label_smoothing,
        )

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

    w2v2_loss_weight: float = 1.0
    """The weight of wav2vec 2.0 loss in loss computation."""

    bert_loss_weight: float = 1.0
    """The weight of masked prediction loss in loss computation."""

    bert_label_smoothing: float = 0.0
    """The amount of label smoothing when computing masked prediction loss."""

    def compute_loss(self) -> W2VBertLoss:
        """Compute the loss."""
        bert_loss = self.compute_bert_loss()

        w2v2_loss = self.w2v2_output.compute_loss()

        l1 = self.bert_loss_weight * bert_loss
        l2 = self.w2v2_loss_weight * w2v2_loss.total

        return W2VBertLoss(l1 + l2, bert_loss, w2v2_loss)

    def compute_bert_loss(self) -> Tensor:
        """Compute the masked prediction loss."""
        return cross_entropy(
            self.bert_logits,
            self.bert_targets,
            reduction="sum",
            label_smoothing=self.bert_label_smoothing,
        )


@dataclass
class W2VBertLoss:
    """Holds the loss of a w2v-BERT model."""

    total: Tensor
    """The weighted total loss."""

    bert: Tensor
    """The masked prediction loss."""

    w2v2_loss: Wav2Vec2Loss
    """The loss of the wav2vec 2.0 model."""

    def backward(self) -> None:
        """Compute the gradient of the loss."""
        self.total.backward()
