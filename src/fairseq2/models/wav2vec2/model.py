# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from fairseq2.models.wav2vec2.frontend import Wav2Vec2Frontend
from fairseq2.models.wav2vec2.masker import apply_temporal_mask
from fairseq2.models.wav2vec2.vector_quantizer import (
    VectorQuantizer,
    VectorQuantizerOutput,
)
from fairseq2.nn.projection import Linear
from fairseq2.nn.transformer import TransformerEncoder


class Wav2Vec2Model(Module):
    """Represents a wav2vec 2.0 model as described in
    :cite:t:`baevski2020wav2vec`."""

    model_dim: int
    encoder_frontend: Wav2Vec2Frontend
    encoder: TransformerEncoder
    vector_quantizer: VectorQuantizer
    final_proj: Linear
    final_target_proj: Linear
    num_negatives: int
    logit_temp: float
    diversity_loss_weight: float

    def __init__(
        self,
        encoder_frontend: Wav2Vec2Frontend,
        encoder: TransformerEncoder,
        quantizer: VectorQuantizer,
        final_dim: int,
        final_proj_bias: bool = True,
        num_negatives: int = 100,
        logit_temp: float = 0.1,
        diversity_loss_weight: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param encoder_frontend:
            The encoder frontend.
        :param encoder:
            The encoder (i.e. context network).
        :param quantizer:
            The quantizer to discretize context network targets.
        :param final_dim:
            The dimensionality of the final projection that is applied to
            context network outputs and quantized targets before computing
            logits.
        :param final_proj_bias:
            If ``True``, the final projection layer learns an additive bias.
        :param num_negatives:
            The number of negative examples for contrastive loss.
        :param logit_temp:
            The temperature to divide logits by.
        :param diversity_loss_weight:
            The weight of diversity in loss computation.
        """
        super().__init__()

        model_dim = encoder.model_dim

        if encoder_frontend.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `encoder_frontend` and `model_dim` of `encoder` must be equal, but are {encoder_frontend.model_dim} and {model_dim} instead."
            )

        self.model_dim = model_dim

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

        if quantizer.input_dim != encoder_frontend.feature_dim:
            raise ValueError(
                f"`input_dim` of `quantizer` and `feature_dim` of `encoder_frontend` must be equal, but are {quantizer.input_dim} and {encoder_frontend.feature_dim} instead."
            )

        self.quantizer = quantizer

        self.final_proj = Linear(
            model_dim, final_dim, bias=final_proj_bias, device=device, dtype=dtype
        )

        self.final_target_proj = Linear(
            self.quantizer.quantized_dim,
            final_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )

        self.num_negatives = num_negatives
        self.logit_temp = logit_temp
        self.diversity_loss_weight = diversity_loss_weight

    def forward(self, seqs: Tensor, seq_lens: Optional[Tensor]) -> "Wav2Vec2Output":
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
        if not self.encoder_frontend.pretraining:
            raise RuntimeError("`encoder_frontend` is not in pretraining mode.")

        frontend_out = self.encoder_frontend(seqs, seq_lens)

        # TODO: Should we pad for fp16?
        encoder_out = self.encoder(frontend_out.seqs, frontend_out.padding_mask)

        seqs = apply_temporal_mask(encoder_out, frontend_out.temporal_mask)

        seqs = self.final_proj(seqs)

        quantizer_out = self.quantizer(frontend_out.targets)

        targets = self.final_target_proj(quantizer_out.quantized)

        negatives = self._sample_negatives(targets, self.num_negatives)

        logits = self._compute_logits(seqs, targets, negatives)

        return Wav2Vec2Output(
            logits,
            targets,
            frontend_out.temporal_mask,
            encoder_out,
            frontend_out.padding_mask,
            quantizer_out,
            self.diversity_loss_weight,
        )

    @staticmethod
    def _sample_negatives(targets: Tensor, num_negatives: int) -> Tensor:
        batch_size, seq_len, model_dim = targets.shape

        device = targets.device

        # (N, S, M) -> (N x S, M)
        targets = targets.view(-1, model_dim)

        # (S x L)
        indices = torch.arange(seq_len, device=device).repeat_interleave(num_negatives)

        # (N, S x L)
        rand_indices = torch.randint(
            low=0,
            high=seq_len - 1,
            size=(batch_size, seq_len * num_negatives),
            device=device,
        )

        # (N, S x L)
        rand_indices[rand_indices >= indices] += 1

        # (N, 1)
        k = torch.arange(batch_size, device=device).unsqueeze(1) * seq_len

        # (N, S x L)
        rand_indices += k

        # (N, S x L) -> (N x S x L)
        rand_indices = rand_indices.view(-1)

        # (N x S x L, M)
        negs = targets[rand_indices]

        # (N x S x L) -> (N, S, L, M)
        negs = negs.view(batch_size, seq_len, num_negatives, model_dim)

        return negs

    def _compute_logits(self, seqs: Tensor, targets: Tensor, negs: Tensor) -> Tensor:
        # (N, S, M) -> (N, S, 1, M)
        seqs, targets = seqs.unsqueeze(2), targets.unsqueeze(2)

        # The true quantized target is always at index 0 for cross-entropy.
        # (N, S, 1, M) + (N, S, L, M) -> (N, S, L + 1, M)
        targets_with_negs = torch.cat([targets, negs], dim=2)

        # Perform in fp32.
        # (N, S, L + 1, M) -> (N, S, L + 1)
        logits = torch.cosine_similarity(
            seqs.float(), targets_with_negs.float(), dim=-1
        )

        if self.logit_temp != 1.0:
            logits = logits / self.logit_temp

        neg_is_pos = (targets == negs).all(-1)

        # If `True`, it means codebook utilization is low. In such case we
        # mask the corresponding logits.
        if neg_is_pos.any():
            logits[:, :, 1:][neg_is_pos] = float("-inf")

        return logits.type_as(seqs)

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}, num_negatives={self.num_negatives}, logit_temp={self.logit_temp}, diversity_loss_weight={self.diversity_loss_weight}"


@dataclass
class Wav2Vec2Output:
    """Holds the output of a wav2vec 2.0 model."""

    logits: Tensor
    """The logits of masked time steps. *Shape:* :math:`(N,S_{msk},L)`, where
    :math:`N` is the batch size, :math:`S_{msk}` is the masked sequence length,
    and :math:`L` is the number of negative examples plus 1 for the true
    target."""

    quantized_targets: Tensor
    """The quantized context network targets extracted from the source
    sequences. *Shape:* :math:`(N,S_{msk},M)`, where :math:`N` is the batch
    size, :math:`S_{msk}` is the masked sequence length, and :math:`M` is the
    dimensionality of the model."""

    temporal_mask: Tensor
    """The boolean temporal mask that has been applied to the encoded source
    sequences. *Shape:* :math:`(N,S_{out})`, where :math:`N` is the batch size
    and :math`S_{out}` is the output sequence length."""

    encoder_out: Tensor
    """The encoded source sequences. *Shape:* :math:`(N,S_{out},M)`, where
    :math:`N` is the batch size, :math:`S_{out}` is the output sequence length,
    and :math:`M` is the dimensionality of the model."""

    encoder_padding_mask: Optional[Tensor]
    """The float padding mask of ``encoder_out``. *Shape:* :math:`(N,S_{out})`,
    where :math:`N` is the batch size and :math:`S_{out}` is the output sequence
    length."""

    quantizer_output: VectorQuantizerOutput
    """The output of the vector quantizer."""

    diversity_loss_weight: float
    """The weight of diversity in loss computation."""

    def compute_loss(self) -> "Wav2Vec2Loss":
        """Compute the loss."""
        contrastive_loss = self.compute_contrastive_loss()

        diversity_loss = self.compute_diversity_loss()

        loss = contrastive_loss + self.diversity_loss_weight * diversity_loss

        return Wav2Vec2Loss(loss, contrastive_loss, diversity_loss)

    def compute_contrastive_loss(self) -> Tensor:
        """Compute the contrastive loss."""
        batch_size, seq_len, num_logits = self.logits.shape

        # (N, S, L) -> (S x N, L)
        logits = self.logits.transpose(0, 1).reshape(-1, num_logits)

        # The first target is always the true one.
        true_target_indices = logits.new_zeros(
            (batch_size * seq_len,), dtype=torch.int64
        )

        return F.cross_entropy(logits, true_target_indices, reduction="sum")

    def compute_diversity_loss(self) -> Tensor:
        """Compute the diversity loss."""
        batch_size, seq_len = self.logits.shape[:2]

        return self.quantizer_output.compute_loss() * batch_size * seq_len


@dataclass
class Wav2Vec2Loss:
    """Holds the loss of a wav2vec 2.0 model."""

    total: Tensor
    """The weighted total loss."""

    contrastive: Tensor
    """The contrastive loss."""

    diversity: Tensor
    """The diversity loss."""

    def backward(self) -> None:
        """Compute the gradient of the loss."""
        self.total.backward()
