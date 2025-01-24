# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cross_entropy

from fairseq2.error import InternalError
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2._frontend import Wav2Vec2Frontend
from fairseq2.models.wav2vec2._masker import Wav2Vec2Masker, extract_masked_elements
from fairseq2.models.wav2vec2._vector_quantizer import (
    VectorQuantizer,
    VectorQuantizerOutput,
)
from fairseq2.nn import Linear
from fairseq2.nn.ops import repeat_interleave
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import TransformerEncoder
from fairseq2.typing import DataType, Device


@final
class Wav2Vec2Model(Module):
    """Represents a wav2vec 2.0 model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    model_dim: int
    encoder_frontend: Wav2Vec2Frontend
    encoder: TransformerEncoder
    masker: Wav2Vec2Masker
    quantizer: VectorQuantizer
    final_proj: Linear
    final_target_proj: Linear
    num_distractors: int
    logit_temp: float

    def __init__(
        self,
        encoder_frontend: Wav2Vec2Frontend,
        encoder: TransformerEncoder,
        masker: Wav2Vec2Masker,
        quantizer: VectorQuantizer,
        final_dim: int,
        *,
        final_proj_bias: bool = True,
        num_distractors: int = 100,
        logit_temp: float = 0.1,
        quantizer_encoder_grad: bool = True,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param encoder_frontend:
            The encoder frontend.
        :param encoder:
            The encoder (i.e. context network).
        :param masker:
            The feature masker.
        :param quantizer:
            The quantizer to discretize context network targets.
        :param final_dim:
            The dimensionality of the final projection that is applied to
            context network outputs and quantized targets.
        :param final_proj_bias:
            If ``True``, the final projection learns an additive bias.
        :param num_distractors:
            The number of distractors to use in contrastive prediction.
        :param logit_temp:
            The temperature to divide logits by.
        """
        super().__init__()

        model_dim = encoder.model_dim

        self.model_dim = model_dim

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

        self.masker = masker

        if quantizer.input_dim != encoder_frontend.feature_dim:
            raise ValueError(
                f"`input_dim` of `quantizer` and `feature_dim` of `encoder_frontend` must be equal, but are {quantizer.input_dim} and {encoder_frontend.feature_dim} instead."
            )

        self.quantizer = quantizer

        self.final_proj = Linear(
            model_dim, final_dim, final_proj_bias, device=device, dtype=dtype
        )

        self.final_target_proj = Linear(
            self.quantizer.output_dim,
            final_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )

        self.num_distractors = num_distractors
        self.logit_temp = logit_temp
        self.quantizer_encoder_grad = quantizer_encoder_grad

    def forward(self, batch: SequenceBatch) -> Wav2Vec2Output:
        """
        :param batch:
            The batch of sequences to process.
        """
        features = self.extract_features(batch)

        return self.quantize_and_contrast(features)

    def extract_features(self, batch: SequenceBatch) -> Wav2Vec2Features:
        """Extract features from the input sequences.

        :param batch:
            The batch of sequences to process.
        """
        features = self.run_frontend(batch.seqs, batch.padding_mask)

        features.seqs, features.padding_mask = self.encoder(
            features.seqs, features.padding_mask
        )

        return features

    def run_frontend(
        self, seqs: Tensor, padding_mask: PaddingMask | None
    ) -> Wav2Vec2Features:
        """Run the encoder frontend in pretraining mode.

        :param seqs:
            The sequences to process. *Shape:* :math:`(N,S,*)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        """
        frontend = self.encoder_frontend

        seqs, padding_mask, raw_features = frontend.extract_features(seqs, padding_mask)

        # We use the extracted features as context network targets after masking
        # and quantization.
        if self.quantizer_encoder_grad:
            targets = seqs.clone()
        else:
            targets = seqs.detach().clone()

        if frontend.first_pass_dropout is not None:
            targets = frontend.first_pass_dropout(targets)

        seqs, padding_mask, temporal_mask = frontend.process_features(
            seqs, padding_mask, self.masker
        )

        if temporal_mask is None:
            raise InternalError("`temporal_mask` is `None`.")

        targets = extract_masked_elements(targets, temporal_mask)

        return Wav2Vec2Features(
            seqs, padding_mask, targets, temporal_mask, raw_features
        )

    def quantize_and_contrast(self, features: Wav2Vec2Features) -> Wav2Vec2Output:
        """Quantize targets and produce logits for contrastive prediction.

        :param features:
            The extracted features from the encoder.
        """
        encoder_output, encoder_padding_mask, targets, temporal_mask = (
            features.seqs,
            features.padding_mask,
            features.targets,
            features.temporal_mask,
        )

        seqs = extract_masked_elements(encoder_output, temporal_mask)

        seqs = self.final_proj(seqs)

        quantizer_output = self.quantizer(targets)

        targets = self.final_target_proj(quantizer_output.quantized_vectors)

        distractors = self._sample_distractors(targets)

        logits = self._compute_logits(seqs, targets, distractors)

        return Wav2Vec2Output(
            logits,
            targets,
            temporal_mask,
            quantizer_output,
            encoder_output,
            encoder_padding_mask,
            features.raw,
        )

    def _sample_distractors(self, targets: Tensor) -> Tensor:
        batch_size, seq_len, model_dim = targets.shape

        device = targets.device

        # (N, S, M) -> (N x S, M)
        targets = targets.view(-1, model_dim)

        # (S)
        indices = torch.arange(seq_len, device=device)

        # (S) -> (S x L)
        indices = repeat_interleave(indices, dim=0, repeat=self.num_distractors)

        # (N, S x L)
        rand_indices = torch.randint(
            low=0,
            high=seq_len - 1,
            size=(batch_size, seq_len * self.num_distractors),
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
        distractors = targets[rand_indices]

        # (N x S x L) -> (N, S, L, M)
        distractors = distractors.view(
            batch_size, seq_len, self.num_distractors, model_dim
        )

        return distractors

    def _compute_logits(
        self, seqs: Tensor, targets: Tensor, distractors: Tensor
    ) -> Tensor:
        # (N, S, M) -> (N, S, 1, M)
        seqs, targets = seqs.unsqueeze(2), targets.unsqueeze(2)

        # The target will be always at index 0 in the candidate list.
        # (N, S, 1, M) + (N, S, L, M) -> (N, S, L + 1, M)
        candidates = torch.cat([targets, distractors], dim=2)

        # Perform in fp32.
        # (N, S, L + 1, M) -> (N, S, L + 1)
        logits = torch.cosine_similarity(seqs.float(), candidates.float(), dim=-1)

        if self.logit_temp != 1.0:
            logits = logits / self.logit_temp

        distractor_is_target = (targets == distractors).all(-1)

        # If `True`, it means codebook utilization is low. In such case we
        # mask the corresponding logits.
        if distractor_is_target.any():
            logits[:, :, 1:][distractor_is_target] = -torch.inf

        return logits

    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
            f"num_distractors={self.num_distractors}, "
            f"logit_temp={self.logit_temp:G}, "
        )


@dataclass
class Wav2Vec2Features:
    """Holds the extracted features of a wav2vec 2.0 model."""

    seqs: Tensor
    """The features. *Shape:* :math:`(N,S_{enc},M)`, where :math:`N` is the
    batch size, :math:`S_{out}` is the output sequence length, and :math:`M` is
    the dimensionality of the model."""

    padding_mask: PaddingMask | None
    """The padding mask of :attr:`seqs`. *Shape:* :math:`(N,S_{out})`, where
    :math:`N` is the batch size and :math:`S_{out}` is the output sequence
    length."""

    targets: Tensor
    """The non-quantized context network targets that have been extracted from
    the input sequences. *Shape:* :math:`(N,S_{msk},M)`, where :math:`N` is the
    batch size, :math:`S_{msk}` is the masked sequence length, and :math:`M` is
    the dimensionality of the model."""

    temporal_mask: Tensor
    """The temporal mask that has been used to extract the context network
    targets. *Shape:* :math:`(N,S_{enc})`, where :math:`N` is the batch size and
    :math`S_{enc}` is the encoder output sequence length."""

    raw: Tensor
    """The raw features returned by the frontend. *Shape*: Same as :attr:`seqs`."""


@dataclass
class Wav2Vec2Output:
    """Holds the output of a wav2vec 2.0 model."""

    logits: Tensor
    """The logits for contrastive feature prediction. *Shape:*
    :math:`(N,S_{msk},L)`, where :math:`N` is the batch size, :math:`S_{msk}`
    is the masked sequence length, and :math:`L` is the number of candidates
    (i.e. the number of distractors plus 1 for the target)."""

    quantized_targets: Tensor
    """The quantized context network targets that have been extracted from the
    input sequences. *Shape:* :math:`(N,S_{msk},M)`, where :math:`N` is the
    batch size, :math:`S_{msk}` is the masked sequence length, and :math:`M` is
    the dimensionality of the model."""

    temporal_mask: Tensor
    """The temporal mask that has been applied to extract the context network
    targets. *Shape:* :math:`(N,S_{enc})`, where :math:`N` is the batch size and
    :math`S_{enc}` is the encoder output sequence length."""

    quantizer_output: VectorQuantizerOutput
    """The output of the vector quantizer."""

    encoder_output: Tensor
    """The context network output. *Shape:* :math:`(N,S_{enc},M)`, where
    :math:`N` is the batch size, :math:`S_{enc}` is the encoder output sequence
    length, and :math:`M` is the dimensionality of the model."""

    encoder_padding_mask: PaddingMask | None
    """The padding mask of :attr:`encoder_output`. *Shape:* :math:`(N,S_{enc})`,
    where :math:`N` is the batch size and :math:`S_{enc}` is the encoder output
    sequence length."""

    raw_features: Tensor
    """The raw features returned by the frontend. *Shape*: Same as
    :attr:`encoder_output`."""

    def compute_loss(
        self, diversity_loss_weight: float = 0.1, feature_penalty_weight: float = 10.0
    ) -> Wav2Vec2Loss:
        """Compute the loss.

        :param diversity_loss_weight:
            The weight of diversity in loss computation.
        :param feature_penalty_weight:
            The weight of the feature penalty in loss computation.
        """
        contrastive_loss = self.compute_contrastive_loss()

        diversity_loss = self.compute_diversity_loss()

        feature_penalty = self.compute_feature_penalty()

        weighted_diversity_loss = diversity_loss_weight * diversity_loss

        weighted_feature_penalty = feature_penalty_weight * feature_penalty

        loss = contrastive_loss + weighted_diversity_loss + weighted_feature_penalty

        return Wav2Vec2Loss(loss, contrastive_loss, diversity_loss, feature_penalty)

    def compute_contrastive_loss(self) -> Tensor:
        """Compute the contrastive loss."""
        batch_size, seq_len, num_logits = self.logits.shape

        # (N, S, L) -> (S x N, L)
        logits = self.logits.transpose(0, 1).reshape(-1, num_logits)

        # For numerical stability in low-precision.
        logits = logits.float()

        # The target is always at index 0 in the candidate list.
        target_indices = logits.new_zeros((batch_size * seq_len,), dtype=torch.int64)

        return cross_entropy(logits, target_indices, reduction="sum")

    def compute_diversity_loss(self) -> Tensor:
        """Compute the diversity loss."""
        batch_size, seq_len = self.logits.shape[:2]

        return self.quantizer_output.compute_loss() * batch_size * seq_len

    def compute_feature_penalty(self) -> Tensor:
        """Compute the feature penalty."""
        batch_size, seq_len = self.logits.shape[:2]

        return self.raw_features.float().pow(2).mean() * batch_size * seq_len


@dataclass
class Wav2Vec2Loss:
    """Holds the loss of a wav2vec 2.0 model."""

    total: Tensor
    """The total loss. *Shape:* :math:`()`."""

    contrastive: Tensor
    """The contrastive loss. *Shape:* :math:`()`."""

    diversity: Tensor
    """The diversity loss. *Shape:* :math:`()`."""

    feature_penalty: Tensor
    """The feature penalty. *Shape:* :math:`()`."""
