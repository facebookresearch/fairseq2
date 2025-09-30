# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cross_entropy
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.error import InternalError
from fairseq2.models.transformer import TransformerEncoder
from fairseq2.models.wav2vec2.frontend import Wav2Vec2Frontend
from fairseq2.models.wav2vec2.masker import Wav2Vec2Masker
from fairseq2.models.wav2vec2.vector_quantizer import (
    Wav2Vec2VectorQuantizer,
    Wav2Vec2VectorQuantizerOutput,
)
from fairseq2.nn import BatchLayout, Linear
from fairseq2.ops import repeat_interleave


@final
class Wav2Vec2Model(Module):
    """Represents a wav2vec 2.0 model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    def __init__(
        self,
        model_dim: int,
        encoder_frontend: Wav2Vec2Frontend,
        encoder: TransformerEncoder,
        masker: Wav2Vec2Masker,
        quantizer: Wav2Vec2VectorQuantizer,
        final_dim: int,
        *,
        quantizer_encoder_grad: bool = True,
        final_proj_bias: bool = True,
        num_distractors: int = 100,
        logit_temp: float = 0.1,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param encoder_frontend:
            The encoder frontend.
        :param encoder:
            The encoder (i.e. resolver network).
        :param masker:
            The feature masker.
        :param quantizer:
            The quantizer to discretize resolver network targets.
        :param final_dim:
            The dimensionality of the final projection that is applied to
            resolver network outputs and quantized targets.
        :param final_proj_bias:
            If ``True``, the final projection learns an additive bias.
        :param num_distractors:
            The number of distractors to use in contrastive prediction.
        :param logit_temp:
            The temperature to divide logits by.
        """
        super().__init__()

        self.model_dim = model_dim

        self.encoder_frontend = encoder_frontend

        self.encoder = encoder

        self.masker = masker

        self.quantizer = quantizer

        self.quantizer_encoder_grad = quantizer_encoder_grad

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

    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        diversity_weight: float = 0.1,
        features_penalty_weight: float = 10.0,
    ) -> tuple[Wav2Vec2Loss, Wav2Vec2Output]:
        features = self.extract_features(seqs, seqs_layout)

        output = self.quantize_and_contrast(features)

        loss = self.compute_loss(
            output,
            diversity_weight=diversity_weight,
            features_penalty_weight=features_penalty_weight,
        )

        return loss, output

    if TYPE_CHECKING:
        __call__ = forward

    def extract_features(
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> Wav2Vec2Features:
        """Extract features from the input sequences."""
        features = self.run_frontend(seqs, seqs_layout)

        features.seqs = self.encoder(features.seqs, features.seqs_layout)

        return features

    def run_frontend(self, seqs: Tensor, seqs_layout: BatchLayout) -> Wav2Vec2Features:
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

        seqs, seqs_layout, raw_features = frontend.extract_features(seqs, seqs_layout)

        # We use the extracted features as resolver network targets after masking
        # and quantization.
        if self.quantizer_encoder_grad:
            targets = seqs.clone()
        else:
            targets = seqs.detach().clone()

        if frontend.first_pass_dropout is not None:
            targets = frontend.first_pass_dropout(targets)

        seqs, temporal_mask = frontend.process_features(seqs, seqs_layout, self.masker)

        if temporal_mask is None:
            raise InternalError("`temporal_mask` is `None`.")

        targets = Wav2Vec2Masker.extract_masked_elements(targets, temporal_mask)

        return Wav2Vec2Features(seqs, seqs_layout, targets, temporal_mask, raw_features)

    def quantize_and_contrast(self, features: Wav2Vec2Features) -> Wav2Vec2Output:
        """Quantize targets and produce logits for contrastive prediction.

        :param features:
            The extracted features from the encoder.
        """
        encoder_output, encoder_output_layout, targets, temporal_mask = (
            features.seqs,
            features.seqs_layout,
            features.targets,
            features.temporal_mask,
        )

        seqs = Wav2Vec2Masker.extract_masked_elements(encoder_output, temporal_mask)

        seqs = self.final_proj(seqs)

        quantizer_output = self.quantizer(targets)

        targets = self.final_target_proj(quantizer_output.quantized_vectors)

        distractors = self._sample_distractors(targets)

        logits = self._compute_logits(seqs, targets, distractors)

        batch_size, seq_len = logits.shape[:2]

        num_targets = batch_size * seq_len

        return Wav2Vec2Output(
            logits,
            targets,
            num_targets,
            temporal_mask,
            quantizer_output,
            encoder_output,
            encoder_output_layout,
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

    def compute_loss(
        self,
        output: Wav2Vec2Output,
        *,
        diversity_weight: float = 0.1,
        features_penalty_weight: float = 10.0,
    ) -> Wav2Vec2Loss:
        contrastive_loss = self.compute_contrastive_loss(output.logits)

        diversity_loss = self.compute_diversity_loss(
            output.logits, output.quantizer_output.prob_perplexity
        )

        features_penalty = self.compute_features_penalty(
            output.logits, output.raw_features
        )

        weighted_diversity_loss = diversity_weight * diversity_loss

        weighted_features_penalty = features_penalty_weight * features_penalty

        aggregate_loss = (
            contrastive_loss + weighted_diversity_loss + weighted_features_penalty
        )

        return Wav2Vec2Loss(
            aggregate_loss, contrastive_loss, diversity_loss, features_penalty
        )

    def compute_contrastive_loss(self, logits: Tensor) -> Tensor:
        batch_size, seq_len, num_logits = logits.shape

        # (N, S, L) -> (S x N, L)
        logits = logits.transpose(0, 1).reshape(-1, num_logits)

        # For numerical stability in low-precision.
        logits = logits.float()

        # The target is always at index 0 in the candidate list.
        targets = logits.new_zeros((batch_size * seq_len,), dtype=torch.int64)

        return cross_entropy(logits, targets, reduction="sum")

    def compute_diversity_loss(self, logits: Tensor, prob_perplexity: Tensor) -> Tensor:
        num_entries = self.quantizer.num_codebooks * self.quantizer.num_codebook_entries

        quantizer_loss = (num_entries - prob_perplexity) / num_entries

        batch_size, seq_len = logits.shape[:2]

        return quantizer_loss * batch_size * seq_len  # type: ignore[no-any-return]

    def compute_features_penalty(self, logits: Tensor, raw_features: Tensor) -> Tensor:
        batch_size, seq_len = logits.shape[:2]

        return raw_features.float().pow(2).mean() * batch_size * seq_len

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
            f"quantizer_encoder_grad={self.quantizer_encoder_grad}, "
            f"num_distractors={self.num_distractors}, "
            f"logit_temp={self.logit_temp:G}"
        )


@dataclass
class Wav2Vec2Features:
    """Holds the extracted features of a wav2vec 2.0 model."""

    seqs: Tensor
    """The features. *Shape:* :math:`(N,S_{enc},M)`, where :math:`N` is the
    batch size, :math:`S_{out}` is the output sequence length, and :math:`M` is
    the dimensionality of the model."""

    seqs_layout: BatchLayout

    targets: Tensor
    """The non-quantized resolver network targets that have been extracted from
    the input sequences. *Shape:* :math:`(N,S_{msk},M)`, where :math:`N` is the
    batch size, :math:`S_{msk}` is the masked sequence length, and :math:`M` is
    the dimensionality of the model."""

    temporal_mask: Tensor
    """The temporal mask that has been used to extract the resolver network
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
    """The quantized resolver network targets that have been extracted from the
    input sequences. *Shape:* :math:`(N,S_{msk},M)`, where :math:`N` is the
    batch size, :math:`S_{msk}` is the masked sequence length, and :math:`M` is
    the dimensionality of the model."""

    num_targets: int
    """The number of targets."""

    temporal_mask: Tensor
    """The temporal mask that has been applied to extract the resolver network
    targets. *Shape:* :math:`(N,S_{enc})`, where :math:`N` is the batch size and
    :math`S_{enc}` is the encoder output sequence length."""

    quantizer_output: Wav2Vec2VectorQuantizerOutput
    """The output of the vector quantizer."""

    encoder_output: Tensor
    """The resolver network output. *Shape:* :math:`(N,S_{enc},M)`, where
    :math:`N` is the batch size, :math:`S_{enc}` is the encoder output sequence
    length, and :math:`M` is the dimensionality of the model."""

    encoder_output_layout: BatchLayout

    raw_features: Tensor
    """The raw features returned by the frontend. *Shape*: Same as
    :attr:`encoder_output`."""


@dataclass
class Wav2Vec2Loss:
    aggregate: Tensor
    contrastive: Tensor
    diversity: Tensor
    features_penalty: Tensor
