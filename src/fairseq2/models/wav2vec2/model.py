# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from fairseq2.models.wav2vec2.frontend import Wav2Vec2Frontend
from fairseq2.models.wav2vec2.vector_quantizer import VectorQuantizer
from fairseq2.nn.projection import Linear
from fairseq2.nn.transformer import TransformerEncoder


class Wav2Vec2Model(Module):
    """Represents a wav2vec 2.0 model as described in
    :cite:t:`baevski2020wav2vec`."""

    model_dim: int
    encoder_frontend: Wav2Vec2Frontend
    encoder: TransformerEncoder
    vector_quantizer: VectorQuantizer
    final_seqs_proj: Linear
    final_tgts_proj: Linear
    num_negatives: int
    logit_temp: float
    diversity_weight: float

    def __init__(
        self,
        encoder_frontend: Wav2Vec2Frontend,
        encoder: TransformerEncoder,
        vector_quantizer: VectorQuantizer,
        final_dim: int = 0,
        num_negatives: int = 100,
        logit_temp: float = 0.1,
        diversity_weight: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        :param encoder_frontend:
            The encoder frontend.
        :param encoder:
            The Transformer encoder.
        :param vector_quantizer:
            The quantizer to use for context network targets.
        :param final_dim:
            The dimensionality of the final projection that is applied to
            intermediate feature representations and quantized targets before
            computing logits. If zero, :attr:`model_dim` will be used.
        :param num_negatives:
            The number of negative examples for contrastive loss.
        :param logit_temp:
            The temperature to divide logits by.
        :param diversity_weight:
            The weight of diversity in loss computation.
        """
        model_dim = encoder.model_dim

        super().__init__()

        self.model_dim = model_dim

        if encoder_frontend.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `encoder_frontend` and `model_dim` of `encoder` must be equal, but are {encoder_frontend.model_dim} and {model_dim} instead."
            )

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder
        self.vector_quantizer = vector_quantizer

        if final_dim == 0:
            final_dim = model_dim

        self.final_seq_proj = Linear(
            model_dim, final_dim, bias=True, device=device, dtype=dtype
        )
        self.final_tgt_proj = Linear(
            final_dim, final_dim, bias=True, device=device, dtype=dtype
        )

        self.num_negatives = num_negatives

        self.logit_temp = logit_temp
        self.diversity_weight = diversity_weight

    def forward(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        :param seqs:
            The sequences to encode. *Shape:* :math:`(N,S,*)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is
            the batch size.

        :returns:
            - The logits of masked time steps. *Shape:* :math:`(N,S_{msk},L)`,
              where :math:`N` is the batch size, :math:`S_{msk}` is the masked
              sequence length, and :math:`L` is the number of negative examples
              plus 1 for the true target.
            - The diversity measure of the vector quantization. The content and
              shape depends on the type of :attr:`vector_quantizer`.
        """
        seqs, padding_mask, tgts, temporal_mask = self.encoder_frontend(seqs, seq_lens)

        # TODO: Should we pad for fp16?
        seqs = self.encoder(seqs, padding_mask)

        batch_size = seqs.size(0)

        seqs = seqs[temporal_mask]
        tgts = tgts[temporal_mask]

        seqs = seqs.unflatten(0, (batch_size, -1))
        tgts = tgts.unflatten(0, (batch_size, -1))

        tgts, diversity = self.vector_quantizer(tgts)

        seqs = self.final_seq_proj(seqs)
        tgts = self.final_tgt_proj(tgts)

        negs = self._sample_negatives(tgts, self.num_negatives)

        logits = self._compute_logits(seqs, tgts, negs)

        return logits, diversity

    def encode(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Encode the specified sequences.

        :param seqs:
            The sequences to encode. *Shape:* :math:`(N,S,*)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is
            the batch size.

        :returns:
            - The encoded output of ``seqs``. *Shape:* :math:`(N,S_{out},M)`,
              where :math:`N` is the batch size, :math:`S_{out}` is the output
              sequence length, and :math:`M` is the dimensionality of the model.
            - The float padding mask of the encoded output. *Shape:*
              :math:`(N,S_{out})`, where :math:`N` is the batch size and
              :math:`S_{out}` is the output sequence length.
        """
        seqs, padding_mask, *_ = self.encoder_frontend(
            seqs, seq_lens, extract_only=True
        )

        # TODO: Should we pad for fp16?
        seqs = self.encoder(seqs, padding_mask)

        return seqs, padding_mask

    @staticmethod
    def _sample_negatives(tgts: Tensor, num_negatives: int) -> Tensor:
        batch_size, seq_len, model_dim = tgts.shape

        device = tgts.device

        # (N, S, M) -> (N x S, M)
        tgts = tgts.view(-1, model_dim)

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
        negs = tgts[rand_indices]

        # (N x S x L) -> (N, S, L, M)
        negs = negs.view(batch_size, seq_len, num_negatives, model_dim)

        return negs

    def _compute_logits(self, seqs: Tensor, tgts: Tensor, negs: Tensor) -> Tensor:
        # (N, S, M) -> (N, S, 1, M)
        seqs = seqs.unsqueeze(2)
        tgts = tgts.unsqueeze(2)

        # The true quantized embedding is always at index 0 for cross-entropy.
        # (N, S, 1, M) + (N, S, L, M) -> (N, S, L + 1, M)
        tgts_with_negs = torch.cat([tgts, negs], dim=2)

        # Perform in fp32.
        # (N, S, L + 1, M) -> (N, S, L + 1)
        logits = torch.cosine_similarity(seqs.float(), tgts_with_negs.float(), dim=-1)

        if self.logit_temp != 1.0:
            logits = logits / self.logit_temp

        neg_is_pos = (tgts == negs).all(-1)

        # If `True`, it means codebook utilization is low. In such case we
        # mask the corresponding logits.
        if neg_is_pos.any():
            logits[:, :, 1:][neg_is_pos] = float("-inf")

        return logits.type_as(seqs)

    def compute_loss(self, logits: Tensor, diversity: Tensor) -> Tensor:
        """Compute the loss.

        :param logits:
            The logits of masked time steps. *Shape:* :math:`(N,S_{msk},L)`,
            where :math:`N` is the batch size, :math:`S_{msk}` is the masked
            sequence length, and :math:`L` is the number of negative examples
            plus 1 for the true target.
        :param diversity:
            The diversity measure of the vector quantization. The content and
            shape depends on the type of :attr:`vector_quantizer`.
        """
        cnt_loss = self._compute_contrastive_loss(logits)

        div_loss = self.vector_quantizer.compute_loss(diversity)

        return cnt_loss + self.diversity_weight * div_loss * logits.numel()

    @staticmethod
    def _compute_contrastive_loss(logits: Tensor) -> Tensor:
        batch_size, seq_len, num_logits = logits.shape

        # (N, S, L) -> (S x N, L)
        logits = logits.transpose(0, 1).reshape(-1, num_logits)

        # The first target at index 0 always corresponds to the the true embedding.
        true_tgt_indices = logits.new_zeros((batch_size * seq_len,), dtype=torch.long)

        return F.cross_entropy(logits, true_tgt_indices, reduction="sum")

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}, num_negatives={self.num_negatives}, logit_temp={self.logit_temp}"
