# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy

from fairseq2.models.bestrq.quantizer import MultiRandomVectorQuantizer
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2.frontend import Wav2Vec2Frontend
from fairseq2.models.wav2vec2.masker import Wav2Vec2Masker, extract_masked_elements
from fairseq2.nn import Linear
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import TransformerEncoder
from fairseq2.typing import DataType, Device
from fairseq2.models.model import Model


@final
class BestRQModel(Model):
    """Represents a wav2vec 2.0 model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    model_dim: int
    encoder_frontend: Wav2Vec2Frontend
    encoder: TransformerEncoder
    masker: Wav2Vec2Masker
    quantizer: MultiRandomVectorQuantizer
    logit_temp: float

    def __init__(
        self,
        encoder_frontend: Wav2Vec2Frontend,
        encoder: TransformerEncoder,
        masker: Wav2Vec2Masker,
        quantizer: MultiRandomVectorQuantizer,
        downsampler: ConcatDownsampler,
        final_dim: int,
        *,
        logit_temp: float = 0.1,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        model_dim = encoder.model_dim

        self.model_dim = model_dim

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

        self.masker = masker

        self.quantizer = quantizer
        self.logit_temp = logit_temp

        self.downsampler = downsampler

        self.final_bert_proj = nn.ModuleList([])
        for _ in range(self.quantizer.num_quantizer):
            proj = Linear(
                self.model_dim,
                self.quantizer.num_codebook_entries,
                bias=True,
                device=device,
                dtype=dtype,
            )
            self.final_bert_proj.append(proj)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        for proj in self.final_bert_proj:
            nn.init.xavier_uniform_(proj.weight)

    def forward(self, batch: SequenceBatch) -> BestRQOutput:
        """
        :param batch:
            The batch of sequences to process.
        """

        # Forward pass the batch through the model -- in the process generate the temporal mask
        features = self.extract_features(batch)

        # Downsample the batch and quantize the masked time steps
        targets, padding_mask = self.downsampler(batch.seqs, batch.padding_mask)
        targets = extract_masked_elements(targets, features.temporal_mask)
        quantizer_output = self.quantizer(targets)

        seqs = extract_masked_elements(features.seqs, features.temporal_mask)
        bert_logits = torch.stack([proj(seqs) for proj in self.final_bert_proj], dim=0)

        bert_targets = quantizer_output.quantized_targets.view(
            self.quantizer.num_quantizer,
            -1,
        )

        # (N, S_msk, V x G) -> (N x S_msk, V, G)
        bert_logits = bert_logits.view(
            self.quantizer.num_quantizer, -1, self.quantizer.num_codebook_entries
        )
        
        if self.logit_temp != 1.0:
            bert_logits = bert_logits / self.logit_temp

        return BestRQOutput(
            logits=bert_logits,
            targets=bert_targets,
            quantizer_output=quantizer_output,
            temporal_mask=features.temporal_mask,
            encoder_output=features.seqs,
        )

    def extract_features(self, batch: SequenceBatch) -> BestRQFeatures:
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
    ) -> BestRQFeatures:
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

        seqs, padding_mask, temporal_mask = frontend.process_features(
            seqs, padding_mask, self.masker
        )

        assert temporal_mask is not None

        return BestRQFeatures(seqs, padding_mask, temporal_mask, raw_features)

    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
        )


@final
@dataclass
class BestRQFeatures:
    """Holds the extracted features of a wav2vec 2.0 model."""

    seqs: Tensor
    """The features. *Shape:* :math:`(N,S_{enc},M)`, where :math:`N` is the
    batch size, :math:`S_{out}` is the output sequence length, and :math:`M` is
    the dimensionality of the model."""

    padding_mask: PaddingMask | None
    """The padding mask of :attr:`seqs`. *Shape:* :math:`(N,S_{out})`, where
    :math:`N` is the batch size and :math:`S_{out}` is the output sequence
    length."""

    temporal_mask: Tensor
    """The temporal mask that has been used to extract the context network
    targets. *Shape:* :math:`(N,S_{enc})`, where :math:`N` is the batch size and
    :math`S_{enc}` is the encoder output sequence length."""

    raw: Tensor
    """The raw features returned by the frontend. *Shape*: Same as :attr:`seqs`."""


@final
@dataclass
class BestRQOutput:
    """Holds the output of a w2v-BERT model."""

    logits: Tensor
    """The logits for masked feature prediction. *Shape:*
    :math:`(C, NxS_{msk}, V)`, where :math:`C` is the number of
    codebooks, :math:`N` is the batch size,
    :math:`S_{msk}` is the masked sequence length, and :math:`V` is the number of
    entries per codebook."""

    targets: Tensor
    """The target entry index per target codebook. *Shape:*
    :math:`(C, NxS_{msk})`, where :math:`N` is the batch size,
    and :math:`S_{msk}` is the masked sequence length."""
    
    quantizer_output: MultiRandomVectorQuantizerOutput

    temporal_mask: Tensor
    """The temporal mask that has been applied to extract the context network
    targets. *Shape:* :math:`(N,S_{enc})`, where :math:`N` is the batch size and
    :math`S_{enc}` is the encoder output sequence length."""

    encoder_output: Tensor
    """The context network output. *Shape:* :math:`(N,S_{enc},M)`, where
    :math:`N` is the batch size, :math:`S_{enc}` is the encoder output sequence
    length, and :math:`M` is the dimensionality of the model."""

    def compute_loss(
        self,
        *,
        label_smoothing: float = 0.0,
    ) -> BestRQLoss:
        """Compute the loss.

        :param label_smoothing:
            The amount of label smoothing when computing masked prediction loss.
        """
        bert_loss = self.compute_bert_loss(label_smoothing=label_smoothing)

        return BestRQLoss(total=bert_loss, bert=bert_loss)

    def compute_bert_loss(self, *, label_smoothing: float = 0.0) -> Tensor:
        """Compute the masked prediction loss.

        :param label_smoothing:
            The amount of label smoothing when computing masked prediction loss.
        """
        # For numerical stability in low-precision.
        logits = self.logits.float()
        targets = self.targets.long()

        return sum(
            [
                cross_entropy(
                    lgts, trgts, reduction="sum", label_smoothing=label_smoothing
                )
                for lgts, trgts in zip(logits, targets)
            ]
        )

    @property
    def predictions(self):
        return self.logits.argmax(dim=-1)
    
    def compute_encoder_entropy(self):

        # Compute entropy of the target codebooks
        prediction_entropy = []
        for logit in self.logits:
            # Compute entropy of the prediction
            predictions = logit.argmax(dim=-1)
            idxs, counts = torch.unique(predictions, return_counts=True)
            counts = counts.float()
            probs = torch.zeros(self.quantizer_output.num_codebook_entries, device=counts.device).scatter(
                0, idxs, counts
            )
            probs /= probs.sum()
            probs += 1e-10
            prediction_entropy.append(-(probs * torch.log(probs)).sum().item())
        
        return torch.tensor(prediction_entropy)


@final
@dataclass
class BestRQLoss:
    """Holds the loss of a w2v-BERT model."""

    total: Tensor
    """The total loss. *Shape:* :math:`()`."""

    bert: Tensor
    """The masked prediction loss. *Shape:* :math:`()`."""
