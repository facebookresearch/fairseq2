# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

import torch.nn as nn
from torch import Tensor

from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.models.transformer.frontend import TransformerFrontend
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.projection import Linear, Projection
from fairseq2.nn.transformer import TransformerDecoder, TransformerEncoder
from fairseq2.nn.utils.module import check_model_dim
from fairseq2.typing import DataType, Device, finaloverride


@final
class TransformerModel(EncoderDecoderModel):
    """Represents a Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    encoder_frontend: TransformerFrontend
    encoder: TransformerEncoder
    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection
    target_pad_idx: Optional[int]

    def __init__(
        self,
        encoder_frontend: TransformerFrontend,
        encoder: TransformerEncoder,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        target_pad_idx: Optional[int],
    ) -> None:
        """
        :param encoder_frontend:
            The encoder frontend.
        :param encoder:
            The encoder.
        :param decoder_frontend:
            The decoder frontend.
        :param decoder:
            The decoder.
        :param final_proj:
            The projection to apply to decoder outputs to produce logits.
        :param target_pad_idx:
            The index of the pad symbol in the target vocabulary.
        """
        model_dim = encoder.model_dim

        super().__init__(model_dim)

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

        self.target_pad_idx = target_pad_idx

        check_model_dim(self)

    @finaloverride
    def encode(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs, padding_mask = self.encoder_frontend(seqs, seq_lens)

        return self.encoder(seqs, padding_mask)  # type: ignore[no-any-return]

    @finaloverride
    def decode(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs, padding_mask = self.decoder_frontend(seqs, seq_lens, state_bag)

        return self.decoder(  # type: ignore[no-any-return]
            seqs, padding_mask, encoder_output, encoder_padding_mask, state_bag
        )

    @finaloverride
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[Tensor]
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.target_pad_idx)


@final
class FinalProjection(Linear):
    """Produces logits from outputs of a Transformer decoder."""

    def __init__(
        self,
        model_dim: int,
        target_vocabulary_size: int,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param target_vocabulary_size:
            The size of the target vocabulary.
        """
        super().__init__(
            model_dim, target_vocabulary_size, bias=False, device=device, dtype=dtype
        )

    @finaloverride
    def _do_reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.normal_(self.weight, std=self.input_dim**-0.5)

        if self.bias is not None:
            nn.init.zeros_(self.bias)
