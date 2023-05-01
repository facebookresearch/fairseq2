# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

import torch
import torch.nn as nn
from overrides import final as finaloverride
from torch import Tensor

from fairseq2.models.encoder_decoder import EncoderDecoderFrontend, EncoderDecoderModel
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.projection import Projection, ResettableProjection
from fairseq2.nn.transformer import TransformerDecoder, TransformerEncoder


@final
class TransformerModel(EncoderDecoderModel):
    """Represents a Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    encoder_frontend: EncoderDecoderFrontend
    encoder: TransformerEncoder
    decoder_frontend: EncoderDecoderFrontend
    decoder: TransformerDecoder
    final_proj: Projection

    def __init__(
        self,
        encoder_frontend: EncoderDecoderFrontend,
        encoder: TransformerEncoder,
        decoder_frontend: EncoderDecoderFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
    ) -> None:
        """
        :param encoder_frontend:
            The encoder frontend.
        :param encoder:
            The Transformer encoder.
        :param decoder_frontend:
            The decoder frontend.
        :param decoder:
            The Transformer decoder.
        :param final_proj:
            The projection to apply to decoder outputs to produce logits.
        """
        model_dim = encoder.model_dim

        super().__init__(model_dim)

        if decoder.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `encoder` and `model_dim` of `decoder` must be equal, but are {encoder.model_dim} and {decoder.model_dim} instead."
            )

        if encoder_frontend.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `encoder_frontend` and `model_dim` of `encoder` must be equal, but are {encoder_frontend.model_dim} and {model_dim} instead."
            )

        if decoder_frontend.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `decoder_frontend` and `model_dim` of `decoder` must be equal, but are {decoder_frontend.model_dim} and {model_dim} instead."
            )

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

    @finaloverride
    def encode(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs, padding_mask = self.encoder_frontend(seqs, seq_lens)

        seqs = self.encoder(seqs, padding_mask)

        return seqs, padding_mask

    @finaloverride
    def decode_and_project(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        enc_out: Tensor,
        enc_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        seqs, padding_mask = self.decoder_frontend(seqs, seq_lens, state_bag)

        seqs = self.decoder(seqs, padding_mask, enc_out, enc_padding_mask, state_bag)

        return self.final_proj(seqs)  # type: ignore[no-any-return]


@final
class FinalProjection(ResettableProjection):
    """Produces logits from outputs of a Transformer decoder."""

    def __init__(
        self,
        num_embed: int,
        model_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param num_embed:
            The size of the output embedding dictionary.
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__(model_dim, num_embed, bias=False, device=device, dtype=dtype)

    @finaloverride
    def reset_parameters(self) -> None:
        """Reset the parameters of the module."""
        nn.init.normal_(self.weight, std=self.inp_dim**-0.5)

        if self.bias is not None:
            nn.init.zeros_(self.bias)
