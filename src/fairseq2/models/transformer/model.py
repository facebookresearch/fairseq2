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
from fairseq2.nn.utils.mask import to_padding_mask


@final
class TransformerModel(EncoderDecoderModel):
    """Represents a Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    encoder_frontend: EncoderDecoderFrontend
    encoder: TransformerEncoder
    decoder_frontend: EncoderDecoderFrontend
    decoder: TransformerDecoder
    score_proj: Projection

    def __init__(
        self,
        encoder_frontend: EncoderDecoderFrontend,
        encoder: TransformerEncoder,
        decoder_frontend: EncoderDecoderFrontend,
        decoder: TransformerDecoder,
        score_proj: Projection,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs).
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

        self.score_proj = score_proj

    @finaloverride
    def encode(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x, seq_lens = self.encoder_frontend(seqs, seq_lens)

        mask = self._get_padding_mask(x, seq_lens)

        x = self.encoder(x, mask)

        return x, mask

    @finaloverride
    def decode_and_score(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        enc_out: Tensor,
        enc_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        x, seq_lens = self.decoder_frontend(seqs, seq_lens, state_bag)

        mask = self._get_padding_mask(x, seq_lens)

        x = self.decoder(x, mask, enc_out, enc_padding_mask, state_bag)

        x = self.score_proj(x)

        return x  # type: ignore[no-any-return]

    @staticmethod
    def _get_padding_mask(x: Tensor, seq_lens: Optional[Tensor]) -> Optional[Tensor]:
        if seq_lens is not None:
            padding_mask = to_padding_mask(seq_lens, mask_seq_len=x.size(-2))

            # Return only if we mask at least one element.
            if padding_mask.any():
                return padding_mask

        return None


@final
class ScoreProjection(ResettableProjection):
    """Produces scores (i.e. logits) from outputs of a Transformer decoder."""

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
