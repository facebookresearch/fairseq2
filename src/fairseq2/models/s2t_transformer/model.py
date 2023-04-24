# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Dropout, Module

from fairseq2.models.s2t_transformer.subsampler import FbankSubsampler
from fairseq2.models.transformer import TransformerTokenFrontend
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.positional_embedding import PositionalEmbedding
from fairseq2.nn.projection import Linear, Projection
from fairseq2.nn.transformer import TransformerDecoder, TransformerEncoder
from fairseq2.nn.utils.mask import to_padding_mask


class TransformerFbankFrontend(Module):
    """Represents a Transformer front-end as described in Section 2.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`."""

    subsampler: FbankSubsampler
    scale: float
    pos_embed: Optional[PositionalEmbedding]
    proj: Optional[Projection]
    dropout: Optional[Dropout]

    def __init__(
        self,
        subsampler: FbankSubsampler,
        pos_embed: Optional[PositionalEmbedding],
        apply_projection: bool = False,
        dropout_p: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param subsampler:
            The log-mel filterbank subsampler.
        :param pos_embed:
            The positional embedding.
        :param apply_projection:
            If ``True``, applies a projection to embeddings before dropout as
            described in Section 2 of
            :cite:t:`https://doi.org/10.48550/arxiv.2005.08100`.
        :param dropout_p:
            The dropout probability on outputs.
        """
        super().__init__()

        embed_dim = subsampler.embed_dim

        self.subsampler = subsampler

        self.scale = math.sqrt(embed_dim)

        if pos_embed is not None:
            if pos_embed.embed_dim != embed_dim:
                raise ValueError(
                    f"`embed_dim` of `pos_embed` and `embed_dim` of `subsampler` must be equal, but are {pos_embed.embed_dim} and {embed_dim} instead."
                )

            self.pos_embed = pos_embed
        else:
            self.register_module("pos_embed", None)

        if apply_projection:
            self.proj = Linear(
                embed_dim, embed_dim, bias=True, device=device, dtype=dtype
            )
        else:
            self.register_module("proj", None)

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

    def forward(
        self, fbanks: Tensor, num_frames: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param fbanks:
            The log-mel filterbanks to subsample. *Shape:* :math:`(N,F,C)`, or
            :math:`(F,C)` when unbatched, where :math:`N` is the batch size,
            :math:`F` is the number of frames, and :math:`C` is the number of
            channels.
        :param num_frames:
            An array where each element represents the number of frames of the
            filterbank at the same index in ``fbanks``. *Shape:* :math:`(N)`,
            :math:`(N,1)`, or :math:`()` when unbatched, where :math:`N` is the
            batch size.

        :returns:
            - The processed audio embeddings, subsampled from ``fbanks``, to
              pass to the encoder. *Shape:* :math:`(N,S,M)`, or :math:`(S,M)`
              when unbatched, where :math:`N` is the batch size, :math:`S` is
              the sequence length, and :math:`M` is the dimensionality of the
              model.
            - The boolean padding mask indicating which key positions to ignore
              for the purpose of self attention. *Shape:* :math:`(N,S)`, or
              :math:`(S)` when unbatched, where :math:`N` is the batch size and
              :math:`S` is the sequence length.

        .. note::
            For a boolean padding mask, a ``True`` indicates that the
            corresponding key position is not allowed to attend.
        """
        embeds, seq_lens = self.subsampler(fbanks, num_frames)

        x = embeds * self.scale

        if self.pos_embed is not None:
            x = self.pos_embed(x)

        if self.proj is not None:
            x = self.proj(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x, self._get_padding_mask(x, seq_lens)

    def _get_padding_mask(self, x: Tensor, seq_lens: Tensor) -> Optional[Tensor]:
        if seq_lens is not None:
            padding_mask = to_padding_mask(seq_lens, mask_seq_len=x.size(-2))

            # Return only if we mask at least one element.
            if padding_mask.any():
                return padding_mask

        return None

    def extra_repr(self) -> str:
        """:meta private:"""
        return "no_scale=False" if self.scale != 1.0 else ""


class S2TTransformerModel(Module):
    """Represents an S2T Transformer model as described in Section 2.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`."""

    model_dim: int
    encoder_frontend: TransformerFbankFrontend
    encoder: TransformerEncoder
    decoder_frontend: TransformerTokenFrontend
    decoder: TransformerDecoder
    score_proj: Projection

    def __init__(
        self,
        encoder_frontend: TransformerFbankFrontend,
        encoder: TransformerEncoder,
        decoder_frontend: TransformerTokenFrontend,
        decoder: TransformerDecoder,
        score_proj: Projection,
    ) -> None:
        """
        :param encoder_frontend:
            The encoder front-end.
        :param encoder:
            The encoder.
        :param decoder_frontend:
            The decoder front-end.
        :param decoder:
            The decoder.
        :param score_proj:
            The projection to apply to outputs of the decoder.
        """
        super().__init__()

        model_dim = encoder.model_dim

        if decoder.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `encoder` and `model_dim` of `decoder` must be equal, but are {encoder.model_dim} and {decoder.model_dim} instead."
            )

        if encoder_frontend.subsampler.embed_dim != model_dim:
            raise ValueError(
                f"`embed_dim` of `encoder_frontend.subsampler` and `model_dim` of `encoder` must be equal, but are {encoder_frontend.subsampler.embed_dim} and {model_dim} instead."
            )

        if decoder_frontend.embed.embed_dim != model_dim:
            raise ValueError(
                f"`embed_dim` of `decoder_frontend.embed` and `model_dim` of `decoder` must be equal, but are {decoder_frontend.embed.embed_dim} and {model_dim} instead."
            )

        self.model_dim = model_dim

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.score_proj = score_proj

    def encode(
        self, fbanks: Tensor, num_frames: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Encode the specified source log-mel filterbanks.

        :param fbanks:
            The log-mel filterbanks to encode. *Shape:* :math:`(N,F,C)`, or
            :math:`(F,C)` when unbatched, where :math:`N` is the batch size,
            :math:`F` is the number of frames, and :math:`C` is the number of
            channels.
        :param num_frames:
            An array where each element represents the number of frames of the
            filterbank at the same index in ``fbanks``. *Shape:* :math:`(N)`,
            :math:`(N,1)`, or :math:`()` when unbatched, where :math:`N` is the
            batch size.

        :returns:
            - The encoded output of ``fbanks``. *Shape:* :math:`(N,S,M)`, or
              :math:`(S,M)` when unbatched, where :math:`N` is the batch size,
              :math:`S` is the sequence length, and :math:`M` is the
              dimensionality of the model.
            - The boolean padding mask indicating which key positions to ignore
              for the purpose of encoder-decoder attention. *Shape:*
              :math:`(N,S)`, or :math:`(S)` when unbatched, where :math:`N` is
              the batch size and :math:`S` is the sequence length.

        .. note::
            For a boolean padding mask, a ``True`` indicates that the
            corresponding key position is not allowed to attend.
        """
        x, padding_mask = self.encoder_frontend(fbanks, num_frames)

        x = self.encoder(x, padding_mask)

        return x, padding_mask

    def decode_and_score(
        self,
        token_indices: Tensor,
        enc_out: Tensor,
        enc_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        """Decode the specified target token indices.

        :param token_indices:
            The token indices to decode. *Shape:* :math:`(N,S)`, or :math:`(S)`
            when unbatched, where :math:`N` is the batch size and :math:`S` is
            the sequence length.
        :param enc_out:
            The encoder output for the encoder-decoder attention. *Shape:*
            :math:`(N,S_{src},M)`, or :math:`(S_{src},M)` when unbatched, where
            :math:`N` is the batch size, :math:`S_{src}` is the source sequence
            length, and :math:`M` is the dimensionality of the model.
        :param enc_padding_mask:
            The boolean or float padding mask indicating which key positions to
            ignore for the purpose of encoder-decoder attention. *Shape:*
            :math:`(N,S_{src})`, or :math:`(S_{src})` when unbatched, where
            :math:`N` is the batch size and :math:`S_{src}` is the source
            sequence length.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            The output of :attr:`score_proj`. A softmax function should be
            applied to the produced scores to obtain the next-token
            probabilities. *Shape:* :math:`(N,S,D)`, or :math:`(S,D)` when
            unbatched, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`D` is the size of the output embedding
            dictionary.
        """
        x, padding_mask = self.decoder_frontend(token_indices, state_bag)

        x = self.decoder(x, padding_mask, enc_out, enc_padding_mask, state_bag)

        x = self.score_proj(x)

        return x  # type: ignore[no-any-return]

    def forward(
        self, fbanks: Tensor, num_frames: Optional[Tensor], tgt_token_indices: Tensor
    ) -> Tensor:
        """
        :param fbanks:
            The source log-mel filterbanks to encode. *Shape:* :math:`(N,F,C)`,
            or :math:`(F,C)` when unbatched, where :math:`N` is the batch size,
            :math:`F` is the number of frames, and :math:`C` is the number of
            channels.
        :param num_frames:
            An array where each element represents the number of frames of the
            source filterbank at the same index in ``fbanks``. *Shape:*
            :math:`(N)`, :math:`(N,1)`, or :math:`()` when unbatched, where
            :math:`N` is the batch size.
        :param tgt_token_indices:
            The target token indices to decode. *Shape:* :math:`(N,S_{tgt})`, or
            :math:`(S_{tgt})` when unbatched, where :math:`N` is the batch size
            and :math:`S_{tgt}` is the target sequence length.

        :returns:
            The output of :attr:`score_proj`. A softmax function should be
            applied to the produced scores to obtain the next-token
            probabilities. *Shape:* :math:`(N,S_{tgt},D)`, or
            :math:`(S_{tgt},D)` when unbatched, where :math:`N` is the batch
            size, :math:`S_{tgt}` is the target sequence length, and :math:`D`
            is the size of the output embedding dictionary.
        """
        enc_out, enc_padding_mask = self.encode(fbanks, num_frames)

        return self.decode_and_score(tgt_token_indices, enc_out, enc_padding_mask)

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"
