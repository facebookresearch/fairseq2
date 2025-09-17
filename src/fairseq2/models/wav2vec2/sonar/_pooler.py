# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Optional

import torch
from fairseq2.models.transformer import TransformerDecoder, TransformerFrontend
from fairseq2.nn import Linear
from fairseq2.nn.padding import apply_padding_mask, PaddingMask
from fairseq2.typing import Device, override
from torch import Tensor
from torch.nn import Module


class EncoderOutputPooler(Module):
    """Represents a pooler module to be called on encoder output."""

    @abstractmethod
    def __call__(
        self,
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
    ) -> Tensor:
        """Apply pooling on encoder_output

        :param encoder_output:
            The encoder output to use in encoder-decoder attention. *Shape:*
            :math:`(N,S_{enc},M)`, where :math:`N` is the batch size,
            :math:`S_{enc}` is the encoder output sequence length, and :math:`M`
            is the dimensionality of the model.
        :param encoder_padding_mask:
            The float padding mask of ``encoder_output``. *Shape:*
            :math:`(N,S_{enc})`, where :math:`N` is the batch size and
            :math:`S_{enc}` is the encoder output sequence length.

        :returns:
        The pooler output. *Shape:* :math:`(N,M)`, where :math:`N` is the
        batch size, and :math:`M` is the dimensionality of the model.
        """


class AttentionEncoderOutputPooler(EncoderOutputPooler):
    """Attention pooling applied using decoder architecture"""

    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    projection_out: Linear
    bos_idx: int

    def __init__(
        self,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        projection_out: Linear,
        bos_idx: int,
    ) -> None:
        super().__init__()

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder
        self.projection_out = projection_out
        self.bos_idx = bos_idx

    @override
    def __call__(
        self,
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
    ) -> Tensor:

        seqs = self._get_pooling_tokens(encoder_output.shape[0], encoder_output.device)
        seqs, padding_mask = self.decoder_frontend(seqs, None)
        decoder_out, _ = self.decoder(
            seqs, padding_mask, encoder_output, encoder_padding_mask
        )
        # print(decoder_out.shape)
        return self.projection_out(decoder_out).squeeze(1)

    def _get_pooling_tokens(self, batch_size: int, device: Device) -> Tensor:
        """TODO Add clear comment on why we need this"""
        return torch.tensor(
            [self.bos_idx] * batch_size, dtype=torch.int64, device=device
        ).unsqueeze(1)


class MeanEncoderOutputPooler(EncoderOutputPooler):
    """Mean pooling encoder features"""

    def __init__(self, projection_out: Linear) -> None:
        super().__init__()

        self.projection_out = projection_out

    @override
    def __call__(
        self,
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
    ) -> Tensor:
        masked_encoder_output = apply_padding_mask(
            encoder_output, encoder_padding_mask, pad_value=0
        )
        if encoder_padding_mask is not None:
            lengths = encoder_padding_mask.seq_lens
        else:
            lengths = None
        if lengths is None:
            weights = 1.0 / (masked_encoder_output.size(1) + 1e-7)
            pooled_encoder_output = masked_encoder_output.sum(dim=1) * weights
        else:
            pooled_encoder_output = masked_encoder_output.sum(
                dim=1
            ) / lengths.unsqueeze(-1)

        return self.projection_out(pooled_encoder_output)


class SelfAttentiveEncoderOutputPooler(EncoderOutputPooler):
    """self-attentive pooling encoder features"""

    def __init__(
        self, h_linear: Linear, attention: Linear, projection_out: Linear
    ) -> None:
        super().__init__()
        self.h_linear = h_linear
        self.attention = attention
        self.projection_out = projection_out

    @override
    def __call__(
        self,
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
    ) -> Tensor:
        masked_encoder_output = apply_padding_mask(
            encoder_output, encoder_padding_mask, pad_value=0
        )
        if encoder_padding_mask is not None:
            lengths = encoder_padding_mask.seq_lens
        else:
            lengths = [masked_encoder_output.size(1)] * masked_encoder_output.size(0)
        pooled_list = []
        for x, x_len in zip(masked_encoder_output, lengths):
            x = x[:x_len].unsqueeze(0)
            h = torch.tanh(self.h_linear(x))
            w = torch.matmul(h, self.attention.weight.transpose(0, 1)).squeeze(dim=2)
            w = torch.nn.functional.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)
            pooled_list.append(x.squeeze(0))
        pooled_encoder_output = torch.stack(pooled_list)

        return self.projection_out(pooled_encoder_output)
