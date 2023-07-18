# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple

from torch import Tensor
from torch.nn import Module, Sequential

from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.models.transformer import TransformerModel
from fairseq2.models.transformer.frontend import TransformerFrontend
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.projection import Linear, Projection
from fairseq2.nn.transformer import TransformerDecoder, TransformerEncoder


class UnitYModel(Module):
    """Represents a UnitY model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2212.08055`."""

    model_dim: int
    s2t_model: TransformerModel
    t2u_proj: Optional[Sequential]
    t2u_encoder: Optional[TransformerEncoder]
    t2u_decoder_frontend: TransformerFrontend
    t2u_decoder: TransformerDecoder
    final_proj: Projection
    target_pad_idx: Optional[int]

    def __init__(
        self,
        s2t_model: TransformerModel,
        t2u_encoder: Optional[TransformerEncoder],
        t2u_decoder_frontend: TransformerFrontend,
        t2u_decoder: TransformerDecoder,
        final_proj: Projection,
        target_pad_idx: Optional[int],
    ) -> None:
        """
        :param s2t_model:
            The S2T UnitY model.
        :param t2u_encoder:
            The T2U encoder.
        :param t2u_decoder_frontend:
            The T2U decoder frontend.
        :param t2u_decoder:
            The T2U decoder.
        :param final_proj:
            The projection to apply to T2U decoder outputs to produce logits.
        :param target_pad_idx:
            The index of the pad symbol in the unit vocabulary.
        """
        super().__init__()

        self.model_dim = s2t_model.model_dim

        self.s2t_model = s2t_model

        if t2u_encoder is not None:
            if t2u_encoder.model_dim != self.model_dim:
                raise ValueError(
                    f"`model_dim` of `s2t_model` and `model_dim` of `t2u_encoder` must be equal, but are {self.model_dim} and {t2u_encoder.model_dim} instead."
                )

            self.t2u_proj = Sequential(
                Linear(self.model_dim, self.model_dim, bias=False),
                Linear(self.model_dim, self.model_dim, bias=False),
            )

            self.t2u_encoder = t2u_encoder

        if t2u_decoder_frontend.model_dim != self.model_dim:
            raise ValueError(
                f"`model_dim` of `s2t_model` and `model_dim` of `t2u_decoder_frontend` must be equal, but are {self.model_dim} and {t2u_decoder_frontend.model_dim} instead."
            )

        if t2u_decoder.model_dim != self.model_dim:
            raise ValueError(
                f"`model_dim` of `s2t_model` and `model_dim` of `t2u_decoder` must be equal, but are {self.model_dim} and {t2u_decoder_frontend.model_dim} instead."
            )

        self.t2u_decoder_frontend = t2u_decoder_frontend
        self.t2u_decoder = t2u_decoder

        self.final_proj = final_proj

        self.target_pad_idx = target_pad_idx

    def forward(self, batch: "UnitYBatch") -> "UnitYOutput":
        """
        :param batch:
            The batch of speech, text, and unit sequences to process.
        """
        s2t_encoder_output, s2t_encoder_padding_mask = self.s2t_model.encode(
            batch.speech_seqs, batch.speech_seq_lens
        )

        s2t_decoder_output, s2t_decoder_padding_mask = self.s2t_model.decode(
            batch.text_seqs,
            batch.text_seq_lens,
            s2t_encoder_output,
            s2t_encoder_padding_mask,
        )

        s2t_output = self.s2t_model.project(
            s2t_decoder_output, s2t_decoder_padding_mask
        )

        t2u_encoder_output, t2u_encoder_padding_mask = self.encode(
            s2t_decoder_output, s2t_decoder_padding_mask
        )

        t2u_decoder_output, t2u_decoder_padding_mask = self.decode(
            batch.unit_seqs,
            batch.unit_seq_lens,
            t2u_encoder_output,
            t2u_encoder_padding_mask,
        )

        t2u_output = self.final_proj(t2u_decoder_output, t2u_decoder_padding_mask)

        return UnitYOutput(t2u_output, s2t_output)

    def encode(
        self,
        s2t_decoder_output: Tensor,
        s2t_decoder_padding_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Encode the specified S2T decoder output for unit generation.

        :param s2t_decoder_output:
            The S2T decoder output to encode. *Shape:* :math:`(N,S_{dec},M)`,
            where :math:`N` is the batch size, :math:`S_{dec}` is the decoder
            output sequence length, and :math:`M` is the dimensionality of the
            model.
        :param s2t_decoder_padding_mask:
            The float padding mask of ``s2t_decoder_out``. *Shape:*
            :math:`(N,S_{dec})`, where :math:`N` is the batch size and
            :math:`S_{dec}` is the decoder output sequence length.

        :returns:
            - The encoder output. *Shape:* :math:`(N,S_{out},M)`, where
              :math:`N` is the batch size, :math:`S_{out}` is the output
              sequence length, and :math:`M` is the dimensionality of the model.
            - The float padding mask of the encoder output. *Shape:*
              :math:`(N,S_{out})`, where :math:`N` is the batch size and
              :math:`S_{out}` is the output sequence length.
        """
        if self.t2u_encoder is None:
            return s2t_decoder_output, s2t_decoder_padding_mask

        assert self.t2u_proj is not None

        seqs = self.t2u_proj(s2t_decoder_output)

        return self.t2u_encoder(seqs, s2t_decoder_padding_mask)  # type: ignore[no-any-return]

    def decode(
        self,
        unit_seqs: Tensor,
        unit_seq_lens: Optional[Tensor],
        t2u_encoder_output: Tensor,
        t2u_encoder_padding_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Decode the specified unit sequences.

        :param unit_seqs:
            The unit sequences to decode. *Shape:* :math:`(N,S_{unt})`, where
            :math:`N` is the batch size and :math:`S_{unt}` is the unit sequence
            length.
        :param unit_seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``unit_seqs``. *Shape:* :math:`(N)`, where
            :math:`N` is the batch size.
        :param t2u_encoder_output:
            The T2U encoder output to use in encoder-decoder attention. *Shape:*
            :math:`(N,S_{enc},M)`, where :math:`N` is the batch size,
            :math:`S_{enc}` is the encoder output sequence length, and :math:`M`
            is the dimensionality of the model.
        :param t2u_encoder_padding_mask:
            The float padding mask of ``t2u_encoder_out``. *Shape:*
            :math:`(N,S_{enc})`, where :math:`N` is the batch size and
            :math:`S_{enc}` is the encoder output sequence length.
        :param state_bag:
            The state bag to use for incremental evaluation.

        :returns:
            - The T2U decoder output. *Shape:* :math:`(N,S_{unt},M)`, where
              :math:`N` is the batch size, :math:`S_{unt}` is the unit sequence
              length, and :math:`M` is the dimensionality of the model.
            - The float padding mask of the T2U decoder output. *Shape:*
              :math:`(N,S_{unt})`, where :math:`N` is the batch size and
              :math:`S_{unt}` is the unit sequence length.
        """
        if state_bag is None:
            unit_seqs = unit_seqs[:, :-1]

            if unit_seq_lens is not None:
                unit_seq_lens = unit_seq_lens - 1

        unit_seqs, unit_padding_mask = self.t2u_decoder_frontend(
            unit_seqs, unit_seq_lens
        )

        t2u_decoder_output, t2u_decoder_padding_mask = self.t2u_decoder(
            unit_seqs,
            unit_padding_mask,
            t2u_encoder_output,
            t2u_encoder_padding_mask,
            state_bag,
        )

        return t2u_decoder_output, t2u_decoder_padding_mask

    def project(
        self, t2u_decoder_output: Tensor, t2u_decoder_padding_mask: Optional[Tensor]
    ) -> SequenceModelOutput:
        """Produce logits for next-step unit prediction.

        :param t2u_decoder_output:
            The T2U decoder output. *Shape:* :math:`(N,S_{unt},M)`, where
            :math:`N` is the batch size, :math:`S_{unt}` is the unit sequence
            length, and :math:`M` is the dimensionality of the model.
        :param t2u_decoder_padding_mask:
            The float padding mask of the T2U decoder output. *Shape:*
            :math:`(N,S_{unt})`, where :math:`N` is the batch size and
            :math:`S_{unt}` is the unit sequence length.
        """
        logits = self.final_proj(t2u_decoder_output)

        return SequenceModelOutput(logits, self.target_pad_idx)


@dataclass
class UnitYBatch:
    """Represents a batch to be passed to a :class:`UnitYModel` instance."""

    speech_seqs: Tensor
    """The speech sequences. *Shape:* :math:`(N,S_{sph},*)`, where :math:`N` is
    the batch size, :math:`S_{sph}` is the speech sequence length, and :math:`*`
    is any number of sequence-specific dimensions."""

    speech_seq_lens: Optional[Tensor]
    """An array where each element represents the length of the sequence at the
    same index in :attr:`speech_seqs`. *Shape:* :math:`(N)`, where :math:`N` is
    the batch size."""

    text_seqs: Tensor
    """The text sequences. *Shape:* :math:`(N,S_{txt})`, where :math:`N` is the
    batch size and :math:`S_{src}` is the text sequence length."""

    text_seq_lens: Optional[Tensor]
    """An array where each element represents the length of the sequence at the
    same index in :attr:`text_seqs`. *Shape:* :math:`(N)`, where :math:`N` is
    the batch size."""

    unit_seqs: Tensor
    """The unit sequences. *Shape:* :math:`(N,S_{unt})`, where :math:`N` is the
    batch size and :math:`S_{unt}` is the unit sequence length."""

    unit_seq_lens: Optional[Tensor]
    """An array where each element represents the length of the sequence at the
    same index in :attr:`unit_seqs`. *Shape:* :math:`(N)`, where :math:`N` is
    the batch size."""


@dataclass
class UnitYOutput:
    """Holds the output of a UnitY model."""

    s2t_output: SequenceModelOutput
    """The output of the S2T model."""

    t2u_output: SequenceModelOutput
    """The output of the T2U encoder/decoder."""

    def compute_loss(
        self, targets: Tensor, ignore_prefix_size: int = 0, label_smoothing: float = 0.0
    ) -> None:
        # TODO: Implement R-Drop based loss
        pass
