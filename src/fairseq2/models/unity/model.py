# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, final

from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Module

from fairseq2.models.encoder_decoder import EncoderDecoderModel, Seq2SeqDecoder
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.models.transformer.frontend import TransformerFrontend
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.projection import Projection
from fairseq2.nn.transformer import TransformerDecoder, TransformerEncoder


@final
class UnitYModel(EncoderDecoderModel):
    """Represents a UnitY model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2212.08055`.

    Note that this implementation is augmented with a text encoder to enable
    translating from text.
    """

    model_dim: int
    default_input_modality: str
    speech_encoder_frontend: TransformerFrontend
    speech_encoder: TransformerEncoder
    text_encoder_frontend: Optional[TransformerFrontend]
    text_encoder: Optional[TransformerEncoder]
    text_decoder_frontend: TransformerFrontend
    text_decoder: TransformerDecoder
    final_proj: Projection
    t2u_model: "UnitYT2UModel"
    pad_idx: Optional[int]

    def __init__(
        self,
        speech_encoder_frontend: TransformerFrontend,
        speech_encoder: TransformerEncoder,
        text_encoder_frontend: Optional[TransformerFrontend],
        text_encoder: Optional[TransformerEncoder],
        text_decoder_frontend: TransformerFrontend,
        text_decoder: TransformerDecoder,
        final_proj: Projection,
        t2u_model: "UnitYT2UModel",
        pad_idx: Optional[int],
        default_input_modality: Literal["speech", "text"] = "speech",
    ) -> None:
        model_dim = speech_encoder.model_dim

        super().__init__(model_dim)

        self.default_input_modality = default_input_modality

        self.speech_encoder_frontend = speech_encoder_frontend
        self.speech_encoder = speech_encoder

        self.text_encoder_frontend = text_encoder_frontend
        self.text_encoder = text_encoder

        self.text_decoder_frontend = text_decoder_frontend
        self.text_decoder = text_decoder

        self.final_proj = final_proj

        self.t2u_model = t2u_model

        self.pad_idx = pad_idx

    @finaloverride
    def encode(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.default_input_modality == "speech":
            return self.encode_speech(seqs, seq_lens)

        if self.default_input_modality == "text":
            return self.encode_text(seqs, seq_lens)

        raise RuntimeError(
            f"`default_input_modality` must be 'speech' or 'text', but is '{self.default_input_modality}' instead."
        )

    def encode_speech(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs, padding_mask = self.speech_encoder_frontend(seqs, seq_lens)

        return self.speech_encoder(seqs, padding_mask)  # type: ignore[no-any-return]

    def encode_text(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.text_encoder is None or self.text_encoder_frontend is None:
            raise ValueError(
                "MT task requires a text encoder, but the current UnitY model does not have one."
            )

        seqs, padding_mask = self.text_encoder_frontend(seqs, seq_lens)

        return self.text_encoder(seqs, padding_mask)  # type: ignore[no-any-return]

    @finaloverride
    def decode(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs, padding_mask = self.text_decoder_frontend(seqs, seq_lens, state_bag)

        return self.text_decoder(  # type: ignore[no-any-return]
            seqs, padding_mask, encoder_output, encoder_padding_mask, state_bag
        )

    @finaloverride
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[Tensor]
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.pad_idx)


@final
class UnitYT2UModel(Module, Seq2SeqDecoder):
    """Represents a UnitY T2U model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2212.08055`."""

    encoder: Optional[TransformerEncoder]
    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection
    pad_idx: Optional[int]

    def __init__(
        self,
        encoder: Optional[TransformerEncoder],
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        pad_idx: Optional[int],
    ) -> None:
        super().__init__()

        self.encoder = encoder

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

        self.pad_idx = pad_idx

    def forward(self, batch: Seq2SeqBatch) -> SequenceModelOutput:
        encoder_output, encoder_padding_mask = self.encode(
            batch.source_seqs, batch.source_seq_lens
        )

        decoder_output, decoder_padding_mask = self.decode(
            batch.target_seqs,
            batch.target_seq_lens,
            encoder_output,
            encoder_padding_mask,
        )

        return self.project(decoder_output, decoder_padding_mask)

    def encode(
        self,
        text_decoder_output: Tensor,
        text_decoder_padding_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.encoder is None:
            return text_decoder_output, text_decoder_padding_mask

        return self.encoder(text_decoder_output, text_decoder_padding_mask)  # type: ignore[no-any-return]

    def decode(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs, padding_mask = self.decoder_frontend(seqs, seq_lens)

        return self.decoder(  # type: ignore[no-any-return]
            seqs, padding_mask, encoder_output, encoder_padding_mask, state_bag
        )

    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[Tensor]
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.pad_idx)


@dataclass
class UnitYOutput:
    """Holds the output of a UnitY model."""

    s2t_output: SequenceModelOutput
    """The S2T output of the multitask model."""

    mt_output: SequenceModelOutput
    """The MT output of the multitask model."""

    t2u_output: SequenceModelOutput
    """The output of the T2U model."""

    def compute_loss(
        self, targets: Tensor, ignore_prefix_size: int = 0, label_smoothing: float = 0.0
    ) -> None:
        # TODO: Implement R-Drop based loss
        pass
