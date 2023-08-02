# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from fairseq2.data.text import TextTokenizer
from fairseq2.generation import (
    Seq2SeqGenerator,
    SequenceGeneratorOptions,
    SequenceGeneratorOutput,
    SequenceToTextGenerator,
    SequenceToTextOutput,
)
from fairseq2.models.unity.model import UnitYModel
from fairseq2.models.unity.unit_tokenizer import UnitTokenDecoder, UnitTokenizer
from fairseq2.nn.utils.module import infer_device


class UnitYGenerator:
    """Generates text translations and speech units from a UnitY model."""

    model: UnitYModel
    text_generator: SequenceToTextGenerator
    unit_decoder: UnitTokenDecoder
    unit_generator: Seq2SeqGenerator

    def __init__(
        self,
        model: UnitYModel,
        text_tokenizer: TextTokenizer,
        unit_tokenizer: UnitTokenizer,
        target_lang: str,
        text_opts: Optional[SequenceGeneratorOptions] = None,
        unit_opts: Optional[SequenceGeneratorOptions] = None,
    ) -> None:
        """
        :param model:
            The UnitY model to use for generation.
        :param text_tokenizer:
            The text tokenizer to use.
        :param unit_tokenizer:
            The unit tokenizer to use.
        :param target_lang:
            The target language.
        :param text_generator_opts:
            The options to pass to the underlying text :class:`Seq2SeqGenerator`.
        :param unit_generator_opts:
            The options to pass to the underlying unit :class:`Seq2SeqGenerator`.
        """
        if model.t2u_model is None:
            raise ValueError(
                "`model` does not have a T2U sub-model. For text generation only, use `SequenceToTextGenerator` instead."
            )

        model.eval()

        self.model = model

        self.text_generator = SequenceToTextGenerator(
            model, text_tokenizer, target_lang, text_opts
        )

        # Set up unit generator.
        self.unit_decoder = unit_tokenizer.create_decoder()

        unit_encoder = unit_tokenizer.create_encoder(
            lang=target_lang, device=infer_device(model.t2u_model)
        )

        if unit_opts is None:
            # Speech sequences are typically much longer than text sequences.
            unit_opts = SequenceGeneratorOptions(
                soft_max_seq_len=(1, 50), hard_max_seq_len=5000
            )

        self.unit_generator = Seq2SeqGenerator(
            model.t2u_model,
            unit_tokenizer.vocab_info,
            unit_encoder.prefix_indices,
            unit_opts,
        )

    @torch.inference_mode()
    def __call__(
        self, source_seqs: Tensor, source_seq_lens: Optional[Tensor]
    ) -> Tuple[SequenceToTextOutput, "SequenceToUnitOutput"]:
        """
        :param source_seqs:
            The source sequences to use for generation. *Shape:* :math:`(N,S,*)`,
            where :math:`N` is the batch size, :math:`S` is the sequence length,
            and :math:`*` is any number of sequence-specific dimensions
            including none.
        :param source_seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``source_seqs``. *Shape:* :math:`(N)`, where
            :math:`N` is the batch size.

        :returns:
            - The output of the text generator.
            - The output of the unit generator.
        """
        text_output = self.text_generator.generate_ex(source_seqs, source_seq_lens)

        text_seqs, text_seq_lens = text_output.generator_output.collate()

        # Use the output of the text generator to compute the decoder output.
        decoder_output, decoder_padding_mask = self.model.decode(
            text_seqs,
            text_seq_lens,
            text_output.encoder_output,
            text_output.encoder_padding_mask,
        )

        assert self.model.t2u_model is not None

        t2u_encoder_output, t2u_encoder_padding_mask = self.model.t2u_model.encode(
            decoder_output, decoder_padding_mask
        )

        unit_gen_output = self.unit_generator(
            t2u_encoder_output,
            t2u_encoder_padding_mask,
            source_seq_len=source_seqs.size(1),
        )

        unit_seqs, _ = unit_gen_output.collate()

        # Convert to speech units.
        units = self.unit_decoder(unit_seqs)

        unit_output = SequenceToUnitOutput(
            units, unit_gen_output, t2u_encoder_output, t2u_encoder_padding_mask
        )

        return text_output, unit_output


@dataclass
class SequenceToUnitOutput:
    units: Tensor
    """The generated units."""

    generator_output: SequenceGeneratorOutput
    """The output of the underlying :class:`Seq2SeqGenerator`."""

    t2u_encoder_output: Tensor
    """The encoder output of the underlying UnitY T2U model used to generate the
    units. *Shape:* :math:`(N,S_{enc},M)`, where :math:`N` is the batch size,
    :math:`S_{enc}` is the encoder output sequence length, and :math:`M` is the
    dimensionality of the model."""

    t2u_encoder_padding_mask: Optional[Tensor]
    """The float padding mask of :attr:`encoder_output`. *Shape:*
    :math:`(N,S_{enc})`, where :math:`N` is the batch size and :math:`S_{enc}`
    is the encoder output sequence length."""
