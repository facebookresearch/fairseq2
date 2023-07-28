# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from fairseq2.data import StringLike
from fairseq2.data.text import TextTokenDecoder, TextTokenizer
from fairseq2.models.unity.model import UnitYModel
from fairseq2.models.unity.unit_tokenizer import UnitTokenDecoder, UnitTokenizer
from fairseq2.nn.utils.module import infer_device
from fairseq2.sequence_generator import Seq2SeqGenerator, SequenceOptions


class UnitYGenerator:
    """Generates text translations and speech units from a provided UnitY model."""

    model: UnitYModel
    text_token_decoder: TextTokenDecoder
    text_prefix_seq: Tensor
    text_seq_generator: Seq2SeqGenerator
    unit_token_decoder: UnitTokenDecoder
    unit_prefix_seq: Tensor
    unit_seq_generator: Seq2SeqGenerator

    def __init__(
        self,
        model: UnitYModel,
        text_tokenizer: TextTokenizer,
        unit_tokenizer: UnitTokenizer,
        target_lang: str,
        text_seq_opts: Optional[SequenceOptions] = None,
        unit_seq_opts: Optional[SequenceOptions] = None,
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
        :param text_seq_opts:
            The options to pass to the underlying text :class:`Seq2SeqGenerator`.
        :param unit_seq_opts:
            The options to pass to the underlying unit :class:`Seq2SeqGenerator`.
        """
        model.eval()

        self.model = model

        self.text_token_decoder = text_tokenizer.create_decoder()

        device = infer_device(model)

        # Set up text sequence generator.
        text_token_encoder = text_tokenizer.create_encoder(
            task="translation", lang=target_lang, mode="target", device=device
        )

        self.text_prefix_seq = text_token_encoder("")

        self.text_seq_generator = Seq2SeqGenerator(
            model, text_tokenizer.vocabulary_info, text_seq_opts
        )

        # Set up unit sequence generator.
        self.unit_token_decoder = unit_tokenizer.create_decoder()

        unit_token_encoder = unit_tokenizer.create_encoder(lang=target_lang)

        self.unit_prefix_seq = unit_token_encoder(
            torch.empty((1, 0), device=device, dtype=torch.int)
        )

        self.unit_seq_generator = Seq2SeqGenerator(
            model.t2u_model, unit_tokenizer.vocabulary_info, unit_seq_opts
        )

    @torch.inference_mode()
    def __call__(
        self,
        source_seqs: Tensor,
        source_seq_lens: Optional[Tensor],
        text_only: bool = False,
    ) -> Tuple[List[StringLike], Optional[Tensor]]:
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
        :param text_only:
            If ``True``, generates text output only, and skips speech unit
            generation.

        :returns:
            - The generated text sentences.
            - The generated speech units.
        """
        encoder_output, encoder_padding_mask = self.model.encode(
            source_seqs, source_seq_lens
        )

        text_seqs, text_seq_lens = self.text_seq_generator(
            self.text_prefix_seq, encoder_output, encoder_padding_mask
        )

        sentences = self.text_token_decoder(text_seqs)

        if text_only:
            return sentences, None

        # Use the output of the text generator to compute the decoder output.
        decoder_output, decoder_padding_mask = self.model.decode(
            text_seqs, text_seq_lens, encoder_output, encoder_padding_mask
        )

        encoder_output, encoder_padding_mask = self.model.t2u_model.encode(
            decoder_output, decoder_padding_mask
        )

        unit_seqs, _ = self.unit_seq_generator(
            self.unit_prefix_seq, encoder_output, encoder_padding_mask
        )

        units = self.unit_token_decoder(unit_seqs)

        return sentences, units
