# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional, Sequence, cast

import torch
from torch import Tensor

from fairseq2.data import Collater, SequenceData, StringLike
from fairseq2.data.text import TextTokenDecoder, TextTokenEncoder, TextTokenizer
from fairseq2.generation.sequence_generator import (
    Seq2SeqGenerator,
    SequenceGeneratorOptions,
    SequenceGeneratorOutput,
)
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.nn.utils.module import infer_device


class SequenceToTextGeneratorBase:
    """Represents an abstract base class for sequence-to-text generators."""

    model: EncoderDecoderModel
    token_decoder: TextTokenDecoder
    generator: Seq2SeqGenerator

    def __init__(
        self,
        model: EncoderDecoderModel,
        tokenizer: TextTokenizer,
        target_lang: Optional[str] = None,
        opts: Optional[SequenceGeneratorOptions] = None,
    ) -> None:
        """
        :param model:
            The encoder-decoder model to use for generation.
        :param tokenizer:
            The text tokenizer to use.
        :param target_lang:
            The target language.
        :param opts:
            The options to pass to the underlying :class:`Seq2SeqGenerator`.
        """
        model.eval()

        self.model = model

        self.token_decoder = tokenizer.create_decoder()

        device = infer_device(model)

        target_encoder = tokenizer.create_encoder(
            task="translation", lang=target_lang, mode="target", device=device
        )

        # Most tokenizers typically use one or more control symbols to indicate
        # the beginning of a sentence.
        self.generator = Seq2SeqGenerator(
            self.model, tokenizer.vocab_info, target_encoder.prefix_indices, opts
        )

    @torch.inference_mode()
    def _do_generate(
        self, source_seqs: Tensor, source_seq_lens: Optional[Tensor]
    ) -> "SequenceToTextOutput":
        """A subclass should call this function for the actual text generation."""
        encoder_output, encoder_padding_mask = self.model.encode(
            source_seqs, source_seq_lens
        )

        gen_output = self.generator(
            encoder_output, encoder_padding_mask, source_seq_len=source_seqs.size(1)
        )

        # TODO: use parallel_invoke
        sentences = [self.token_decoder(b[0].seq)[0] for b in gen_output.results]

        return SequenceToTextOutput(
            sentences, gen_output, encoder_output, encoder_padding_mask
        )


@dataclass
class SequenceToTextOutput:
    """Holds the output of a sequence-to-text generation."""

    sentences: List[StringLike]
    """The generated sentences."""

    generator_output: SequenceGeneratorOutput
    """The output of the underlying :class:`Seq2SeqGenerator`."""

    encoder_output: Tensor
    """The encoder output of the underlying encoder-decoder model used to
    generate the sentences. *Shape:* :math:`(N,S_{enc},M)`, where :math:`N` is
    the batch size, :math:`S_{enc}` is the encoder output sequence length, and
    :math:`M` is the dimensionality of the model."""

    encoder_padding_mask: Optional[Tensor]
    """The float padding mask of :attr:`encoder_output`. *Shape:*
    :math:`(N,S_{enc})`, where :math:`N` is the batch size and :math:`S_{enc}`
    is the encoder output sequence length."""


class SequenceToTextGenerator(SequenceToTextGeneratorBase):
    """Generates text output from input sequences.

    The interpretation of input sequences depends on the underlying encoder-
    decoder model.
    """

    def __call__(
        self,
        source_seqs: Tensor,
        source_seq_lens: Optional[Tensor],
    ) -> List[StringLike]:
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
            The generated text sentences.
        """
        output = self.generate_ex(source_seqs, source_seq_lens)

        return output.sentences

    def generate_ex(
        self,
        source_seqs: Tensor,
        source_seq_lens: Optional[Tensor],
    ) -> SequenceToTextOutput:
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
        """
        return self._do_generate(source_seqs, source_seq_lens)


class TextTranslator(SequenceToTextGeneratorBase):
    """Translates text from one language to another."""

    source_encoder: TextTokenEncoder
    collater: Collater

    def __init__(
        self,
        model: EncoderDecoderModel,
        tokenizer: TextTokenizer,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        opts: Optional[SequenceGeneratorOptions] = None,
    ) -> None:
        """
        :param model:
            The encoder-decoder model to use for translation.
        :param tokenizer:
            The text tokenizer to use.
        :param source_lang:
            The source language.
        :param target_lang:
            The target language.
        :param opts:
            The options to pass to the underlying :class:`Seq2SeqGenerator`.
        """
        super().__init__(model, tokenizer, target_lang, opts)

        device = infer_device(model)

        self.source_encoder = tokenizer.create_encoder(
            task="translation", lang=source_lang, mode="source", device=device
        )

        self.collater = Collater(pad_idx=tokenizer.vocab_info.pad_idx)

    def __call__(self, source_sentences: Sequence[StringLike]) -> List[StringLike]:
        """
        :param source_sentences:
            The sentences in the source language.

        :returns:
            The translated sentences in target language.
        """
        output = self.translate_ex(source_sentences)

        return output.sentences

    def translate_ex(
        self, source_sentences: Sequence[StringLike]
    ) -> SequenceToTextOutput:
        """
        :param source_sentences:
            The sentences in the source language.
        """
        indices: List[Tensor] = []

        # TODO: use parallel_invoke
        for source_sentence in source_sentences:
            indices.append(self.source_encoder(source_sentence))

        batch = cast(SequenceData, self.collater(indices))

        return self._do_generate(
            batch["seqs"], batch["seq_lens"] if batch["is_ragged"] else None
        )
