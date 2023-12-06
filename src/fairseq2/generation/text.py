# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from typing import List, Optional, Sequence, Tuple, final

from torch import Tensor

from fairseq2.data import StringLike
from fairseq2.data.text import TextTokenDecoder, TextTokenEncoder, TextTokenizer
from fairseq2.generation.generator import (
    Seq2SeqGenerator,
    Seq2SeqGeneratorOutput,
    SequenceGenerator,
    SequenceGeneratorOutput,
)
from fairseq2.nn.padding import PaddingMask, pad_seqs
from fairseq2.nn.utils.module import infer_device


class SequenceToTextConverterBase(ABC):
    """Represents an abstract base class for sequence-to-text converters."""

    generator: Seq2SeqGenerator
    target_prefix_seq: Tensor
    text_decoder: TextTokenDecoder

    def __init__(
        self,
        generator: Seq2SeqGenerator,
        tokenizer: TextTokenizer,
        task: str,
        target_lang: Optional[str] = None,
    ) -> None:
        """
        :param generator:
            The sequence-to-sequence generator.
        :param tokenizer:
            The text tokenizer.
        :param task:
            The conversion task (e.g. translation, transcription).
        :param target_lang:
            The target language for conversion.
        """
        self.generator = generator

        device = infer_device(generator.model)

        target_text_encoder = tokenizer.create_encoder(
            task=task, lang=target_lang, mode="target", device=device
        )

        # (S)
        target_prefix_seq = target_text_encoder.prefix_indices
        if target_prefix_seq is None:
            raise ValueError(
                "`tokenizer` must specify a prefix sequence for the target language."
            )

        self.target_prefix_seq = target_prefix_seq

        self.text_decoder = tokenizer.create_decoder()

    def _do_convert(
        self,
        source_seqs: Tensor,
        source_padding_mask: Optional[PaddingMask],
    ) -> Tuple[List[StringLike], Seq2SeqGeneratorOutput]:
        """A subclass should call this method for actual text conversion.

        :param source_seqs:
            The source sequences. *Shape:* :math:`(N,S,*)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`*` is
            any number of sequence-specific dimensions including none.
        :param source_padding_mask:
            The padding mask of ``source_seqs``. *Shape:* :math:`(N,S)`, where
            :math:`N` is the batch size and :math:`S` is the sequence length.

        :returns:
            - The converted texts.
            - The output of the underlying sequence-to-sequence generator.
        """
        batch_size = source_seqs.size(0)

        # (S) -> (N, S)
        target_prefix_seqs = self.target_prefix_seq.expand(batch_size, -1)

        generator_output = self.generator(
            source_seqs, source_padding_mask, target_prefix_seqs, None
        )

        texts: List[StringLike] = []

        for idx, hypotheses in enumerate(generator_output.hypotheses):
            if len(hypotheses) == 0:
                raise RuntimeError(
                    f"The sequence generator returned no hypothesis at index {idx}. Please file a bug report."
                )

            texts.append(self.text_decoder(hypotheses[0].seq))

        return texts, generator_output


@final
class SequenceToTextConverter(SequenceToTextConverterBase):
    """Converts source sequences to text."""

    def __call__(self, source_seq: Tensor) -> Tuple[StringLike, Seq2SeqGeneratorOutput]:
        """
        :param source_seq:
            The source sequence. *Shape:* :math:`(S,*)`, where :math:`S` is the
            sequence length and :math:`*` is any number of sequence-specific
            dimensions including none.

        :returns:
            - The converted text.
            - The output of the underlying sequence-to-sequence generator.
        """
        texts, generator_output = self._do_convert(
            source_seq.unsqueeze(0), source_padding_mask=None
        )

        return texts[0], generator_output

    def batch_convert(
        self,
        source_seqs: Tensor,
        source_padding_mask: Optional[PaddingMask],
    ) -> Tuple[List[StringLike], Seq2SeqGeneratorOutput]:
        """
        :param source_seqs:
            The source sequences. *Shape:* :math:`(N,S,*)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`*` is
            any number of sequence-specific dimensions including none.
        :param source_padding_mask:
            The padding mask of ``source_seqs``. *Shape:* :math:`(N,S)`, where
            :math:`N` is the batch size and :math:`S` is the sequence length.

        :returns:
            - The converted texts.
            - The output of the underlying sequence-to-sequence generator.
        """
        if len(source_seqs) == 0:
            raise ValueError(
                "`source_seqs` must contain at least one element, but is empty instead."
            )

        return self._do_convert(source_seqs, source_padding_mask)


@final
class TextTranslator(SequenceToTextConverterBase):
    """Translates text from one language to another."""

    pad_idx: int
    source_text_encoder: TextTokenEncoder

    def __init__(
        self,
        generator: Seq2SeqGenerator,
        tokenizer: TextTokenizer,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> None:
        """
        :param generator:
            The sequence-to-sequence generator.
        :param tokenizer:
            The text tokenizer.
        :param source_lang:
            The source language.
        :param target_lang:
            The target language.
        """
        super().__init__(generator, tokenizer, "translation", target_lang)

        pad_idx = tokenizer.vocab_info.pad_idx
        if pad_idx is None:
            raise ValueError(
                "``vocab_info` of `tokenizer` must have a PAD symbol defined."
            )

        self.pad_idx = pad_idx

        device = infer_device(generator.model)

        self.source_text_encoder = tokenizer.create_encoder(
            task="translation", lang=source_lang, mode="source", device=device
        )

    def __call__(
        self, source_text: StringLike
    ) -> Tuple[StringLike, Seq2SeqGeneratorOutput]:
        """
        :param source_text:
            The text in the source language.

        :returns:
            - The translated text.
            - The output of the underlying sequence-to-sequence generator.
        """
        source_seq = self.source_text_encoder(source_text)

        translations, generator_output = self._do_convert(
            source_seq.unsqueeze(0), source_padding_mask=None
        )

        return translations[0], generator_output

    def batch_translate(
        self, source_texts: Sequence[StringLike]
    ) -> Tuple[List[StringLike], Seq2SeqGeneratorOutput]:
        """
        :param source_texts:
            The texts in the source language.

        :returns:
            - The translated texts.
            - The output of the underlying sequence-to-sequence generator.
        """
        if len(source_texts) == 0:
            raise ValueError(
                "`source_texts` must contain at least one element, but is empty instead."
            )

        source_seq_list = [self.source_text_encoder(t) for t in source_texts]

        source_seqs, source_padding_mask = pad_seqs(source_seq_list, self.pad_idx)

        return self._do_convert(source_seqs, source_padding_mask)


class TextCompleter:
    """Completes text prompts."""

    generator: SequenceGenerator
    text_encoder: TextTokenEncoder
    text_decoder: TextTokenDecoder

    def __init__(self, generator: SequenceGenerator, tokenizer: TextTokenizer) -> None:
        """
        :param generator:
            The sequence generator.
        :param tokenizer:
            The text tokenizer.
        """
        self.generator = generator

        device = infer_device(generator.model)

        self.text_encoder = tokenizer.create_encoder(mode="prompt", device=device)
        self.text_decoder = tokenizer.create_decoder()

    def __call__(
        self, prompt: StringLike
    ) -> Tuple[StringLike, SequenceGeneratorOutput]:
        """
        :param prompt:
            The text prompt.

        :returns:
            - The completed text.
            - The output of the underlying sequence generator.
        """
        prompt_seq = self.text_encoder(prompt)

        texts, generator_output = self._do_complete(
            prompt_seq.unsqueeze(0), prompt_padding_mask=None
        )

        return texts[0], generator_output

    def batch_complete(
        self, prompts: Sequence[StringLike]
    ) -> Tuple[List[StringLike], SequenceGeneratorOutput]:
        """
        :param prompts:
            The text prompts.

        :returns:
            - The completed texts.
            - The output of the underlying sequence generator.
        """
        if len(prompts) == 0:
            raise ValueError(
                "`prompts` must contain at least one element, but is empty instead."
            )

        prompt_seq_list = [self.text_encoder(p) for p in prompts]

        prompt_seqs, prompt_padding_mask = pad_seqs(prompt_seq_list)

        return self._do_complete(prompt_seqs, prompt_padding_mask)

    def _do_complete(
        self, prompt_seqs: Tensor, prompt_padding_mask: Optional[PaddingMask]
    ) -> Tuple[List[StringLike], SequenceGeneratorOutput]:
        generator_output = self.generator(prompt_seqs, prompt_padding_mask)

        texts: List[StringLike] = []

        for idx, hypotheses in enumerate(generator_output.hypotheses):
            if len(hypotheses) == 0:
                raise RuntimeError(
                    f"The sequence generator returned no hypothesis at index {idx}. Please file a bug report."
                )

            texts.append(self.text_decoder(hypotheses[0].seq))

        return texts, generator_output
