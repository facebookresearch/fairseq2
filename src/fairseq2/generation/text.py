# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from torch import Tensor

from fairseq2.data.text.tokenizers import (
    TextTokenDecoder,
    TextTokenEncoder,
    TextTokenizer,
)
from fairseq2.error import ContractError
from fairseq2.generation import (
    Seq2SeqGenerator,
    Seq2SeqGeneratorOutput,
    SequenceGenerator,
    SequenceGeneratorOutput,
)
from fairseq2.nn.padding import PaddingMask, pad_seqs
from fairseq2.nn.utils.module import infer_device


@final
class SequenceToTextConverter:
    """Converts source sequences to text."""

    _generator: Seq2SeqGenerator
    _target_prefix_seq: Tensor
    _text_decoder: TextTokenDecoder

    def __init__(
        self,
        generator: Seq2SeqGenerator,
        tokenizer: TextTokenizer,
        task: str,
        target_lang: str | None = None,
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
        self._generator = generator

        try:
            device = infer_device(generator.model)
        except ValueError as ex:
            raise ValueError(
                "The device of `generator.model` is not valid. See the nested exception for details."
            ) from ex

        target_text_encoder = tokenizer.create_encoder(
            task=task, lang=target_lang, mode="target", device=device
        )

        # (S)
        target_prefix_seq = target_text_encoder.prefix_indices
        if target_prefix_seq is None:
            raise ValueError(
                "`tokenizer` must specify a prefix sequence for the target language."
            )

        self._target_prefix_seq = target_prefix_seq

        self._text_decoder = tokenizer.create_decoder()

    def __call__(self, source_seq: Tensor) -> tuple[str, Seq2SeqGeneratorOutput]:
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
        source_padding_mask: PaddingMask | None,
    ) -> tuple[list[str], Seq2SeqGeneratorOutput]:
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

    def _do_convert(
        self,
        source_seqs: Tensor,
        source_padding_mask: PaddingMask | None,
    ) -> tuple[list[str], Seq2SeqGeneratorOutput]:
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
        target_prefix_seqs = self._target_prefix_seq.expand(batch_size, -1)

        generator_output = self._generator(
            source_seqs, source_padding_mask, target_prefix_seqs, None
        )

        texts: list[str] = []

        for idx, hypotheses in enumerate(generator_output.hypotheses):
            if len(hypotheses) == 0:
                raise ContractError(
                    f"The sequence generator returned no hypothesis at index {idx}."
                )

            texts.append(self._text_decoder(hypotheses[0].seq))

        return texts, generator_output


@final
class TextTranslator:
    """Translates text from one language to another."""

    _converter: SequenceToTextConverter
    _pad_idx: int
    _source_text_encoder: TextTokenEncoder
    _max_source_len: int | None

    def __init__(
        self,
        generator: Seq2SeqGenerator,
        tokenizer: TextTokenizer,
        source_lang: str | None = None,
        target_lang: str | None = None,
        *,
        max_source_len: int | None = None,
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
        :param max_source_len:
            The maximum number of tokens above which the source sequence gets
            truncated.
        """
        self._converter = SequenceToTextConverter(
            generator, tokenizer, "translation", target_lang
        )

        pad_idx = tokenizer.vocab_info.pad_idx
        if pad_idx is None:
            raise ValueError(
                "``vocab_info` of `tokenizer` must have a PAD symbol defined."
            )

        self._pad_idx = pad_idx

        try:
            device = infer_device(generator.model)
        except ValueError as ex:
            raise ValueError(
                "The device of `generator.model` is not valid. See the nested exception for details."
            ) from ex

        self._source_text_encoder = tokenizer.create_encoder(
            task="translation", lang=source_lang, mode="source", device=device
        )

        if max_source_len is not None and max_source_len <= 0:
            raise ValueError(
                f"`max_source_len` must be greater than or equal to 1, but is {max_source_len} instead."
            )

        self._max_source_len = max_source_len

    def __call__(self, source_text: str) -> tuple[str, Seq2SeqGeneratorOutput]:
        """
        :param source_text:
            The text in the source language.

        :returns:
            - The translated text.
            - The output of the underlying sequence-to-sequence generator.
        """
        source_seq = self._source_text_encoder(source_text)

        if self._max_source_len:
            source_seq = source_seq[: self._max_source_len]

        return self._converter(source_seq)

    def batch_translate(
        self, source_texts: Sequence[str]
    ) -> tuple[list[str], Seq2SeqGeneratorOutput]:
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

        source_seq_list = [self._source_text_encoder(t) for t in source_texts]

        if self._max_source_len:
            source_seq_list = [seq[: self._max_source_len] for seq in source_seq_list]

        source_seqs, source_padding_mask = pad_seqs(source_seq_list, self._pad_idx)

        return self._converter.batch_convert(source_seqs, source_padding_mask)


@final
class TextCompleter:
    """Completes text prompts."""

    _generator: SequenceGenerator
    _text_encoder: TextTokenEncoder
    _text_decoder: TextTokenDecoder

    def __init__(self, generator: SequenceGenerator, tokenizer: TextTokenizer) -> None:
        """
        :param generator:
            The sequence generator.
        :param tokenizer:
            The text tokenizer.
        """
        self._generator = generator

        try:
            device = infer_device(generator.model)
        except ValueError as ex:
            raise ValueError(
                "The device of `generator.model` is not valid. See the nested exception for details."
            ) from ex

        self._text_encoder = tokenizer.create_encoder(mode="prompt", device=device)
        self._text_decoder = tokenizer.create_decoder()

    def __call__(self, prompt: str) -> tuple[str, SequenceGeneratorOutput]:
        """
        :param prompt:
            The text prompt.

        :returns:
            - The completed text.
            - The output of the underlying sequence generator.
        """
        prompt_seq = self._text_encoder(prompt)

        texts, generator_output = self._do_complete(
            prompt_seq.unsqueeze(0), prompt_padding_mask=None
        )

        return texts[0], generator_output

    def batch_complete(
        self, prompts: Sequence[str]
    ) -> tuple[list[str], SequenceGeneratorOutput]:
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

        prompt_seq_list = [self._text_encoder(p) for p in prompts]

        prompt_seqs, prompt_padding_mask = pad_seqs(prompt_seq_list)

        return self._do_complete(prompt_seqs, prompt_padding_mask)

    def _do_complete(
        self, prompt_seqs: Tensor, prompt_padding_mask: PaddingMask | None
    ) -> tuple[list[str], SequenceGeneratorOutput]:
        generator_output = self._generator(prompt_seqs, prompt_padding_mask)

        texts: list[str] = []

        for idx, hypotheses in enumerate(generator_output.hypotheses):
            if len(hypotheses) == 0:
                raise ContractError(
                    f"The sequence generator returned no hypothesis at index {idx}."
                )

            texts.append(self._text_decoder(hypotheses[0].seq))

        return texts, generator_output
