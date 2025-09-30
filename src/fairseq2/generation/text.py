# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from torch import Tensor

from fairseq2.data.tokenizers import Tokenizer
from fairseq2.error import InternalError
from fairseq2.generation.generator import (
    Seq2SeqGenerator,
    SequenceGenerator,
    SequenceGeneratorOutput,
)
from fairseq2.nn import BatchLayout
from fairseq2.nn.utils.module import maybe_infer_device
from fairseq2.nn.utils.padding import pad_seqs


@final
class SequenceToTextConverter:
    """Converts source sequences to text."""

    def __init__(
        self,
        generator: Seq2SeqGenerator,
        tokenizer: Tokenizer,
        task: str,
        target_lang: str | None = None,
        *,
        skip_special_tokens: bool = False,
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

        device = maybe_infer_device(generator.model)
        if device is None:
            raise ValueError(
                "All parameters and buffers of `generator.model` must be on the same device."
            )

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

        self._text_decoder = tokenizer.create_decoder(
            skip_special_tokens=skip_special_tokens
        )

    def __call__(self, source_seq: Tensor) -> tuple[str, SequenceGeneratorOutput]:
        """
        :param source_seq:
            The source sequence. *Shape:* :math:`(S,*)`, where :math:`S` is the
            sequence length and :math:`*` is any number of sequence-specific
            dimensions including none.

        :returns:
            - The converted text.
            - The output of the underlying sequence-to-sequence generator.
        """
        # (S, *) -> (1, S, *)
        source_seqs = source_seq.unsqueeze(0)

        source_seqs_layout = BatchLayout.of(source_seqs)

        texts, generator_output = self._do_convert(source_seqs, source_seqs_layout)

        return texts[0], generator_output

    def batch_convert(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
    ) -> tuple[list[str], SequenceGeneratorOutput]:
        """
        :param source_seqs:
            The source sequences. *Shape:* :math:`(N,S,*)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`*` is
            any number of sequence-specific dimensions including none.

        :returns:
            - The converted texts.
            - The output of the underlying sequence-to-sequence generator.
        """
        if len(source_seqs) == 0:
            raise ValueError(
                "`source_seqs` must contain at least one element, but is empty instead."
            )

        return self._do_convert(source_seqs, source_seqs_layout)

    def _do_convert(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
    ) -> tuple[list[str], SequenceGeneratorOutput]:
        """A subclass should call this method for actual text conversion.

        :param source_seqs:
            The source sequences. *Shape:* :math:`(N,S,*)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`*` is
            any number of sequence-specific dimensions including none.

        :returns:
            - The converted texts.
            - The output of the underlying sequence-to-sequence generator.
        """
        batch_size = source_seqs.size(0)

        # (S) -> (N, S)
        target_prefix_seqs = self._target_prefix_seq.expand(batch_size, -1)

        target_prefix_seqs_layout = BatchLayout.of(target_prefix_seqs)

        generator_output = self._generator(
            source_seqs,
            source_seqs_layout,
            prompt_seqs=target_prefix_seqs,
            prompt_seqs_layout=target_prefix_seqs_layout,
        )

        texts: list[str] = []

        for idx, hypotheses in enumerate(generator_output.hypotheses):
            if len(hypotheses) == 0:
                raise InternalError(
                    f"Sequence generator returned no hypothesis at index {idx}."
                )

            text = self._text_decoder(hypotheses[0].seq)

            texts.append(text)

        return texts, generator_output


@final
class TextTranslator:
    """Translates text from one language to another."""

    def __init__(
        self,
        generator: Seq2SeqGenerator,
        tokenizer: Tokenizer,
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
        task = "translation"

        self._converter = SequenceToTextConverter(
            generator,
            tokenizer,
            task,
            target_lang,
            skip_special_tokens=True,
        )

        pad_idx = tokenizer.vocab_info.pad_idx
        if pad_idx is None:
            raise ValueError(
                "``vocab_info` of `tokenizer` must have a PAD symbol defined."
            )

        self._pad_idx = pad_idx

        device = maybe_infer_device(generator.model)
        if device is None:
            raise ValueError(
                "All parameters and buffers of `generator.model` must be on the same device."
            )

        self._source_text_encoder = tokenizer.create_encoder(
            task="translation", lang=source_lang, mode="source", device=device
        )

        if max_source_len is not None and max_source_len <= 0:
            raise ValueError(
                f"`max_source_len` must be greater than or equal to 1, but is {max_source_len} instead."
            )

        self._max_source_len = max_source_len

    def __call__(self, source_text: str) -> tuple[str, SequenceGeneratorOutput]:
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
    ) -> tuple[list[str], SequenceGeneratorOutput]:
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

        source_seqs = [self._source_text_encoder(t) for t in source_texts]

        if self._max_source_len:
            source_seqs = [seq[: self._max_source_len] for seq in source_seqs]

        source_seqs_pt, source_seqs_layout = pad_seqs(
            source_seqs, pad_value=self._pad_idx
        )

        return self._converter.batch_convert(source_seqs_pt, source_seqs_layout)


@final
class TextCompleter:
    """Completes text prompts."""

    def __init__(
        self,
        generator: SequenceGenerator,
        tokenizer: Tokenizer,
        *,
        skip_special_tokens: bool = False,
    ) -> None:
        """
        :param generator:
            The sequence generator.
        :param tokenizer:
            The text tokenizer.
        """
        self._generator = generator

        device = maybe_infer_device(generator.model)
        if device is None:
            raise ValueError(
                "All parameters and buffers of `generator.model` must be on the same device."
            )

        self._text_encoder = tokenizer.create_encoder(mode="prompt", device=device)

        self._text_decoder = tokenizer.create_decoder(
            skip_special_tokens=skip_special_tokens
        )

    def __call__(self, prompt: str) -> tuple[str, SequenceGeneratorOutput]:
        """
        :param prompt:
            The text prompt.

        :returns:
            - The completed text.
            - The output of the underlying sequence generator.
        """
        prompt_seq = self._text_encoder(prompt)

        # (S_prm, *) -> (1, S_prm, *)
        prompt_seqs = prompt_seq.unsqueeze(0)

        prompt_seqs_layout = BatchLayout.of(prompt_seqs)

        texts, generator_output = self._do_complete(prompt_seqs, prompt_seqs_layout)

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

        prompt_seqs = [self._text_encoder(p) for p in prompts]

        prompt_seqs_pt, prompt_seqs_layout = pad_seqs(prompt_seqs)

        return self._do_complete(prompt_seqs_pt, prompt_seqs_layout)

    def _do_complete(
        self, prompt_seqs: Tensor, prompt_seqs_layout: BatchLayout
    ) -> tuple[list[str], SequenceGeneratorOutput]:
        generator_output = self._generator(prompt_seqs, prompt_seqs_layout)

        texts: list[str] = []

        for idx, hypotheses in enumerate(generator_output.hypotheses):
            if len(hypotheses) == 0:
                raise InternalError(
                    f"Sequence generator returned no hypothesis at index {idx}."
                )

            text = self._text_decoder(hypotheses[0].seq)

            texts.append(text)

        return texts, generator_output
