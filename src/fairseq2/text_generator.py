# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Sequence, cast

import torch
from torch import Tensor

from fairseq2.data import Collater, SequenceData, StringLike
from fairseq2.data.text import TextTokenDecoder, TextTokenEncoder, TextTokenizer
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.nn.utils.module import infer_device
from fairseq2.sequence_generator import Seq2SeqGenerator, SequenceOptions


class SequenceToTextGeneratorBase:
    """Represents abstract base class for sequence-to-text generators."""

    model: EncoderDecoderModel
    token_decoder: TextTokenDecoder
    prefix_seq: Tensor
    seq_generator: Seq2SeqGenerator

    def __init__(
        self,
        model: EncoderDecoderModel,
        tokenizer: TextTokenizer,
        target_lang: Optional[str] = None,
        seq_opts: Optional[SequenceOptions] = None,
    ) -> None:
        """
        :param model:
            The encoder-decoder model to use for generation.
        :param tokenizer:
            The text tokenizer to use.
        :param target_lang:
            The target language.
        :param seq_opts:
            The options to pass to the underlying :class:`Seq2SeqGenerator`.
        """
        model.eval()

        self.model = model

        self.token_decoder = tokenizer.create_decoder()

        device = infer_device(model)

        target_token_encoder = tokenizer.create_encoder(
            task="translation", lang=target_lang, mode="target", device=device
        )

        # Most tokenizers typically use one or more control symbols to indicate
        # the beginning of a sentence.
        self.prefix_seq = target_token_encoder("")

        self.seq_generator = Seq2SeqGenerator(
            self.model, tokenizer.vocabulary_info, seq_opts
        )

    def _do_generate(
        self, source_seqs: Tensor, source_seq_lens: Optional[Tensor]
    ) -> List[StringLike]:
        """A subclass should call this function for the actual text generation."""
        encoder_output, encoder_padding_mask = self.model.encode(
            source_seqs, source_seq_lens
        )

        target_seqs, _ = self.seq_generator(
            self.prefix_seq, encoder_output, encoder_padding_mask, source_seqs.size(1)
        )

        return self.token_decoder(target_seqs)


class SequenceToTextGenerator(SequenceToTextGeneratorBase):
    """Generates text output from input sequences.

    The interpretation of input sequences depends on the underlying encoder-
    decoder model.
    """

    @torch.inference_mode()
    def __call__(
        self, source_seqs: Tensor, source_seq_lens: Optional[Tensor]
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
        return self._do_generate(source_seqs, source_seq_lens)


class TextToTextTranslator(SequenceToTextGeneratorBase):
    """Translates text from one language to another."""

    source_token_encoder: TextTokenEncoder
    collater: Collater

    def __init__(
        self,
        model: EncoderDecoderModel,
        tokenizer: TextTokenizer,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        seq_opts: Optional[SequenceOptions] = None,
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
        :param seq_opts:
            The options to pass to the underlying :class:`Seq2SeqGenerator`.
        """
        super().__init__(model, tokenizer, target_lang, seq_opts)

        device = infer_device(model)

        self.source_token_encoder = tokenizer.create_encoder(
            task="translation", lang=source_lang, mode="source", device=device
        )

        self.collater = Collater(pad_idx=tokenizer.vocabulary_info.pad_idx)

    @torch.inference_mode()
    def __call__(self, source_sentences: Sequence[StringLike]) -> List[StringLike]:
        """
        :param source_sentences:
            The sentences in the source language.

        :returns:
            The translated sentences in the target language.
        """
        indices: List[Tensor] = []

        # TODO: use parallel_invoke op
        for source_sentence in source_sentences:
            indices.append(self.source_token_encoder(source_sentence))

        batch = cast(SequenceData, self.collater(indices))

        return self._do_generate(
            batch["seqs"], batch["seq_lens"] if batch["is_ragged"] else None
        )
