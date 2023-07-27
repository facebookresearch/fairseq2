# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence

import torch
from torch import Tensor

from fairseq2.data import StringLike, VocabularyInfo
from fairseq2.data.text import TextTokenDecoder, TextTokenEncoder, TextTokenizer
from fairseq2.models.transformer import TransformerModel
from fairseq2.nn import IncrementalStateBag
from fairseq2.sequence_generator import (
    BeamSearchStrategy,
    SearchStrategy,
    _stretch_to_beams,
)
from fairseq2.typing import Device


# TODO: This is a temporary class for demo purposes and will be removed.
class Translator:
    model: TransformerModel
    token_encoder: TextTokenEncoder
    token_decoder: TextTokenDecoder
    vocabulary_info: VocabularyInfo
    strategy: SearchStrategy
    device: Device

    def __init__(
        self,
        model: TransformerModel,
        tokenizer: TextTokenizer,
        target_lang: str,
        device: Device,
    ) -> None:
        model.eval()

        self.model = model

        self.token_encoder = tokenizer.create_encoder(
            task="translation", lang=target_lang, mode="target", device=device
        )
        self.token_decoder = tokenizer.create_decoder()

        self.vocabulary_info = tokenizer.vocabulary_info

        self.strategy = BeamSearchStrategy(
            vocab_info=self.vocabulary_info, beam_size=1, max_len=256
        )

    def translate(self, fbank: Tensor) -> StringLike:
        t = self.translate_batch(fbank.unsqueeze(0))

        return t[0]

    def translate_batch(
        self, fbanks: Tensor, num_frames: Optional[Tensor] = None
    ) -> Sequence[StringLike]:
        encoder_out, encoder_padding_mask = self.model.encode(fbanks, num_frames)

        encoder_out = _stretch_to_beams(encoder_out, beam_size=1)
        if encoder_padding_mask is not None:
            encoder_padding_mask = _stretch_to_beams(encoder_padding_mask, beam_size=1)

        # TODO: This is a manual, boilerplate code to run beam search with S2T
        # Transformer. It has to be reduced to a single line after revising the
        # strategy API.
        job = self.strategy.new_search_job(
            fbanks, prefix_tokens=self.token_encoder("").unsqueeze(0)
        )

        state_bag = IncrementalStateBag()

        # `prefix_tokens` has already </s> and lang tokens.
        state_bag.increment_step(2)

        while not job.done:
            query_tokens = job.next_query()

            if self.vocabulary_info.pad_idx is None:
                seq_lens = None
            else:
                padding_mask = query_tokens.ne(self.vocabulary_info.pad_idx)
                seq_lens = torch.count_nonzero(padding_mask, dim=-1)

            decoder_out, decoder_padding_mask = self.model.decode(
                query_tokens, seq_lens, encoder_out, encoder_padding_mask, state_bag
            )
            model_out = self.model.project(decoder_out, decoder_padding_mask)
            logits = model_out.logits.squeeze(1)

            state_bag.increment_step()

            job.update(logits)

        tokens = job.finalize(top=0).tokens
        tokens = tokens.view(-1, tokens.shape[-1])

        return self.token_decoder(tokens)
