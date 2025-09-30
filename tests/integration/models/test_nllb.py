# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final

import pytest
import torch

from fairseq2.generation.beam_search import BeamSearchSeq2SeqGenerator
from fairseq2.generation.text import TextTranslator
from fairseq2.models.nllb import get_nllb_model_hub, get_nllb_tokenizer_hub
from tests.common import device

ENG_SENTENCE: Final = (
    "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each."
)
DEU_SENTENCE: Final = (
    "Am Montag kündigten Wissenschaftler der Medizinischen Fakultät der Stanford University die Erfindung eines neuen Diagnosetools an, das Zellen nach Typ sortieren kann: Ein winziger druckbarer Chip, der mit standardmäßigen Inkjet-Drucker für möglicherweise etwa einen US-Cent pro Stück hergestellt werden kann."
)


def test_load_dense_distill_600m() -> None:
    model_name = "nllb-200_dense_distill_600m"

    model = get_nllb_model_hub().load_model(
        model_name, device=device, dtype=torch.float32
    )

    tokenizer = get_nllb_tokenizer_hub().load_tokenizer(model_name)

    generator = BeamSearchSeq2SeqGenerator(
        model, tokenizer.vocab_info, echo_prompt=True, max_seq_len=128
    )

    translator = TextTranslator(
        generator, tokenizer, source_lang="eng_Latn", target_lang="deu_Latn"
    )

    text, _ = translator(ENG_SENTENCE)

    assert text == DEU_SENTENCE

    # testing that truncation prevents length-related errors
    with pytest.raises(
        ValueError, match="The lengths of all sequences in `seqs` must be less than"
    ):
        text, _ = translator(ENG_SENTENCE * 20)

    translator = TextTranslator(
        generator,
        tokenizer,
        source_lang="eng_Latn",
        target_lang="deu_Latn",
        max_source_len=1024,
    )
    text, _ = translator(ENG_SENTENCE * 20)


def test_tokenizer_special_tokens() -> None:
    model_name = "nllb-200_dense_distill_600m"

    tokenizer = get_nllb_tokenizer_hub().load_tokenizer(model_name)

    text = "Hello world!"

    # By default, the "source" mode is active.
    tokens = tokenizer.create_encoder(mode=None).encode_as_tokens(text)

    assert tokens == ["__eng_Latn__", "▁Hello", "▁world", "!", "</s>"]

    tokens = tokenizer.create_encoder(mode="source").encode_as_tokens(text)

    assert tokens == ["__eng_Latn__", "▁Hello", "▁world", "!", "</s>"]

    # "target" mode creates the decoder input tokens
    tokens = tokenizer.create_encoder(mode="target").encode_as_tokens(text)

    assert tokens == ["</s>", "__eng_Latn__", "▁Hello", "▁world", "!", "</s>"]
