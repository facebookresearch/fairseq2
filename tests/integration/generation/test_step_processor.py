# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import ClassVar, Final, Sequence

import pytest
import torch

from fairseq2.data.text import TextTokenizer
from fairseq2.generation import (
    BannedSequenceProcessor,
    BeamSearchSeq2SeqGenerator,
    TextTranslator,
)
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.nllb import load_nllb_model, load_nllb_tokenizer
from tests.common import device

BANNED_ENG_SENTENCES: Final = [
    "I think shaggy grandparents are the best.",
    "I think that shitty grandparents are the best you can hope for.",
]

BANNED_TEST_INPUTS: Final = [
    (
        [],
        [
            "Creo que los abuelos de mierda son los mejores.",
            "Creo que esos abuelos de mierda son lo mejor que puedes esperar.",
        ],
    ),
    (
        ["mierda"],
        [
            "Creo que los abuelos chulosos son los mejores.",
            "Creo que los abuelos porqueros son lo mejor que puedes esperar.",
        ],
    ),
    (
        ["mierda", "abuelos"],
        [
            "Creo que las abuelas maricotas son las mejores.",
            "Creo que esas abuelas porcas son lo mejor que puedes esperar.",
        ],
    ),
    (
        ["mierda", "abuelos", "lo mejor", "Creo"],
        [
            "Yo creo que las abuelas maricotas son las mejores.",
            "Pienso que esas abuelas porqueras son las mejores que puedes esperar.",
        ],
    ),
]


class TestBannedSequenceProcessor:
    model: ClassVar[EncoderDecoderModel]
    tokenizer: ClassVar[TextTokenizer]

    @classmethod
    def setup_class(cls) -> None:
        model_name = "nllb-200_dense_distill_600m"

        cls.model = load_nllb_model(
            model_name, device=device, dtype=torch.float32, progress=False
        )

        cls.tokenizer = load_nllb_tokenizer(model_name, progress=False)

    @classmethod
    def teardown_class(cls) -> None:
        del cls.model

        del cls.tokenizer

    @pytest.mark.parametrize("banned_words,expected_translations", BANNED_TEST_INPUTS)
    def test_call_works(
        self, banned_words: Sequence[str], expected_translations: Sequence[str]
    ) -> None:
        text_encoder = self.tokenizer.create_raw_encoder(device=device)

        banned_seqs = [text_encoder(b) for b in banned_words]

        generator = BeamSearchSeq2SeqGenerator(
            self.model, step_processors=[BannedSequenceProcessor(banned_seqs)]
        )

        translator = TextTranslator(
            generator,
            self.tokenizer,
            source_lang="eng_Latn",
            target_lang="spa_Latn",
        )

        texts, _ = translator.batch_translate(BANNED_ENG_SENTENCES)

        assert texts == expected_translations
