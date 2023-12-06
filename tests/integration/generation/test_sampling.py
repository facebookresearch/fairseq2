# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final

import torch

from fairseq2.generation import SamplingSeq2SeqGenerator, TextTranslator, TopKSampler
from fairseq2.models.nllb import load_nllb_model, load_nllb_tokenizer
from tests.common import device

ENG_SENTENCE1: Final = "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each."
ENG_SENTENCE2: Final = "How are you doing today?"
DEU_SENTENCE1: Final = "Am Montag kündigten Wissenschaftler der Stanford University School of Medicine die Erfindung eines neuen Diagnosetools an, das Zellen nach Typ sortieren kann: Ein winziger Druckschrauber, der mit Standard-Tinte-Drucker für möglicherweise etwa einen US-Cent pro Stück hergestellt werden kann."
DEU_SENTENCE2: Final = "Wie geht es Ihnen heute?"


def test_greedy_sampling() -> None:
    model_name = "nllb-200_dense_distill_600m"

    model = load_nllb_model(
        model_name, device=device, dtype=torch.float32, progress=False
    )

    tokenizer = load_nllb_tokenizer(model_name, progress=False)

    sampler = TopKSampler(k=1)

    generator = SamplingSeq2SeqGenerator(model, sampler)

    translator = TextTranslator(
        generator, tokenizer, source_lang="eng_Latn", target_lang="deu_Latn"
    )

    texts, _ = translator.batch_translate([ENG_SENTENCE1, ENG_SENTENCE2])

    assert texts == [DEU_SENTENCE1, DEU_SENTENCE2]
