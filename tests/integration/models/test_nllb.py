# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final

from fairseq2.generation import TextTranslator
from fairseq2.models.nllb import load_nllb_model, load_nllb_tokenizer
from tests.common import device

ENG_SENTENCE: Final = "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each."
DEU_SENTENCE: Final = "Am Montag kündigten Wissenschaftler der Medizinischen Fakultät der Universität Stanford die Erfindung eines neuen Diagnosetools an, das Zellen nach Typ sortieren kann: Ein winziger druckbarer Chip, der mit standardmäßigen Inkjet-Drucker für möglicherweise etwa einen US-Cent pro Stück hergestellt werden kann."


def test_load_dense_distill_600m() -> None:
    model = load_nllb_model(
        "nllb-200_dense_distill_600m", device=device, progress=False
    )

    tokenizer = load_nllb_tokenizer("nllb-200_dense_distill_600m", progress=False)

    translator = TextTranslator(
        model, tokenizer, source_lang="eng_Latn", target_lang="deu_Latn"
    )

    assert translator([ENG_SENTENCE]) == [DEU_SENTENCE]
