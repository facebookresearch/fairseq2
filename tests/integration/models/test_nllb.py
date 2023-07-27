# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final

from fairseq2.models.nllb import load_nllb_model, load_nllb_tokenizer
from fairseq2.text_generator import TextToTextTranslator
from tests.common import device

ENG_SENTENCE: Final = "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each."
FRA_SENTENCE: Final = "Lundi, des scientifiques de l'École de médecine de l'Université de Stanford ont annoncé l'invention d'un nouvel outil de diagnostic capable de trier les cellules par type: une minuscule puce imprimable qui peut être fabriquée à l'aide d'imprimantes à jet d'encre standard pour environ un centime de centime chacun."


def test_load_dense_distill_600m() -> None:
    model = load_nllb_model(
        "nllb-200_dense_distill_600m", device=device, progress=False
    )

    tokenizer = load_nllb_tokenizer("nllb-200_dense_distill_600m", progress=False)

    t2t_translator = TextToTextTranslator(
        model, tokenizer, source_lang="eng_Latn", target_lang="fra_Latn"
    )

    assert t2t_translator([ENG_SENTENCE]) == [FRA_SENTENCE]
