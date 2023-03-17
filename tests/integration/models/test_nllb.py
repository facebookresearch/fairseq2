# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.generate import BeamSearchStrategy
from fairseq2.models.nllb import load_nllb_model
from tests.common import device

ENG = "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each."
FRA = "Lundi, des scientifiques de l'École de médecine de l'Université de Stanford ont annoncé l'invention d'un nouvel outil de diagnostic capable de trier les cellules par type: une minuscule puce imprimable qui peut être fabriquée à l'aide d'imprimantes à jet d'encre standard pour environ un centime de centime chacun."


def test_load_dense_distill_600m() -> None:
    model, tokenizer = load_nllb_model(
        "dense_distill_600m", device=device, progress=False
    )

    model.eval()

    strategy = BeamSearchStrategy(
        vocab_info=tokenizer.vocab_info, beam_size=1, max_len=256
    )

    fra = strategy.generate_str_ex(
        model,
        tokenizer,
        ENG,
        src_lang="eng_Latn",
        tgt_lang="fra_Latn",
        device=device,
    )

    assert fra == FRA
