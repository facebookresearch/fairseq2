# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Final

import torch

from fairseq2.generation import SequenceToTextGenerator
from fairseq2.models.s2t_transformer import (
    S2TTransformerTokenizer,
    load_s2t_transformer_model,
    load_s2t_transformer_tokenizer,
)
from fairseq2.models.transformer import TransformerModel
from tests.common import device

TEST_FBANK_PATH: Final = Path(__file__).parent.joinpath("fbank.pt")

# fmt: off
TRANSFORMER_DE:       Final = "<lang:de> Es war Zeit des Abendessens, und wir suchten nach einem Ort, an dem wir essen konnten."
CONFORMER_DE:         Final = "Es war das Abendessen, und wir begannen, nach dem Essen zu suchen."
CONFORMER_REL_POS_DE: Final = "Es war Essenszeit, und wir beginnen nach Ort zu suchen."
# fmt: on


def test_load_s2t_transformer_mustc_st_jt_m() -> None:
    model = load_s2t_transformer_model(
        "s2t_transformer_mustc_st_jt_m", device=device, progress=False
    )

    tokenizer = load_s2t_transformer_tokenizer(
        "s2t_transformer_mustc_st_jt_m", progress=False
    )

    assert_translation(model, tokenizer, expected=TRANSFORMER_DE)


def test_load_s2t_conformer_covost_st_en_de() -> None:
    model = load_s2t_transformer_model(
        "s2t_conformer_covost_st_en_de", device=device, progress=False
    )

    tokenizer = load_s2t_transformer_tokenizer(
        "s2t_conformer_covost_st_en_de", progress=False
    )

    assert_translation(model, tokenizer, expected=CONFORMER_DE)


def test_load_s2t_conformer_rel_pos_covost_st_en_de() -> None:
    model = load_s2t_transformer_model(
        "s2t_conformer_rel_pos_covost_st_en_de", device=device, progress=False
    )

    tokenizer = load_s2t_transformer_tokenizer(
        "s2t_conformer_rel_pos_covost_st_en_de", progress=False
    )

    assert_translation(model, tokenizer, expected=CONFORMER_REL_POS_DE)


def assert_translation(
    model: TransformerModel, tokenizer: S2TTransformerTokenizer, expected: str
) -> None:
    fbanks = torch.load(TEST_FBANK_PATH).unsqueeze(0).to(device)

    generator = SequenceToTextGenerator(model, tokenizer, target_lang="de")

    assert generator(fbanks, None) == [expected]
