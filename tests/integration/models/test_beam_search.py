# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# from pathlib import Path
# from typing import Final, Optional, Sequence
#
# import pytest
# import torch
#
# from fairseq2.data.text.text_tokenizer import TextTokenizer
# from fairseq2.data.typing import StringLike
# from fairseq2.generation import (
#    BannedSequenceLogitsProcessor,
#    SequenceGeneratorOptions,
#    SequenceToTextGenerator,
#    TextTranslator,
# )
# from fairseq2.models.nllb import load_nllb_model, load_nllb_tokenizer
# from fairseq2.models.unity import load_unity_model, load_unity_text_tokenizer
# from tests.common import device
#
## Types
# BatchTextData = Sequence[StringLike]
# BannedSequences = Optional[Sequence[StringLike]]
#
## NLLB
# NLLB_SRC_LANG: Final = "eng_Latn"
# NLLB_TGT_LANG: Final = "spa_Latn"
# NLLB_MODEL: Final = load_nllb_model(
#    "nllb-200_dense_distill_600m", device=device, progress=False
# ).eval()
# NLLB_TOKENIZER: Final = load_nllb_tokenizer(
#    "nllb-200_dense_distill_600m", progress=False
# )
# NLLB_SRC_DATA: Final = [
#    "I think shaggy grandparents are the best.",
#    "I think that shitty grandparents are the best you can hope for.",
# ]
# NLLB_TEST_DATA: Final = [
#    (
#        None,
#        [
#            "Creo que los abuelos de mierda son los mejores.",
#            "Creo que esos abuelos de mierda son lo mejor que puedes esperar.",
#        ],
#    ),
#    (
#        ["mierda"],
#        [
#            "Creo que los abuelos chulosos son los mejores.",
#            "Creo que los abuelos porqueros son lo mejor que puedes esperar.",
#        ],
#    ),
#    (
#        ["mierda", "abuelos"],
#        [
#            "Creo que las abuelas maricotas son las mejores.",
#            "Creo que esas abuelas porcas son lo mejor que puedes esperar.",
#        ],
#    ),
#    (
#        ["mierda", "abuelos", "lo mejor", "Creo"],
#        [
#            "Yo creo que las abuelas maricotas son las mejores.",
#            "Pienso que esas abuelas porqueras son las mejores que puedes esperar.",
#        ],
#    ),
# ]
#
#
# @pytest.mark.parametrize("banned_sequences,expected", NLLB_TEST_DATA)
# def test_nllb(banned_sequences: BannedSequences, expected: BatchTextData) -> None:
#    options = _sequence_generator_options(
#        banned_sequences, NLLB_TOKENIZER, NLLB_TGT_LANG
#    )
#    translator = TextTranslator(
#        NLLB_MODEL,
#        NLLB_TOKENIZER,
#        source_lang=NLLB_SRC_LANG,
#        target_lang=NLLB_TGT_LANG,
#        opts=options,
#    )
#
#    output = translator(NLLB_SRC_DATA)
#    assert output == expected
#
#
## UnitY
# UNITY_MODEL: Final = load_unity_model(
#    "multitask_unity", device=device, dtype=torch.float32
# ).eval()
# UNITY_TOKENIZER = load_unity_text_tokenizer("multitask_unity")
#
# FBANK_PATH: Final = Path(__file__).parent.joinpath("fbank.pt")
# FBANKS: Final = torch.load(FBANK_PATH).to(device).unsqueeze(0)
#
# UNITY_TGT_LANG: Final = "fra"
#
# UNITY_TEST_DATA: Final = [
#    (
#        None,
#        [
#            "C'était l'heure du dîner, et nous avons commencé à chercher un endroit pour manger."
#        ],
#    ),
#    (
#        ["endroit"],
#        ["C'était l'heure du dîner, et nous avons commencé à chercher à manger."],
#    ),
#    (
#        ["endroit", "nous avons"],
#        ["C'était l'heure du dîner, et on a commencé à chercher à manger."],
#    ),
# ]
#
#
# @pytest.mark.parametrize("banned_sequences,expected", UNITY_TEST_DATA)
# def test_unity(banned_sequences: BannedSequences, expected: BatchTextData) -> None:
#    options = _sequence_generator_options(
#        banned_sequences, UNITY_TOKENIZER, UNITY_TGT_LANG
#    )
#    generator = SequenceToTextGenerator(
#        UNITY_MODEL, UNITY_TOKENIZER, target_lang=UNITY_TGT_LANG, opts=options
#    )
#
#    output = generator(FBANKS, None)
#
#    assert output == expected
#
#
# def _sequence_generator_options(
#    banned_sequences: BannedSequences, tokenizer: TextTokenizer, tgt_lang: str
# ) -> Optional[SequenceGeneratorOptions]:
#    if banned_sequences is None or len(banned_sequences) == 0:
#        return None
#
#    token_encoder = tokenizer.create_encoder(
#        "translation", lang=tgt_lang, mode="target", device=device
#    )
#
#    banned_tokens = BannedSequenceLogitsProcessor.compute_banned_words_seqs(
#        banned_strings=banned_sequences, token_encoder=token_encoder
#    )
#    pad_idx = tokenizer.vocab_info.pad_idx or 0
#
#    return SequenceGeneratorOptions(
#        logits_processor=BannedSequenceLogitsProcessor(banned_tokens, pad_idx, device)
#    )
