# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Final

import torch

from fairseq2.generate import BeamSearchStrategy
from fairseq2.generate.search import _stretch_to_beams
from fairseq2.models.s2t_transformer import (
    S2TTransformerTokenizer,
    load_s2t_transformer_model,
)
from fairseq2.models.transformer import TransformerModel
from fairseq2.nn import IncrementalStateBag
from tests.common import device

TEST_FBANK_PATH: Final = Path(__file__).parent.joinpath("fbank.pt")

TRANSFORMER_DE: Final = "<lang:de> Es war Zeit des Abendessens, und wir suchten nach einem Ort, wo man essen kann."

CONFORMER_DE: Final = (
    "Es war das Essenszeit-Abendessen und wir begannen, nach dem Abendessen zu suchen."
)


def test_load_s2t_transformer_mustc_st_jt_m() -> None:
    model, tokenizer = load_s2t_transformer_model(
        "s2t_transformer_mustc_st_jt_m", device=device, progress=False
    )

    assert_translation(model, tokenizer, expected=TRANSFORMER_DE)


def test_load_s2t_conformer_covost_st_en_de() -> None:
    model, tokenizer = load_s2t_transformer_model(
        "s2t_conformer_covost_st_en_de", device=device, progress=False
    )

    assert_translation(model, tokenizer, expected=CONFORMER_DE)


def assert_translation(
    model: TransformerModel, tokenizer: S2TTransformerTokenizer, expected: str
) -> None:
    model.eval()

    # TODO: The strategy API needs to be revised to be generic. As of today, it
    # is pretty much limited to `TransformerModel`.
    strategy = BeamSearchStrategy(
        vocab_info=tokenizer.vocab_info, beam_size=1, max_len=256
    )

    encoder = tokenizer.create_encoder(lang="de", device=device)
    decoder = tokenizer.create_decoder()

    # TODO(bug): BeamSearchStrategy does not support unbatched input.
    fbanks = torch.load(TEST_FBANK_PATH).to(device).unsqueeze(0)
    num_frames = torch.tensor(fbanks.size(1)).unsqueeze(0)

    encoder_out, encoder_padding_mask = model.encode(fbanks, num_frames)

    # TODO: This is a manual, boilerplate code to run beam search with S2T
    # Transformer. It has to be reduced to a single line after revising the
    # strategy API.
    job = strategy.new_search_job(fbanks, prefix_tokens=encoder([""]))

    state_bag = IncrementalStateBag()

    # `prefix_tokens` has already </s> and <lang:de> tokens.
    state_bag.increment_step(2)

    encoder_out = _stretch_to_beams(encoder_out, beam_size=1)
    if encoder_padding_mask is not None:
        encoder_padding_mask = _stretch_to_beams(encoder_padding_mask, beam_size=1)

    while not job.done:
        query_tokens = job.next_query()

        logits = model.decode_and_project(
            query_tokens, None, encoder_out, encoder_padding_mask, state_bag
        )

        logits = logits.squeeze(1)

        state_bag.increment_step()

        job.update(logits)

    tokens = job.finalize(top=0).tokens

    tokens = tokens.view(-1, tokens.shape[-1])

    de = decoder(tokens)

    assert de == [expected]
