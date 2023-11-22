# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final

import torch

from fairseq2.models.nllb import load_nllb_model, load_nllb_tokenizer
from fairseq2.nn import IncrementalStateBag
from fairseq2.nn.padding import pad_seqs
from tests.common import assert_close, device

# fmt: off
DE_SENTENCE: Final = "Löwenrudel agieren ähnlich wie Wolfs- oder Hunderudel, Tiere, die den Löwen (aber nicht anderen Großkatzen) im Verhalten überraschend ähneln und ebenso tödlich gegenüber ihrer Beute sind."
EN_SENTENCE: Final = "Lion prides act much like packs of wolves or dogs, animals surprisingly similar to lions (but not other big cats) in behavior, and also very deadly to their prey."
# fmt: on


def test_incremental_decoding_works() -> None:
    model_name = "nllb-200_dense_distill_600m"

    model = load_nllb_model(
        model_name, device=device, dtype=torch.float32, progress=False
    )

    model.eval()

    tokenizer = load_nllb_tokenizer(model_name, progress=False)

    pad_idx = tokenizer.vocab_info.pad_idx

    assert pad_idx is not None

    # Set up encoder and decoder inputs.
    source_token_encoder = tokenizer.create_encoder(
        task="translation", lang="deu_Latn", mode="source", device=device
    )
    target_token_encoder = tokenizer.create_encoder(
        task="translation", lang="eng_Latn", mode="target", device=device
    )

    source_indices = [
        source_token_encoder(DE_SENTENCE),
        source_token_encoder(DE_SENTENCE),
    ]

    target_indices = [
        target_token_encoder(EN_SENTENCE),
        target_token_encoder(EN_SENTENCE),
    ]

    source_seqs, source_padding_mask = pad_seqs(source_indices, pad_idx)
    target_seqs, target_padding_mask = pad_seqs(target_indices, pad_idx)

    # Generate the expected decoder output.
    encoder_output, encoder_padding_mask = model.encode(
        source_seqs, source_padding_mask
    )

    decoder_output, decoder_padding_mask = model.decode(
        target_seqs, target_padding_mask, encoder_output, encoder_padding_mask
    )

    assert decoder_padding_mask is None

    # Now try to match the decoder output with incremental decoding.
    state_bag = IncrementalStateBag(max_num_steps=256)

    incremental_output = torch.empty(
        (2, 0, model.model_dim), device=device, dtype=torch.float32
    )

    for idx in range(target_seqs.size(1)):
        pos_output, pos_padding_mask = model.decode(
            target_seqs[:, idx : idx + 1],
            None,
            encoder_output,
            encoder_padding_mask,
            state_bag=state_bag,
        )

        assert pos_padding_mask is None

        state_bag.increment_step_nr()

        incremental_output = torch.cat([incremental_output, pos_output], dim=1)

    assert_close(decoder_output, incremental_output)
