# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final

import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq2.models.nllb import load_nllb_model, load_nllb_tokenizer
from fairseq2.nn import IncrementalStateBag
from tests.common import assert_close, device

# fmt: off
DE_SENTENCE1: Final = "Löwenrudel agieren ähnlich wie Wolfs- oder Hunderudel, Tiere, die den Löwen (aber nicht anderen Großkatzen) im Verhalten überraschend ähneln und ebenso tödlich gegenüber ihrer Beute sind."
EN_SENTENCE1: Final = "Lion prides act much like packs of wolves or dogs, animals surprisingly similar to lions (but not other big cats) in behavior, and also very deadly to their prey."

DE_SENTENCE2: Final = "Leider fällt das Studium von Verkehrsflüssen schwer, da man Fahrerverhalten nicht mit hundertprozentiger Sicherheit voraussagen kann."
EN_SENTENCE2: Final = "Unfortunately, studying traffic flow is difficult because driver behavior cannot be predicted with one-hundred percent certainty."

DE_SENTENCE3: Final = "Es gibt Familienstrände, die manchmal überlaufen sind, mit einer schönen Einkaufspromenade entlang der Küste. Das Schwimmen ist hier sicher."
EN_SENTENCE3: Final = "These are sometimes-crowded family beaches with a good range of shops lining the shore. Swimming is safe."
# fmt: on


def test_incremental_decoding_works() -> None:
    # fmt: off
    model = load_nllb_model(
        "nllb-200_dense_distill_600m", device=device, dtype=torch.float32, progress=False
    )

    model.eval()

    tokenizer = load_nllb_tokenizer("nllb-200_dense_distill_600m", progress=False)

    pad_idx = tokenizer.vocab_info.pad_idx

    # Set up encoder and decoder inputs.
    source_token_encoder = tokenizer.create_encoder(
        task="translation", lang="deu_Latn", mode="source", device=device
    )
    target_token_encoder = tokenizer.create_encoder(
        task="translation", lang="eng_Latn", mode="target", device=device
    )

    source_indices = [
        source_token_encoder(DE_SENTENCE1),
        source_token_encoder(DE_SENTENCE2),
        source_token_encoder(DE_SENTENCE3),
    ]

    target_indices = [
        target_token_encoder(EN_SENTENCE1),
        target_token_encoder(EN_SENTENCE2),
        target_token_encoder(EN_SENTENCE3),
    ]

    source_seqs = pad_sequence(source_indices, batch_first=True, padding_value=pad_idx)  # type: ignore[arg-type]
    target_seqs = pad_sequence(target_indices, batch_first=True, padding_value=pad_idx)  # type: ignore[arg-type]

    source_seq_lens = torch.tensor(
        [s.numel() for s in source_indices], device=device, dtype=torch.int
    )
    target_seq_lens = torch.tensor(
        [t.numel() for t in target_indices], device=device, dtype=torch.int
    )

    target_seq_len_mask = torch.zeros_like(target_seqs)

    target_seq_len_mask[0, : target_seq_lens[0]] = torch.ones((target_seq_lens[0].item(),), device=device)  # type: ignore[arg-type]
    target_seq_len_mask[1, : target_seq_lens[1]] = torch.ones((target_seq_lens[1].item(),), device=device)  # type: ignore[arg-type]
    target_seq_len_mask[2, : target_seq_lens[2]] = torch.ones((target_seq_lens[2].item(),), device=device)  # type: ignore[arg-type]

    # Generate the expected decoder output.
    encoder_output, encoder_padding_mask = model.encode(source_seqs, source_seq_lens)

    decoder_output, decoder_padding_mask = model.decode(
        target_seqs, target_seq_lens, encoder_output, encoder_padding_mask
    )

    assert decoder_padding_mask is not None

    # Now try to match the decoder output with incremental decoding.
    state_bag = IncrementalStateBag()

    incremental_output       = torch.empty((3, 0, model.model_dim), device=device, dtype=torch.float32)
    incremental_padding_mask = torch.empty((3, 0),                  device=device, dtype=torch.float32)

    for idx in range(target_seqs.size(1)):
        pos_output, pos_padding_mask = model.decode(
            target_seqs[:, idx:idx+1],
            target_seq_len_mask[:, idx],
            encoder_output,
            encoder_padding_mask,
            state_bag,
        )

        state_bag.increment_step()

        # The returned padding mask will be `None` if all sequences have the
        # same length (i.e. 1).
        if pos_padding_mask is None:
            pos_padding_mask = torch.zeros((3, 1), device=device, dtype=torch.float32)

        incremental_output       = torch.cat([incremental_output,       pos_output],       dim=1)
        incremental_padding_mask = torch.cat([incremental_padding_mask, pos_padding_mask], dim=1)

    assert_close(decoder_output,       incremental_output)
    assert_close(decoder_padding_mask, incremental_padding_mask)
    # fmt: on
