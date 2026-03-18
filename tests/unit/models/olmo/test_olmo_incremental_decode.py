# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Incremental decoding test for OLMO2-1B using a pretrained checkpoint."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from fairseq2.data.tokenizers import load_tokenizer
from fairseq2.models.hub import ModelHub
from fairseq2.models.olmo import get_olmo_model_hub
from fairseq2.models.olmo.config import OLMOConfig
from fairseq2.models.transformer_lm import TransformerLM
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.nn.utils.padding import pad_seqs
from tests.common import assert_close, device

OLMO2_1B_NAME = "olmo-2-0425-1b"

TEST_SENTENCE = (
    "The capital of Germany is Berlin, which has a rich history dating back centuries."
)


@pytest.fixture(scope="module")
def hub() -> ModelHub[TransformerLM, OLMOConfig]:
    return get_olmo_model_hub()


def test_olmo2_incremental_decode(hub: Any) -> None:
    """Full-sequence forward must match step-by-step incremental decode (OLMO2-1B).

    Loads the pretrained OLMO2-1B checkpoint and tokenizes real sentences to
    verify that KV-cached incremental decoding produces identical logits to a
    single full-sequence pass.
    """
    model = hub.load_model(OLMO2_1B_NAME, device=device, dtype=torch.float32)
    model.eval()

    tokenizer = load_tokenizer(OLMO2_1B_NAME)
    encoder = tokenizer.create_encoder(mode="prompt", device=device)

    # Use the same sentence for both batch elements (avoids padding divergence).
    token_indices = [encoder(TEST_SENTENCE), encoder(TEST_SENTENCE)]

    pad_idx = tokenizer.vocab_info.pad_idx
    assert pad_idx is not None

    seqs, seqs_layout = pad_seqs(token_indices, pad_value=pad_idx)

    with torch.no_grad():
        expected_logits = model(seqs, seqs_layout)

    state_bag = IncrementalStateBag(max_num_steps=model.max_seq_len)

    with torch.no_grad():
        for idx in range(seqs.size(1)):
            step_seqs = seqs[:, idx : idx + 1]
            step_layout = BatchLayout.of(step_seqs)

            step_logits = model(step_seqs, step_layout, state_bag=state_bag)
            state_bag.increment_step_nr()

            assert_close(step_logits, expected_logits[:, idx : idx + 1], atol=2e-4)

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
