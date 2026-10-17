# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Module

from fairseq2.data.tokenizers import VocabularyInfo
from fairseq2.generation.sampling import SamplingSequenceGenerator, TopKSampler
from fairseq2.models.clm import CausalLM
from fairseq2.nn import BatchLayout
from tests.common import device

VOCAB_SIZE = 8
EOS_IDX = 2


class _NeverEosModel(Module):
    """Always predicts token 5, so generation only stops at the length limit."""

    max_seq_len = 1024

    def forward(self, seqs: Tensor, seqs_layout: BatchLayout, **kwargs: Any) -> Tensor:
        logits = torch.zeros(
            (seqs.size(0), seqs.size(1), VOCAB_SIZE), device=seqs.device
        )

        logits[..., 5] = 10.0

        return logits


class TestSamplingSequenceGenerator:
    def test_max_gen_len_is_respected_for_shorter_prompts_in_batch(self) -> None:
        vocab_info = VocabularyInfo(
            size=VOCAB_SIZE, unk_idx=None, bos_idx=0, eos_idx=EOS_IDX, pad_idx=1
        )

        model = cast(CausalLM, _NeverEosModel())

        generator = SamplingSequenceGenerator(
            model, vocab_info, TopKSampler(k=1), max_gen_len=10
        )

        prompt_seqs = torch.tensor(
            [[4, 1, 1, 1, 1, 1], [1, 5, 3, 6, 7, 4]], device=device
        )

        prompt_seqs_layout = BatchLayout.of(prompt_seqs, seq_lens=[1, 6])

        output = generator(prompt_seqs, prompt_seqs_layout)

        for hypotheses in output.hypotheses:
            seq = hypotheses[0].seq

            assert len(seq) == 10

            assert int(seq[-1]) == EOS_IDX
