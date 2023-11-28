# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq2.generation import NGramRepeatBlockProcessor
from tests.common import assert_close, device


class TestNGramRepeatBlockProcessor:
    def test_call_works(self) -> None:
        seq1 = torch.tensor([1, 2, 3, 1, 2, 0, 1, 2], device=device)
        seq2 = torch.tensor([1, 2, 3, 4, 3, 2, 1, 0], device=device)
        seq3 = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], device=device)

        seqs = torch.stack([seq1, seq2, seq3])

        probs1 = torch.tensor([0.1, 0.1, 0.1, 0.1], device=device)
        probs2 = torch.tensor([0.1, 0.1, 0.1, 0.1], device=device)
        probs3 = torch.tensor([0.1, 0.1, 0.1, 0.1], device=device)

        probs = torch.stack([probs1, probs2, probs3])

        processor = NGramRepeatBlockProcessor(ngram_size=3)

        processor(seqs, probs)

        assert_close(probs[0], [0.0, 0.1, 0.1, 0.0])
        assert_close(probs[1], [0.1, 0.1, 0.1, 0.1])
        assert_close(probs[2], [0.0, 0.0, 0.1, 0.1])

    def test_call_works_when_ngram_size_is_1(self) -> None:
        seq1 = torch.tensor([1, 3, 0], device=device)
        seq2 = torch.tensor([2, 1, 1], device=device)

        seqs = torch.stack([seq1, seq2])

        probs1 = torch.tensor([0.1, 0.1, 0.1, 0.1], device=device)
        probs2 = torch.tensor([0.1, 0.1, 0.1, 0.1], device=device)

        probs = torch.stack([probs1, probs2])

        processor = NGramRepeatBlockProcessor(ngram_size=1)

        processor(seqs, probs)

        assert_close(probs[0], [0.0, 0.0, 0.1, 0.0])
        assert_close(probs[1], [0.1, 0.0, 0.0, 0.1])

    def test_call_works_when_seq_len_is_less_than_ngram_size(self) -> None:
        seq1 = torch.tensor([3, 2], device=device)
        seq2 = torch.tensor([1, 0], device=device)

        seqs = torch.stack([seq1, seq2])

        probs1 = torch.tensor([0.1, 0.1, 0.1, 0.1], device=device)
        probs2 = torch.tensor([0.1, 0.1, 0.1, 0.1], device=device)

        probs = torch.stack([probs1, probs2])

        processor = NGramRepeatBlockProcessor(ngram_size=3)

        processor(seqs, probs)

        assert_close(probs[0], [0.1, 0.1, 0.1, 0.1])
        assert_close(probs[1], [0.1, 0.1, 0.1, 0.1])
