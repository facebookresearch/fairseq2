# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from fairseq2.compat.nn import FairseqSinusoidalPositionalEmbedding
from tests.common import assert_close, assert_equal, has_no_inf, has_no_nan


def test_sinusoidal_embedding_is_backward_compatible() -> None:
    _assert_embedding_is_backward_compatible(16, 0, 32)
    _assert_embedding_is_backward_compatible(15, 0, 32)
    _assert_embedding_is_backward_compatible(16, 10, 32)
    _assert_embedding_is_backward_compatible(15, 5, 32)


def _assert_embedding_is_backward_compatible(
    max_len: int,
    pad_idx: int,
    emb_dim: int,
) -> None:
    fairseq = pytest.importorskip("fairseq")

    a = fairseq.modules.SinusoidalPositionalEmbedding(
        embedding_dim=emb_dim, padding_idx=pad_idx, init_size=max_len + pad_idx + 1
    )
    b = FairseqSinusoidalPositionalEmbedding(
        max_seq_len=max_len,
        padding_token_idx=pad_idx,
        embedding_dim=emb_dim,
        batch_first=True,
    )

    n = max_len
    seq = torch.arange(pad_idx + 1, pad_idx + 1 + 2 * n).reshape(2, -1)
    seq[:, -3:] = pad_idx
    a_pos = a(seq)
    b_pos = b(torch.zeros(1), seq)
    assert_equal(a_pos[0], a_pos[1])
    assert_equal(b_pos[0], b_pos[1])
    assert_close(a_pos[0], b_pos[0])
    assert_close(a_pos, b_pos)
    assert has_no_inf(b_pos)
    assert has_no_nan(b_pos)
