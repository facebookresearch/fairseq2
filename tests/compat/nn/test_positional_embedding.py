# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from fairseq2.compat.nn import Fairseq1SinusoidalPositionalEmbedding
from tests.common import assert_close, assert_equal, has_no_inf, has_no_nan


@pytest.mark.parametrize("max_len,pad_idx", [(15, 0), (16, 0), (15, 5), (16, 10)])
def test_sinusoidal_embedding_is_backward_compatible(
    max_len: int, pad_idx: int
) -> None:
    fairseq = pytest.importorskip("fairseq")

    embed_dim = 32

    v1 = fairseq.modules.SinusoidalPositionalEmbedding(
        embedding_dim=embed_dim, padding_idx=pad_idx, init_size=max_len + pad_idx + 1
    )
    v2 = Fairseq1SinusoidalPositionalEmbedding(
        max_seq_len=max_len,
        embedding_dim=embed_dim,
        padding_token_idx=pad_idx,
        batch_first=True,
    )

    seq = torch.arange(pad_idx + 1, pad_idx + 1 + (2 * max_len)).reshape(2, -1)

    embed = torch.zeros(seq.shape + (embed_dim,))

    v1_pos = v1(seq)
    v2_pos = v2(embed)

    assert_equal(v1_pos[0], v1_pos[1])
    assert_equal(v2_pos[0], v2_pos[1])
    assert_close(v1_pos[0], v2_pos[0])
    assert_close(v1_pos, v2_pos)

    assert has_no_inf(v2_pos)
    assert has_no_nan(v2_pos)
