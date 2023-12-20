# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import pytest
import torch
from torch import Tensor

from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import CustomAttentionMask, NaiveSDPA, TorchSDPA
from tests.common import assert_close, device


class TestScaledDotProductAttention:
    # fmt: off
    @pytest.mark.parametrize("use_key_padding_mask,use_attn_mask,training",
        [
            (False, False, True),
            (True,  True,  True),
            (False, True,  True),
            (True,  False, True),
            (False, False, False),
            (False, True,  False),
        ],
    )
    # fmt: on
    def test_torch_sdpa(
        self, use_key_padding_mask: bool, use_attn_mask: bool, training: bool
    ) -> None:
        torch_sdpa = TorchSDPA()
        naive_sdpa = NaiveSDPA()

        if training:
            torch_sdpa.eval()
            naive_sdpa.eval()

        kwargs = self._get_sdpa_args(use_key_padding_mask, use_attn_mask)

        attn1, _ = torch_sdpa(**kwargs)
        attn2, _ = naive_sdpa(**kwargs)

        assert_close(attn1, attn2)

    @staticmethod
    def _get_sdpa_args(
        use_key_padding_mask: bool, use_attn_mask: bool
    ) -> Dict[str, Any]:
        batch_size = 2

        num_heads = 4

        source_seq_len = 3
        target_seq_len = 2

        k_size = 2
        v_size = 3

        def random_tensor(*args: int) -> Tensor:
            return torch.randn(*args, device=device)

        q = random_tensor(batch_size, num_heads, target_seq_len, k_size)
        k = random_tensor(batch_size, num_heads, source_seq_len, k_size)
        v = random_tensor(batch_size, num_heads, source_seq_len, v_size)

        if use_key_padding_mask:
            key_padding_mask = PaddingMask(
                torch.tensor([2, 3], device=device), source_seq_len
            )
        else:
            key_padding_mask = None

        if use_attn_mask:
            m = random_tensor(target_seq_len, source_seq_len)

            attn_mask = CustomAttentionMask(m)
        else:
            attn_mask = None

        return {
            "seqs": q,
            "keys": k,
            "key_padding_mask": key_padding_mask,
            "values": v,
            "attn_mask": attn_mask,
        }
