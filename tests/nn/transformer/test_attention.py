from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from fairseq2.nn.transformer.attention import scaled_dot_product_attention
from tests import tensor_matchers as tm
from tests.common import TestCase
from tests.utils import tmp_rng_seed


class TestScaledDotProductAttention(TestCase):
    def build_qkvm(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        N = 2
        T = 2
        V = 3
        K = 2
        S = 3

        # (N: batch size, T: target sequence length, K: key size)
        queries = torch.rand((N, T, K))

        # (N: batch size, S: source sequence length, K: key size)
        keys = torch.rand((N, S, K))

        # (N: batch size, S: source sequence length, V: value size)
        values = torch.rand((N, S, V))

        # (T: target sequence length, S: source sequence length)
        mask = torch.rand((T, S))

        return (
            queries,
            keys,
            values,
            mask,
        )

    def test_nomask_nodropout(self) -> None:
        queries, keys, values, _mask = self.build_qkvm()

        attn, attn_weights = scaled_dot_product_attention(
            queries=queries,
            keys=keys,
            values=values,
        )

        expected_attn_weights = F.softmax(
            torch.bmm(
                queries * (queries.size(-1) ** -0.5),
                keys.transpose(1, 2),
            ),
            dim=-1,
        )

        expected_attn = torch.bmm(expected_attn_weights, values)

        tm.assert_tensor_equals(
            attn_weights,
            expected_attn_weights,
            close=True,
        )

        tm.assert_tensor_equals(
            attn,
            expected_attn,
            close=True,
        )

    def test_mask_nodropout(self) -> None:
        queries, keys, values, mask = self.build_qkvm()

        attn, attn_weights = scaled_dot_product_attention(
            queries=queries,
            keys=keys,
            values=values,
            mask=mask,
        )

        expected_attn_weights = F.softmax(
            torch.baddbmm(
                mask,
                queries * (queries.size(-1) ** -0.5),
                keys.transpose(1, 2),
            ),
            dim=-1,
        )

        expected_attn = torch.bmm(expected_attn_weights, values)

        tm.assert_tensor_equals(
            attn_weights,
            expected_attn_weights,
            close=True,
        )

        tm.assert_tensor_equals(
            attn,
            expected_attn,
            close=True,
        )

    def test_nomask_dropout_training(self) -> None:
        queries, keys, values, _mask = self.build_qkvm()

        dropout_p = 0.5

        with tmp_rng_seed(self.device):
            attn, attn_weights = scaled_dot_product_attention(
                queries=queries,
                keys=keys,
                values=values,
                dropout_p=dropout_p,
            )

        with tmp_rng_seed(self.device):
            expected_attn_weights = F.dropout(
                F.softmax(
                    torch.bmm(
                        queries * (queries.size(-1) ** -0.5),
                        keys.transpose(1, 2),
                    ),
                    dim=-1,
                ),
                p=dropout_p,
                training=True,
            )

        expected_attn = torch.bmm(expected_attn_weights, values)

        tm.assert_tensor_equals(
            attn_weights,
            expected_attn_weights,
            close=True,
        )

        tm.assert_tensor_equals(
            attn,
            expected_attn,
            close=True,
        )

    def test_nomask_dropout_notraining(self) -> None:
        queries, keys, values, _mask = self.build_qkvm()

        dropout_p = 0.5

        attn, attn_weights = scaled_dot_product_attention(
            queries=queries,
            keys=keys,
            values=values,
            dropout_p=dropout_p,
            training=False,
        )

        expected_attn_weights = F.softmax(
            torch.bmm(
                queries * (queries.size(-1) ** -0.5),
                keys.transpose(1, 2),
            ),
            dim=-1,
        )

        expected_attn = torch.bmm(expected_attn_weights, values)

        tm.assert_tensor_equals(
            attn_weights,
            expected_attn_weights,
            close=True,
        )

        tm.assert_tensor_equals(
            attn,
            expected_attn,
            close=True,
        )
