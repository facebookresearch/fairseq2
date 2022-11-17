# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor

from fairseq2.nn import LearnedPositionalEmbedding, SinusoidalPositionalEmbedding
from tests.common import TestCase, tmp_rng_seed


class TestSinusoidalPositionalEmbedding(TestCase):
    def _make_expected_weight(self) -> Tensor:
        # fmt: off
        return torch.tensor([
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  # noqa
              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  # noqa
              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  # noqa
              0.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  # noqa
              1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  # noqa
              1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  # noqa
              1.0000e+00,  1.0000e+00],                                        # noqa
            [ 8.4147e-01,  5.1514e-01,  2.8870e-01,  1.5783e-01,  8.5664e-02,  # noqa
              4.6399e-02,  2.5116e-02,  1.3593e-02,  7.3564e-03,  3.9811e-03,  # noqa
              2.1544e-03,  1.1659e-03,  6.3096e-04,  3.4146e-04,  1.8479e-04,  # noqa
              1.0000e-04,  5.4030e-01,  8.5711e-01,  9.5742e-01,  9.8747e-01,  # noqa
              9.9632e-01,  9.9892e-01,  9.9968e-01,  9.9991e-01,  9.9997e-01,  # noqa
              9.9999e-01,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  # noqa
              1.0000e+00,  1.0000e+00],                                        # noqa
            [ 9.0930e-01,  8.8306e-01,  5.5281e-01,  3.1170e-01,  1.7070e-01,  # noqa
              9.2699e-02,  5.0217e-02,  2.7184e-02,  1.4712e-02,  7.9621e-03,  # noqa
              4.3089e-03,  2.3318e-03,  1.2619e-03,  6.8291e-04,  3.6957e-04,  # noqa
              2.0000e-04, -4.1615e-01,  4.6926e-01,  8.3331e-01,  9.5018e-01,  # noqa
              9.8532e-01,  9.9569e-01,  9.9874e-01,  9.9963e-01,  9.9989e-01,  # noqa
              9.9997e-01,  9.9999e-01,  1.0000e+00,  1.0000e+00,  1.0000e+00,  # noqa
              1.0000e+00,  1.0000e+00],                                        # noqa
            [ 1.4112e-01,  9.9861e-01,  7.6984e-01,  4.5775e-01,  2.5448e-01,  # noqa
              1.3880e-01,  7.5285e-02,  4.0769e-02,  2.2067e-02,  1.1943e-02,  # noqa
              6.4633e-03,  3.4977e-03,  1.8929e-03,  1.0244e-03,  5.5436e-04,  # noqa
              3.0000e-04, -9.8999e-01, -5.2688e-02,  6.3823e-01,  8.8908e-01,  # noqa
              9.6708e-01,  9.9032e-01,  9.9716e-01,  9.9917e-01,  9.9976e-01,  # noqa
              9.9993e-01,  9.9998e-01,  9.9999e-01,  1.0000e+00,  1.0000e+00,  # noqa
              1.0000e+00,  1.0000e+00],                                        # noqa
            [-7.5680e-01,  8.2877e-01,  9.2132e-01,  5.9234e-01,  3.3639e-01,  # noqa
              1.8460e-01,  1.0031e-01,  5.4347e-02,  2.9421e-02,  1.5924e-02,  # noqa
              8.6176e-03,  4.6636e-03,  2.5238e-03,  1.3658e-03,  7.3914e-04,  # noqa
              4.0000e-04, -6.5364e-01, -5.5958e-01,  3.8881e-01,  8.0569e-01,  # noqa
              9.4172e-01,  9.8281e-01,  9.9496e-01,  9.9852e-01,  9.9957e-01,  # noqa
              9.9987e-01,  9.9996e-01,  9.9999e-01,  1.0000e+00,  1.0000e+00,  # noqa
              1.0000e+00,  1.0000e+00],                                        # noqa
            [-9.5892e-01,  4.2209e-01,  9.9434e-01,  7.1207e-01,  4.1582e-01,  # noqa
              2.3000e-01,  1.2526e-01,  6.7916e-02,  3.6774e-02,  1.9904e-02,  # noqa
              1.0772e-02,  5.8295e-03,  3.1548e-03,  1.7073e-03,  9.2393e-04,  # noqa
              5.0000e-04,  2.8366e-01, -9.0656e-01,  1.0627e-01,  7.0211e-01,  # noqa
              9.0945e-01,  9.7319e-01,  9.9212e-01,  9.9769e-01,  9.9932e-01,  # noqa
              9.9980e-01,  9.9994e-01,  9.9998e-01,  1.0000e+00,  1.0000e+00,  # noqa
              1.0000e+00,  1.0000e+00],                                        # noqa
            [-2.7942e-01, -1.0523e-01,  9.8268e-01,  8.1396e-01,  4.9220e-01,  # noqa
              2.7491e-01,  1.5014e-01,  8.1471e-02,  4.4124e-02,  2.3884e-02,  # noqa
              1.2926e-02,  6.9954e-03,  3.7857e-03,  2.0487e-03,  1.1087e-03,  # noqa
              6.0000e-04,  9.6017e-01, -9.9445e-01, -1.8531e-01,  5.8092e-01,  # noqa
              8.7048e-01,  9.6147e-01,  9.8866e-01,  9.9668e-01,  9.9903e-01,  # noqa
              9.9971e-01,  9.9992e-01,  9.9998e-01,  9.9999e-01,  1.0000e+00,  # noqa
              1.0000e+00,  1.0000e+00],                                        # noqa
            [ 6.5699e-01, -6.0247e-01,  8.8734e-01,  8.9544e-01,  5.6496e-01,  # noqa
              3.1922e-01,  1.7493e-01,  9.5011e-02,  5.1472e-02,  2.7864e-02,  # noqa
              1.5080e-02,  8.1613e-03,  4.4167e-03,  2.3902e-03,  1.2935e-03,  # noqa
              7.0000e-04,  7.5390e-01, -7.9814e-01, -4.6112e-01,  4.4518e-01,  # noqa
              8.2512e-01,  9.4768e-01,  9.8458e-01,  9.9548e-01,  9.9867e-01,  # noqa
              9.9961e-01,  9.9989e-01,  9.9997e-01,  9.9999e-01,  1.0000e+00,  # noqa
              1.0000e+00,  1.0000e+00],                                        # noqa
            [ 9.8936e-01, -9.2754e-01,  7.1643e-01,  9.5448e-01,  6.3357e-01,  # noqa
              3.6285e-01,  1.9960e-01,  1.0853e-01,  5.8817e-02,  3.1843e-02,  # noqa
              1.7235e-02,  9.3272e-03,  5.0476e-03,  2.7316e-03,  1.4783e-03,  # noqa
              8.0000e-04, -1.4550e-01, -3.7374e-01, -6.9766e-01,  2.9827e-01,  # noqa
              7.7369e-01,  9.3185e-01,  9.7988e-01,  9.9409e-01,  9.9827e-01,  # noqa
              9.9949e-01,  9.9985e-01,  9.9996e-01,  9.9999e-01,  1.0000e+00,  # noqa
              1.0000e+00,  1.0000e+00],                                        # noqa
            [ 4.1212e-01, -9.8752e-01,  4.8452e-01,  9.8959e-01,  6.9752e-01,  # noqa
              4.0570e-01,  2.2415e-01,  1.2204e-01,  6.6159e-02,  3.5822e-02,  # noqa
              1.9389e-02,  1.0493e-02,  5.6786e-03,  3.0731e-03,  1.6631e-03,  # noqa
              9.0000e-04, -9.1113e-01,  1.5748e-01, -8.7478e-01,  1.4389e-01,  # noqa
              7.1657e-01,  9.1401e-01,  9.7455e-01,  9.9253e-01,  9.9781e-01,  # noqa
              9.9936e-01,  9.9981e-01,  9.9994e-01,  9.9998e-01,  1.0000e+00,  # noqa
              1.0000e+00,  1.0000e+00]], device=self.device)                   # noqa
        # fmt: on

    def test_init_initializes_embeddings_correctly(self) -> None:
        m = SinusoidalPositionalEmbedding(
            max_seq_len=10, embedding_dim=32, device=self.device
        )

        expected_weight = self._make_expected_weight()

        self.assertAllClose(m.weight, expected_weight)

    def test_init_prepends_zero_embedding_if_padding_token_idx_is_set(self) -> None:
        m = SinusoidalPositionalEmbedding(
            max_seq_len=10, embedding_dim=32, padding_token_idx=3, device=self.device
        )

        self.assertAllClose(m.weight[0], torch.zeros(32, device=self.device))

        expected_weight = self._make_expected_weight()

        self.assertAllClose(m.weight[1:], expected_weight)

    def test_forward_returns_correct_embeddings(self) -> None:
        m = SinusoidalPositionalEmbedding(
            max_seq_len=10, embedding_dim=4, device=self.device
        )

        b = torch.randint(0, 20, (9, 3), device=self.device)

        a = m(b)

        self.assertEqual(a.shape, (9, 3, 4))

        self.assertAllClose(a, m.weight[:9].unsqueeze(1).expand_as(a))

    def test_forward_returns_correct_embeddings_if_batch_first(self) -> None:
        m = SinusoidalPositionalEmbedding(
            max_seq_len=10, embedding_dim=4, batch_first=True, device=self.device
        )

        b = torch.randint(0, 20, (3, 9), device=self.device)

        a = m(b)

        self.assertEqual(a.shape, (3, 9, 4))

        self.assertAllClose(a, m.weight[:9].expand_as(a))

    def test_forward_returns_correct_embeddings_if_no_batch(self) -> None:
        m = SinusoidalPositionalEmbedding(
            max_seq_len=10, embedding_dim=4, device=self.device
        )

        b = torch.randint(0, 20, (9,), device=self.device)

        a = m(b)

        self.assertAllClose(a, m.weight[:9])

    def test_forward_returns_correct_embeddings_if_padding_token_idx_is_set(
        self,
    ) -> None:
        m = SinusoidalPositionalEmbedding(
            max_seq_len=10,
            embedding_dim=4,
            padding_token_idx=20,
            batch_first=True,
            device=self.device,
        )

        b = torch.randint(0, 19, (3, 8), device=self.device)

        b[:, 4] = 20

        a = m(b)

        self.assertEqual(a.shape, (3, 8, 4))

        expected = torch.cat(
            [m.weight[1:5], torch.zeros(1, 4, device=self.device), m.weight[5:8]], dim=0
        )

        self.assertAllClose(a, expected.expand_as(a))

    def test_forward_returns_correct_embedding_in_incremental_eval(self) -> None:
        for padding_token_idx, embed_idx in ((None, 2), (2, 3), (1, 0)):
            with self.subTest(padding_token_idx=padding_token_idx):
                m = SinusoidalPositionalEmbedding(
                    max_seq_len=3,
                    embedding_dim=32,
                    padding_token_idx=padding_token_idx,
                    device=self.device,
                )

                m.eval()

                b = torch.ones(3, 5, device=self.device)

                a = m(b, incremental_eval=True)

                self.assertEqual(a.shape, (1, 5, 32))

                self.assertAllClose(a, m.weight[embed_idx].expand_as(a))

    def test_forward_errors_if_input_dim_is_greater_than_2(self) -> None:
        m = SinusoidalPositionalEmbedding(
            max_seq_len=3, embedding_dim=32, device=self.device
        )

        b = torch.ones((5, 5, 5), device=self.device)

        with self.assertRaisesRegex(
            ValueError, r"The number of dimensions of `seq` \(3\) must be 1 or 2."
        ):
            m(b)

    def test_forward_errors_if_seq_len_is_out_of_range(self) -> None:
        m = SinusoidalPositionalEmbedding(
            max_seq_len=3, embedding_dim=32, device=self.device
        )

        b = torch.ones((5), device=self.device)

        with self.assertRaisesRegex(
            ValueError, r"The input sequence length \(5\) cannot be greater than 3."
        ):
            m(b)

    def test_forward_ignores_incremental_in_training(self) -> None:
        m = SinusoidalPositionalEmbedding(
            max_seq_len=3, embedding_dim=32, device=self.device
        )

        b = torch.ones((2, 5), device=self.device)

        a = m(b, incremental_eval=True)

        self.assertEqual(a.shape, (2, 5, 32))


class TestLearnedPositionalEmbedding(TestCase):
    def test_init_initializes_embeddings_correctly(self) -> None:
        with tmp_rng_seed(self.device):
            m = LearnedPositionalEmbedding(
                max_seq_len=10, embedding_dim=32, device=self.device
            )

        self.assertEqual(m.weight.dtype, torch.float)

        with tmp_rng_seed(self.device):
            expected_weight = torch.randn(10, 32, device=self.device)

        self.assertAllClose(m.weight, expected_weight)

    def test_init_prepends_zero_embedding_if_padding_token_idx_is_set(self) -> None:
        with tmp_rng_seed(self.device):
            m = LearnedPositionalEmbedding(
                max_seq_len=10,
                embedding_dim=32,
                padding_token_idx=3,
                device=self.device,
            )

        self.assertAllClose(m.weight[0], torch.zeros(32, device=self.device))

        with tmp_rng_seed(self.device):
            expected_weight = torch.randn(11, 32, device=self.device)

        self.assertAllClose(m.weight[1:], expected_weight[1:])

    def test_forward_returns_correct_embeddings(self) -> None:
        m = LearnedPositionalEmbedding(
            max_seq_len=10, embedding_dim=4, device=self.device
        )

        b = torch.randint(0, 20, (9, 3), device=self.device)

        a = m(b)

        self.assertEqual(a.shape, (9, 3, 4))

        self.assertAllClose(a, m.weight[:9].unsqueeze(1).expand_as(a))

    def test_forward_returns_correct_embeddings_if_batch_first(self) -> None:
        m = LearnedPositionalEmbedding(
            max_seq_len=10, embedding_dim=4, batch_first=True, device=self.device
        )

        b = torch.randint(0, 20, (3, 9), device=self.device)

        a = m(b)

        self.assertEqual(a.shape, (3, 9, 4))

        self.assertAllClose(a, m.weight[:9].expand_as(a))

    def test_forward_returns_correct_embeddings_if_no_batch(self) -> None:
        m = LearnedPositionalEmbedding(
            max_seq_len=10, embedding_dim=4, device=self.device
        )

        b = torch.randint(0, 20, (9,), device=self.device)

        a = m(b)

        self.assertAllClose(a, m.weight[:9])

    def test_forward_returns_correct_embeddings_if_padding_token_idx_is_set(
        self,
    ) -> None:
        m = LearnedPositionalEmbedding(
            max_seq_len=10,
            embedding_dim=4,
            padding_token_idx=20,
            batch_first=True,
            device=self.device,
        )

        b = torch.randint(0, 19, (3, 8), device=self.device)

        b[:, 4] = 20

        a = m(b)

        self.assertEqual(a.shape, (3, 8, 4))

        expected = torch.cat(
            [m.weight[1:5], torch.zeros(1, 4, device=self.device), m.weight[5:8]], dim=0
        )

        self.assertAllClose(a, expected.expand_as(a))

    def test_forward_returns_correct_embedding_in_incremental_eval(self) -> None:
        for padding_token_idx, embed_idx in ((None, 2), (2, 3), (1, 0)):
            with self.subTest(padding_token_idx=padding_token_idx):
                m = LearnedPositionalEmbedding(
                    max_seq_len=3,
                    embedding_dim=32,
                    padding_token_idx=padding_token_idx,
                    device=self.device,
                )

                m.eval()

                b = torch.ones(3, 5, device=self.device)

                a = m(b, incremental_eval=True)

                self.assertEqual(a.shape, (1, 5, 32))

                self.assertAllClose(a, m.weight[embed_idx].expand_as(a))

    def test_forward_errors_if_input_dim_is_greater_than_2(self) -> None:
        m = LearnedPositionalEmbedding(
            max_seq_len=3, embedding_dim=32, device=self.device
        )

        b = torch.ones((5, 5, 5), device=self.device)

        with self.assertRaisesRegex(
            ValueError, r"The number of dimensions of `seq` \(3\) must be 1 or 2."
        ):
            m(b)

    def test_forward_errors_if_seq_len_is_out_of_range(self) -> None:
        m = LearnedPositionalEmbedding(
            max_seq_len=3, embedding_dim=32, device=self.device
        )

        b = torch.ones((5), device=self.device)

        with self.assertRaisesRegex(
            ValueError, r"The input sequence length \(5\) cannot be greater than 3."
        ):
            m(b)

    def test_forward_ignores_incremental_in_training(self) -> None:
        m = LearnedPositionalEmbedding(
            max_seq_len=3, embedding_dim=32, device=self.device
        )

        b = torch.ones((2, 5), device=self.device)

        a = m(b, incremental_eval=True)

        self.assertEqual(a.shape, (2, 5, 32))
