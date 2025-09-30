# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fairseq2.nn import (
    BatchLayout,
    IncrementalStateBag,
    LearnedPositionEncoder,
    RotaryEncoder,
    Sinusoidal2dPositionEncoder,
    Sinusoidal3dPositionEncoder,
    SinusoidalPositionEncoder,
)
from tests.common import assert_close, device, temporary_manual_seed


class TestSinusoidalPositionEncoder:
    @staticmethod
    def expected_freqs() -> Tensor:
        # fmt: off
        return torch.tensor([
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
              0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
              0.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,
              1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,
              1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,
              1.0000e+00,  1.0000e+00],
            [ 8.4147e-01,  5.1514e-01,  2.8870e-01,  1.5783e-01,  8.5664e-02,
              4.6399e-02,  2.5116e-02,  1.3593e-02,  7.3564e-03,  3.9811e-03,
              2.1544e-03,  1.1659e-03,  6.3096e-04,  3.4146e-04,  1.8479e-04,
              1.0000e-04,  5.4030e-01,  8.5711e-01,  9.5742e-01,  9.8747e-01,
              9.9632e-01,  9.9892e-01,  9.9968e-01,  9.9991e-01,  9.9997e-01,
              9.9999e-01,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,
              1.0000e+00,  1.0000e+00],
            [ 9.0930e-01,  8.8306e-01,  5.5281e-01,  3.1170e-01,  1.7070e-01,
              9.2699e-02,  5.0217e-02,  2.7184e-02,  1.4712e-02,  7.9621e-03,
              4.3089e-03,  2.3318e-03,  1.2619e-03,  6.8291e-04,  3.6957e-04,
              2.0000e-04, -4.1615e-01,  4.6926e-01,  8.3331e-01,  9.5018e-01,
              9.8532e-01,  9.9569e-01,  9.9874e-01,  9.9963e-01,  9.9989e-01,
              9.9997e-01,  9.9999e-01,  1.0000e+00,  1.0000e+00,  1.0000e+00,
              1.0000e+00,  1.0000e+00],
            [ 1.4112e-01,  9.9861e-01,  7.6984e-01,  4.5775e-01,  2.5448e-01,
              1.3880e-01,  7.5285e-02,  4.0769e-02,  2.2067e-02,  1.1943e-02,
              6.4633e-03,  3.4977e-03,  1.8929e-03,  1.0244e-03,  5.5436e-04,
              3.0000e-04, -9.8999e-01, -5.2688e-02,  6.3823e-01,  8.8908e-01,
              9.6708e-01,  9.9032e-01,  9.9716e-01,  9.9917e-01,  9.9976e-01,
              9.9993e-01,  9.9998e-01,  9.9999e-01,  1.0000e+00,  1.0000e+00,
              1.0000e+00,  1.0000e+00],
            [-7.5680e-01,  8.2877e-01,  9.2132e-01,  5.9234e-01,  3.3639e-01,
              1.8460e-01,  1.0031e-01,  5.4347e-02,  2.9421e-02,  1.5924e-02,
              8.6176e-03,  4.6636e-03,  2.5238e-03,  1.3658e-03,  7.3914e-04,
              4.0000e-04, -6.5364e-01, -5.5958e-01,  3.8881e-01,  8.0569e-01,
              9.4172e-01,  9.8281e-01,  9.9496e-01,  9.9852e-01,  9.9957e-01,
              9.9987e-01,  9.9996e-01,  9.9999e-01,  1.0000e+00,  1.0000e+00,
              1.0000e+00,  1.0000e+00],
            [-9.5892e-01,  4.2209e-01,  9.9434e-01,  7.1207e-01,  4.1582e-01,
              2.3000e-01,  1.2526e-01,  6.7916e-02,  3.6774e-02,  1.9904e-02,
              1.0772e-02,  5.8295e-03,  3.1548e-03,  1.7073e-03,  9.2393e-04,
              5.0000e-04,  2.8366e-01, -9.0656e-01,  1.0627e-01,  7.0211e-01,
              9.0945e-01,  9.7319e-01,  9.9212e-01,  9.9769e-01,  9.9932e-01,
              9.9980e-01,  9.9994e-01,  9.9998e-01,  1.0000e+00,  1.0000e+00,
              1.0000e+00,  1.0000e+00],
            [-2.7942e-01, -1.0523e-01,  9.8268e-01,  8.1396e-01,  4.9220e-01,
              2.7491e-01,  1.5014e-01,  8.1471e-02,  4.4124e-02,  2.3884e-02,
              1.2926e-02,  6.9954e-03,  3.7857e-03,  2.0487e-03,  1.1087e-03,
              6.0000e-04,  9.6017e-01, -9.9445e-01, -1.8531e-01,  5.8092e-01,
              8.7048e-01,  9.6147e-01,  9.8866e-01,  9.9668e-01,  9.9903e-01,
              9.9971e-01,  9.9992e-01,  9.9998e-01,  9.9999e-01,  1.0000e+00,
              1.0000e+00,  1.0000e+00],
            [ 6.5699e-01, -6.0247e-01,  8.8734e-01,  8.9544e-01,  5.6496e-01,
              3.1922e-01,  1.7493e-01,  9.5011e-02,  5.1472e-02,  2.7864e-02,
              1.5080e-02,  8.1613e-03,  4.4167e-03,  2.3902e-03,  1.2935e-03,
              7.0000e-04,  7.5390e-01, -7.9814e-01, -4.6112e-01,  4.4518e-01,
              8.2512e-01,  9.4768e-01,  9.8458e-01,  9.9548e-01,  9.9867e-01,
              9.9961e-01,  9.9989e-01,  9.9997e-01,  9.9999e-01,  1.0000e+00,
              1.0000e+00,  1.0000e+00],
            [ 9.8936e-01, -9.2754e-01,  7.1643e-01,  9.5448e-01,  6.3357e-01,
              3.6285e-01,  1.9960e-01,  1.0853e-01,  5.8817e-02,  3.1843e-02,
              1.7235e-02,  9.3272e-03,  5.0476e-03,  2.7316e-03,  1.4783e-03,
              8.0000e-04, -1.4550e-01, -3.7374e-01, -6.9766e-01,  2.9827e-01,
              7.7369e-01,  9.3185e-01,  9.7988e-01,  9.9409e-01,  9.9827e-01,
              9.9949e-01,  9.9985e-01,  9.9996e-01,  9.9999e-01,  1.0000e+00,
              1.0000e+00,  1.0000e+00],
            [ 4.1212e-01, -9.8752e-01,  4.8452e-01,  9.8959e-01,  6.9752e-01,
              4.0570e-01,  2.2415e-01,  1.2204e-01,  6.6159e-02,  3.5822e-02,
              1.9389e-02,  1.0493e-02,  5.6786e-03,  3.0731e-03,  1.6631e-03,
              9.0000e-04, -9.1113e-01,  1.5748e-01, -8.7478e-01,  1.4389e-01,
              7.1657e-01,  9.1401e-01,  9.7455e-01,  9.9253e-01,  9.9781e-01,
              9.9936e-01,  9.9981e-01,  9.9994e-01,  9.9998e-01,  1.0000e+00,
              1.0000e+00,  1.0000e+00]], device=device)
        # fmt: on

    def test_init_works(self) -> None:
        m = SinusoidalPositionEncoder(encoding_dim=32, max_seq_len=10, device=device)

        assert_close(m.freqs, self.expected_freqs())

    def test_init_raises_error_when_encoding_dim_is_odd(self) -> None:
        with pytest.raises(
            ValueError, match=r"^`encoding_dim` must be even, but is 13 instead\.$"
        ):
            SinusoidalPositionEncoder(encoding_dim=13, max_seq_len=10, device=device)

    def test_forward_works(self) -> None:
        m = SinusoidalPositionEncoder(encoding_dim=4, max_seq_len=10, device=device)

        x = torch.randn((3, 9, 4), device=device)

        x_layout = BatchLayout.of(x)

        y = m(x, x_layout)

        assert y.shape == (3, 9, 4)

        assert_close(y - x, m.freqs[1:10].expand_as(y))

        # Test with multiple dimensions.
        x = torch.randn((4, 3, 9, 4), device=device)

        x_layout = BatchLayout.of(x)

        y = m(x, x_layout)

        assert y.shape == (4, 3, 9, 4)

        freqs = m.freqs[1:4].unsqueeze(1).expand_as(y)

        assert_close(y - x, freqs)

    def test_forward_works_with_padding(self) -> None:
        m = SinusoidalPositionEncoder(encoding_dim=4, max_seq_len=10, device=device)

        x = torch.randn((4, 9, 3, 4), device=device)

        x[0, 7:] = 0.0
        x[2, 5:] = 0.0

        x_layout = BatchLayout.of(x, seq_lens=[7, 9, 5, 9])

        y = m(x, x_layout)

        assert y.shape == (4, 9, 3, 4)

        freqs = m.freqs[1:10].unsqueeze(1).expand_as(y).clone()

        freqs[0, 7:] = 0.0
        freqs[2, 5:] = 0.0

        assert_close(y - x, freqs)

    def test_forward_works_with_packing(self) -> None:
        m = SinusoidalPositionEncoder(encoding_dim=4, max_seq_len=10, device=device)

        x = torch.randn((20, 4), device=device)

        x[18:] = 0.0

        x_layout = BatchLayout.of(x, seq_lens=[5, 10, 3], packed=True)

        y = m(x, x_layout)

        assert y.shape == (20, 4)

        freqs = torch.cat((m.freqs[1:6], m.freqs[1:11], m.freqs[1:4]))

        assert_close(y[:18] - x[:18], freqs[:18])

        assert_close(torch.sum(y[18:]), 0.0)

    @pytest.mark.parametrize("step_nr", [0, 1, 2])
    def test_forward_works_in_incremental_decode(self, step_nr: int) -> None:
        m = SinusoidalPositionEncoder(encoding_dim=32, max_seq_len=4, device=device)

        state_bag = IncrementalStateBag(max_num_steps=3)

        state_bag.increment_step_nr(step_nr)

        seq_len = 2

        m.eval()

        x = torch.randn((5, seq_len, 32), device=device)

        x_layout = BatchLayout.of(x)

        y = m(x, x_layout, state_bag=state_bag)

        assert y.shape == (5, seq_len, 32)

        assert_close(y - x, m.freqs[step_nr + 1 : step_nr + 1 + seq_len].expand_as(y))

    def test_forward_raises_error_when_seq_len_is_out_of_range(self) -> None:
        m = SinusoidalPositionEncoder(encoding_dim=32, max_seq_len=3, device=device)

        x = torch.randn((1, 5, 32), device=device)

        x_layout = BatchLayout.of(x)

        with pytest.raises(
            ValueError, match=r"^The lengths of all sequences in `seqs` must be less than or equal to the maximum sequence length \(3\), but at least one sequence is of length 5 instead\.$"  # fmt: skip
        ):
            m(x, x_layout)

    def test_forward_works_when_state_bag_is_not_none_in_training(self) -> None:
        m = SinusoidalPositionEncoder(encoding_dim=32, max_seq_len=3, device=device)

        x = torch.randn((5, 2, 32), device=device)

        x_layout = BatchLayout.of(x)

        state_bag = IncrementalStateBag(max_num_steps=30)

        state_bag.increment_step_nr(20)  # out of range

        y = m(x, x_layout, state_bag=state_bag)

        assert y.shape == (5, 2, 32)


class TestLearnedPositionEncoder:
    def test_init_works(self) -> None:
        with temporary_manual_seed(2, device):
            m = LearnedPositionEncoder(encoding_dim=32, max_seq_len=10, device=device)

        assert m.weight.dtype == torch.float32

        with temporary_manual_seed(2, device):
            expected_weight = torch.randn(11, 32, device=device)

            expected_weight[0] = 0.0

        assert_close(m.weight, expected_weight)

    def test_forward_works(self) -> None:
        m = LearnedPositionEncoder(encoding_dim=4, max_seq_len=10, device=device)

        x = torch.randn((3, 9, 4), device=device)

        x_layout = BatchLayout.of(x)

        y = m(x, x_layout)

        assert y.shape == (3, 9, 4)

        assert_close(y - x, m.weight[1:10].expand_as(y))

        # Test with multiple dimensions.
        x = torch.randn((4, 3, 9, 4), device=device)

        x_layout = BatchLayout.of(x)

        y = m(x, x_layout)

        assert y.shape == (4, 3, 9, 4)

        weight = m.weight[1:4].unsqueeze(1).expand_as(y)

        assert_close(y - x, weight)

    def test_forward_works_with_padding(self) -> None:
        m = LearnedPositionEncoder(encoding_dim=4, max_seq_len=10, device=device)

        x = torch.randn((4, 9, 3, 4), device=device)

        x[0, 7:] = 0.0
        x[2, 5:] = 0.0

        x_layout = BatchLayout.of(x, seq_lens=[7, 9, 5, 9])

        y = m(x, x_layout)

        assert y.shape == (4, 9, 3, 4)

        freqs = m.weight[1:10].unsqueeze(1).expand_as(y).clone()

        freqs[0, 7:] = 0.0
        freqs[2, 5:] = 0.0

        assert_close(y - x, freqs)

    def test_forward_works_with_packing(self) -> None:
        m = LearnedPositionEncoder(encoding_dim=4, max_seq_len=10, device=device)

        x = torch.randn((20, 4), device=device)

        x[18:] = 0.0

        x_layout = BatchLayout.of(x, seq_lens=[5, 10, 3], packed=True)

        y = m(x, x_layout)

        assert y.shape == (20, 4)

        weight = torch.cat((m.weight[1:6], m.weight[1:11], m.weight[1:4]))

        assert_close(y[:18] - x[:18], weight[:18])

        assert_close(torch.sum(y[18:]), 0.0)

    @pytest.mark.parametrize("step_nr", [0, 1, 2])
    def test_forward_works_in_incremental_decode(self, step_nr: int) -> None:
        m = LearnedPositionEncoder(encoding_dim=32, max_seq_len=4, device=device)

        state_bag = IncrementalStateBag(max_num_steps=3)

        state_bag.increment_step_nr(step_nr)

        seq_len = 2

        m.eval()

        x = torch.randn((5, seq_len, 32), device=device)

        x_layout = BatchLayout.of(x)

        y = m(x, x_layout, state_bag=state_bag)

        assert y.shape == (5, seq_len, 32)

        assert_close(y - x, m.weight[step_nr + 1 : step_nr + 1 + seq_len].expand_as(y))

    def test_forward_raises_error_when_seq_len_is_out_of_range(self) -> None:
        m = LearnedPositionEncoder(encoding_dim=32, max_seq_len=3, device=device)

        x = torch.randn((1, 5, 32), device=device)

        x_layout = BatchLayout.of(x)

        with pytest.raises(
            ValueError, match=r"^The lengths of all sequences in `seqs` must be less than or equal to the maximum sequence length \(3\), but at least one sequence is of length 5 instead\.$"  # fmt: skip
        ):
            m(x, x_layout)

    def test_forward_works_when_state_bag_is_not_none_in_training(self) -> None:
        m = LearnedPositionEncoder(encoding_dim=32, max_seq_len=3, device=device)

        x = torch.randn((5, 2, 32), device=device)

        x_layout = BatchLayout.of(x)

        state_bag = IncrementalStateBag(max_num_steps=30)

        state_bag.increment_step_nr(value=20)  # out of range

        y = m(x, x_layout, state_bag=state_bag)

        assert y.shape == (5, 2, 32)


class TestRotaryEncoder:
    def test_init_raises_error_when_encoding_dim_is_odd(self) -> None:
        with pytest.raises(
            ValueError, match=r"^`encoding_dim` must be even, but is 13 instead\.$"
        ):
            RotaryEncoder(encoding_dim=13, max_seq_len=10, device=device)

    def test_forward_works(self) -> None:
        m = RotaryEncoder(encoding_dim=4, max_seq_len=10, device=device)

        x = torch.randn((4, 3, 9, 4), device=device)

        x_layout = BatchLayout.of(x, seq_lens=[3, 1, 3, 2])

        y = m(x, x_layout)

        # We apply a rotation, the magnitudes should stay the same.
        assert_close(torch.norm(x[0]), torch.norm(y[0]))
        assert_close(torch.norm(x[2]), torch.norm(y[2]))

        assert_close(torch.norm(x[1, :1]), torch.norm(y[1, :1]))
        assert_close(torch.norm(x[3, :2]), torch.norm(y[3, :2]))

        assert_close(torch.sum(y[1, 1:]), 0.0)
        assert_close(torch.sum(y[3, 2:]), 0.0)

        x1 = torch.randn((4), device=device)
        x2 = torch.randn((4), device=device)

        seq1 = torch.zeros((1, 6, 4), device=device)

        seq1_layout = BatchLayout.of(seq1)

        seq1[0, 1] = x1
        seq1[0, 4] = x2

        y1 = m(seq1, seq1_layout)

        seq2 = torch.zeros((1, 6, 4), device=device)

        seq2_layout = BatchLayout.of(seq2)

        seq2[0, 2] = x1
        seq2[0, 5] = x2

        y2 = m(seq2, seq2_layout)

        # If the angles are same, the dot-product must be same as well.
        dot1 = torch.dot(y1[0, 1], y1[0, 4])
        dot2 = torch.dot(y2[0, 2], y2[0, 5])

        assert_close(dot1, dot2)

    def test_forward_works_with_packing(self) -> None:
        m = RotaryEncoder(encoding_dim=4, max_seq_len=10, device=device)

        x = torch.ones((12, 4), device=device)

        x[10:] = 0.0

        x_layout = BatchLayout.of(x, seq_lens=[4, 4, 2], packed=True)

        y = m(x, x_layout)

        assert y.shape == (12, 4)

        assert_close(torch.norm(x[:10]), torch.norm(y[:10]))

        assert_close(x[:4], x[4:8])
        assert_close(x[:2], x[8:10])

        assert_close(torch.sum(y[10:]), 0.0)

    @pytest.mark.parametrize("step_nr", [0, 1, 2])
    def test_forward_works_in_incremental_decode(self, step_nr: int) -> None:
        m = RotaryEncoder(encoding_dim=32, max_seq_len=4, device=device)

        state_bag = IncrementalStateBag(max_num_steps=3)

        state_bag.increment_step_nr(step_nr)

        seq_len = 2

        m.eval()

        x1 = torch.ones((5, seq_len, 32), device=device)

        x1_layout = BatchLayout.of(x1)

        y1 = m(x1, x1_layout, state_bag=state_bag)

        assert y1.shape == (5, seq_len, 32)

        x2 = torch.ones((5, seq_len + step_nr, 32), device=device)

        x2_layout = BatchLayout.of(x2)

        y2 = m(x2, x2_layout)

        assert_close(y1, y2[:, step_nr:])

    def test_forward_raises_error_when_seq_len_is_out_of_range(self) -> None:
        m = RotaryEncoder(encoding_dim=32, max_seq_len=3, device=device)

        x = torch.randn((1, 5, 32), device=device)

        x_layout = BatchLayout.of(x)

        with pytest.raises(
            ValueError, match=r"^The lengths of all sequences in `seqs` must be less than or equal to the maximum sequence length \(3\), but at least one sequence is of length 5 instead\.$"  # fmt: skip
        ):
            m(x, x_layout)

    def test_forward_works_when_state_bag_is_not_none_in_training(self) -> None:
        m = RotaryEncoder(encoding_dim=32, max_seq_len=3, device=device)

        x = torch.randn((5, 2, 32), device=device)

        x_layout = BatchLayout.of(x)

        state_bag = IncrementalStateBag(max_num_steps=30)

        state_bag.increment_step_nr(20)  # out of range

        y = m(x, x_layout, state_bag=state_bag)

        assert y.shape == (5, 2, 32)

    def test_forward_works_with_custom_freqs_init(self) -> None:
        def custom_freqs_init(encoder: RotaryEncoder) -> torch.Tensor:
            return torch.ones(encoder.encoding_dim // 2, device=device)

        m = RotaryEncoder(
            encoding_dim=4,
            max_seq_len=10,
            freqs_init_fn=custom_freqs_init,
            device=device,
        )

        x = torch.randn((3, 9, 4), device=device)

        x_layout = BatchLayout.of(x)

        y = m(x, x_layout)

        assert y.shape == (3, 9, 4)


class TestSinusoidal2dPositionEncoder:
    def test_init_raises_error_when_encoding_dim_is_odd(self) -> None:
        with pytest.raises(
            ValueError, match=r"^`encoding_dim` must be even, but is 13 instead\.$"
        ):
            Sinusoidal2dPositionEncoder(
                encoding_dim=13, grid_dims=(10, 10), device=device
            )

    def test_forward_works(self) -> None:
        m = Sinusoidal2dPositionEncoder(encoding_dim=4, grid_dims=(8, 8), device=device)

        # Test with same dimensions as grid
        x = torch.randn((2, 8, 8, 4), device=device)
        y = m(x)

        assert y.shape == (2, 8, 8, 4)
        assert y.dtype == x.dtype

        # Test with different dimensions (should trigger interpolation)
        x = torch.randn((2, 16, 16, 4), device=device)
        y = m(x)

        assert y.shape == (2, 16, 16, 4)
        assert y.dtype == x.dtype

    def test_forward_raises_error_on_wrong_dims(self) -> None:
        m = Sinusoidal2dPositionEncoder(encoding_dim=4, grid_dims=(8, 8), device=device)

        # Test with wrong number of dimensions
        x = torch.randn((2, 8, 4), device=device)

        with pytest.raises(
            ValueError,
            match=r"^`x` must be 4 dimensional, but is 3 dimensional instead\.$",
        ):
            m(x)

    def test_extra_repr_works(self) -> None:
        m = Sinusoidal2dPositionEncoder(encoding_dim=4, grid_dims=(8, 8), device=device)

        assert m.extra_repr() == "encoding_dim=4, grid_dims=(8, 8)"


class TestSinusoidal3dPositionEncoder:
    def test_init_raises_error_when_encoding_dim_is_odd(self) -> None:
        with pytest.raises(
            ValueError, match=r"^`encoding_dim` must be even, but is 13 instead\.$"
        ):
            Sinusoidal3dPositionEncoder(
                encoding_dim=13, grid_dims=(8, 8, 8), device=device
            )

    def test_forward_works(self) -> None:
        m = Sinusoidal3dPositionEncoder(
            encoding_dim=6, grid_dims=(4, 4, 4), device=device
        )

        # Test with same dimensions as grid
        x = torch.randn((2, 4, 4, 4, 6), device=device)
        y = m(x)

        assert y.shape == (2, 4, 4, 4, 6)
        assert y.dtype == x.dtype

        # Test with different dimensions (should trigger interpolation)
        x = torch.randn((2, 8, 8, 8, 6), device=device)
        y = m(x)

        assert y.shape == (2, 8, 8, 8, 6)
        assert y.dtype == x.dtype

    def test_forward_works_with_uniform_power(self) -> None:
        m = Sinusoidal3dPositionEncoder(
            encoding_dim=6, grid_dims=(4, 4, 4), uniform_power=True, device=device
        )

        x = torch.randn((2, 4, 4, 4, 6), device=device)
        y = m(x)

        assert y.shape == (2, 4, 4, 4, 6)

    def test_forward_raises_error_on_wrong_dims(self) -> None:
        m = Sinusoidal3dPositionEncoder(
            encoding_dim=6, grid_dims=(4, 4, 4), device=device
        )

        # Test with wrong number of dimensions
        x = torch.randn((2, 4, 4, 6), device=device)

        with pytest.raises(
            ValueError,
            match=r"^`x` must be 5 dimensional, but is 4 dimensional instead\.$",
        ):
            m(x)

    def test_extra_repr_works(self) -> None:
        m = Sinusoidal3dPositionEncoder(
            encoding_dim=6, grid_dims=(4, 4, 4), device=device
        )

        assert m.extra_repr() == "encoding_dim=6, grid_dims=(4, 4, 4)"
