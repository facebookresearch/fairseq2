# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random

import pytest
import torch
from torch import Tensor

from fairseq2.nn.audio import SpecAugmentTransform
from tests.common import assert_close, device


class TestSpecAugmentTransform:
    def test_output_has_same_shape(self) -> None:
        specgram = torch.rand((1, 10, 10))

        spec_aug = SpecAugmentTransform()
        augmented = spec_aug(specgram)

        assert augmented.shape == specgram.shape

    def test_output_is_copy(self) -> None:
        specgram = torch.rand((1, 10, 10))

        spec_aug = SpecAugmentTransform()
        augmented = spec_aug(specgram)

        assert specgram.data_ptr() != augmented.data_ptr()

    def test_is_training_no_transform(self) -> None:
        specgram = torch.rand((1, 10, 10))

        spec_aug = SpecAugmentTransform()
        spec_aug.training = False
        augmented = spec_aug(specgram)

        assert torch.equal(augmented, specgram)

    def test_stretch_param_is_exceeded(self) -> None:
        specgram = torch.rand((1, 10, 10))

        exceeded_stretch_param = 8
        spec_aug = SpecAugmentTransform(max_stretch_length=exceeded_stretch_param)

        with pytest.raises(ValueError):
            spec_aug(specgram)

    @staticmethod
    def expected_specgram() -> Tensor:
        # fmt: off
        return torch.tensor([
            [[0.57826, 0.44452, 0.11746, 0.54816, 0.75437, 0.60912, 0.00000,
              0.53873, 0.59796, 0.55851, 0.65565, 0.33211, 0.00000, 0.45804,
              0.51373, 0.54588, 0.59209, 0.91698, 0.67570, 0.38704, 0.00000,
              0.30711, 0.36910, 0.29621, 0.33978],
             [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
              0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
              0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
              0.00000, 0.00000, 0.00000, 0.00000],
             [0.11017, 0.72221, 0.92058, 0.63567, 0.16626, 0.66772, 0.00000,
              0.41598, 0.40821, 0.82022, 0.81354, 0.93345, 0.00000, 0.34699,
              0.45401, 0.51986, 0.47607, 0.12230, 0.14343, 0.63769, 0.00000,
              0.56350, 0.66719, 0.55758, 0.66417],
             [0.63689, 0.65899, 0.66233, 0.13238, 0.53041, 0.43957, 0.00000,
              0.50205, 0.15510, 0.22873, 0.83260, 0.45715, 0.00000, 0.70377,
              0.39290, 0.29484, 0.60466, 0.58761, 0.19771, 0.24186, 0.00000,
              0.63002, 0.48162, 0.30046, 0.34864],
             [0.75152, 0.77372, 0.24426, 0.64051, 0.52611, 0.77488, 0.00000,
              0.63221, 0.73676, 0.89504, 0.91654, 0.22577, 0.00000, 0.76220,
              0.86767, 0.84379, 0.66652, 0.13827, 0.13982, 0.18526, 0.00000,
              0.78057, 0.70756, 0.30894, 0.76644],
             [0.15074, 0.44302, 0.94217, 0.45137, 0.93143, 0.85278, 0.00000,
              0.83346, 0.84137, 0.77616, 0.59550, 0.19174, 0.00000, 0.20276,
              0.35633, 0.35312, 0.29840, 0.68091, 0.65676, 0.44420, 0.00000,
              0.67688, 0.71291, 0.81201, 0.83672],
             [0.57736, 0.44727, 0.08491, 0.19676, 0.28614, 0.73348, 0.00000,
              0.23148, 0.32851, 0.68546, 0.67178, 0.81215, 0.00000, 0.47728,
              0.46319, 0.34286, 0.32635, 0.11372, 0.25994, 0.33085, 0.00000,
              0.65258, 0.60283, 0.70398, 0.99905],
             [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
              0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
              0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
              0.00000, 0.00000, 0.00000, 0.00000],
             [0.43704, 0.84927, 0.73929, 0.48124, 0.48405, 0.56336, 0.00000,
              0.65794, 0.60602, 0.52347, 0.10838, 0.80612, 0.00000, 0.92138,
              0.89146, 0.60874, 0.17678, 0.35938, 0.89613, 0.29525, 0.00000,
              0.53018, 0.12141, 0.27565, 0.74444],
             [0.23252, 0.86778, 0.96327, 0.21906, 0.99231, 0.52181, 0.00000,
              0.45808, 0.45348, 0.29569, 0.82755, 0.81446, 0.00000, 0.09429,
              0.27276, 0.54685, 0.66191, 0.46844, 0.26354, 0.45465, 0.00000,
              0.08833, 0.15395, 0.32672, 0.91177]]], device=device)
        # fmt: on

    def test_forward_returns_correct_specgram(self) -> None:
        torch.manual_seed(1)
        random.seed(1)
        specgram = torch.rand((1, 10, 25))

        params = {
            "stretch_axis": 2,
            "max_stretch_length": 5,
            "num_freq_masks": 3,
            "freq_max_mask_length": 3,
            "num_time_masks": 3,
            "time_max_mask_length": 3,
        }
        spec_aug = SpecAugmentTransform(**params)

        augmented = spec_aug(specgram)

        assert_close(augmented, self.expected_specgram())
