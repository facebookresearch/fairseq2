# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import pytest
import torch

from fairseq2.models.olmo.yarn_rope import YaRNRotaryEncoder
from tests.common import device


class TestYaRNRotaryEncoder:
    def test_init_rejects_odd_encoding_dim(self) -> None:
        with pytest.raises(ValueError, match=r"encoding_dim.*even"):
            YaRNRotaryEncoder(
                encoding_dim=13,
                max_seq_len=16,
                scale_factor=8.0,
                original_max_seq_len=2048,
                device=device,
            )

    def test_attention_scaling_formula(self) -> None:
        # For scale_factor=8.0, mscale=1.0, mscale_all_dim=0.0:
        # Since mscale_all_dim == 0.0, we use get_mscale(8.0, 1.0)
        # = 0.1 * 1.0 * math.log(8.0) + 1.0
        expected = 0.1 * 1.0 * math.log(8.0) + 1.0

        encoder = YaRNRotaryEncoder(
            encoding_dim=4,
            max_seq_len=4,
            scale_factor=8.0,
            original_max_seq_len=2,
            mscale=1.0,
            mscale_all_dim=0.0,
            device=device,
        )

        assert abs(encoder.attention_scaling - expected) < 1e-6

    def test_buffers_for_small_config(self) -> None:
        encoder = YaRNRotaryEncoder(
            encoding_dim=4,
            max_seq_len=4,
            scale_factor=1.0,  # scale<=1 means no scaling -> extrapolation only
            original_max_seq_len=4,
            theta=10_000.0,
            beta_fast=32.0,
            beta_slow=1.0,
            mscale=1.0,
            mscale_all_dim=0.0,
            truncate=True,
            device=device,
        )

        # Row 0 should be all zeros (padding)
        assert torch.all(encoder.cos_freqs[0] == 0.0)
        assert torch.all(encoder.sin_freqs[0] == 0.0)

        # Remaining rows should not be identically zero
        assert not torch.all(encoder.cos_freqs[1:] == 0.0)
        assert not torch.all(encoder.sin_freqs[1:] == 0.0)

        # Verify shape
        assert encoder.cos_freqs.shape == (5, 4)  # max_seq_len+1, encoding_dim
        assert encoder.sin_freqs.shape == (5, 4)

    def test_extra_repr_shows_scale_factor(self) -> None:
        encoder = YaRNRotaryEncoder(
            encoding_dim=4,
            max_seq_len=4,
            scale_factor=8.0,
            original_max_seq_len=2,
            device=device,
        )
        repr_str = encoder.extra_repr()
        assert "scale_factor=" in repr_str
        assert "attention_scaling=" in repr_str
