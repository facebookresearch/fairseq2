# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for NemotronH Mamba2 SSM module."""

from __future__ import annotations

import pytest
import torch

from fairseq2.models.nemotron.mamba2 import (
    NemotronHMamba2Mixer,
    NemotronHMamba2State,
    RMSNormGated,
)
from fairseq2.nn import IncrementalStateBag


class TestRMSNormGated:
    def test_output_shape(self) -> None:
        norm = RMSNormGated(64)
        x = torch.randn(2, 8, 64)
        gate = torch.randn(2, 8, 64)
        out = norm(x, gate)
        assert out.shape == (2, 8, 64)

    def test_zero_gate(self) -> None:
        """With zero gate, SiLU(0) = 0, so output should be ~0."""
        norm = RMSNormGated(64)
        x = torch.randn(2, 8, 64)
        gate = torch.zeros(2, 8, 64)
        out = norm(x, gate)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_preserves_dtype(self) -> None:
        norm = RMSNormGated(64)
        x = torch.randn(2, 8, 64, dtype=torch.float32)
        gate = torch.randn(2, 8, 64, dtype=torch.float32)
        out = norm(x, gate)
        assert out.dtype == torch.float32


class TestNemotronHMamba2Mixer:
    @pytest.fixture
    def small_mixer(self) -> NemotronHMamba2Mixer:
        return NemotronHMamba2Mixer(
            model_dim=128,
            num_heads=4,
            head_dim=32,
            state_size=16,
            n_groups=2,
            conv_kernel=4,
            chunk_size=32,
        )

    def test_output_shape(self, small_mixer: NemotronHMamba2Mixer) -> None:
        x = torch.randn(2, 16, 128)
        out = small_mixer(x)
        assert out.shape == (2, 16, 128)

    def test_single_token(self, small_mixer: NemotronHMamba2Mixer) -> None:
        x = torch.randn(2, 1, 128)
        out = small_mixer(x)
        assert out.shape == (2, 1, 128)

    def test_long_sequence(self, small_mixer: NemotronHMamba2Mixer) -> None:
        x = torch.randn(1, 256, 128)
        out = small_mixer(x)
        assert out.shape == (1, 256, 128)

    def test_parameter_counts(self, small_mixer: NemotronHMamba2Mixer) -> None:
        """Verify the expected parameter shapes."""
        assert small_mixer.in_proj.weight.shape == (
            small_mixer.projection_size,
            128,
        )
        assert small_mixer.conv1d.weight.shape == (
            small_mixer.conv_dim,
            1,
            4,
        )
        assert small_mixer.A_log.shape == (4,)
        assert small_mixer.D.shape == (4,)
        assert small_mixer.dt_bias.shape == (4,)
        assert small_mixer.out_proj.weight.shape == (128, small_mixer.intermediate_size)

    def test_a_log_positive(self, small_mixer: NemotronHMamba2Mixer) -> None:
        """A_log should be initialized as log of positive values."""
        A = torch.exp(small_mixer.A_log)
        assert (A > 0).all()

    def test_incremental_state_creation(
        self, small_mixer: NemotronHMamba2Mixer
    ) -> None:
        """Test that incremental state is properly created."""
        x = torch.randn(2, 8, 128)
        state_bag = IncrementalStateBag(max_num_steps=100)

        with torch.no_grad():
            small_mixer(x, state_bag=state_bag)

        state = state_bag.maybe_get_state(small_mixer, NemotronHMamba2State)
        assert state is not None
        assert state.conv_state.shape == (2, small_mixer.conv_dim, 4)
        assert state.ssm_state.shape == (2, 4, 32, 16)

    def test_deterministic(self, small_mixer: NemotronHMamba2Mixer) -> None:
        """Same input should give same output."""
        x = torch.randn(2, 8, 128)
        with torch.no_grad():
            out1 = small_mixer(x)
            out2 = small_mixer(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_gradient_flow(self, small_mixer: NemotronHMamba2Mixer) -> None:
        """Verify gradients flow through the module."""
        x = torch.randn(2, 8, 128, requires_grad=True)
        out = small_mixer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 8, 128)

    def test_derived_dimensions(self) -> None:
        """Test that derived dimensions match spec for default config."""
        mixer = NemotronHMamba2Mixer(
            model_dim=2688,
            num_heads=64,
            head_dim=64,
            state_size=128,
            n_groups=8,
        )
        assert mixer.intermediate_size == 4096
        assert mixer.conv_dim == 6144  # 4096 + 2*8*128
        assert mixer.projection_size == 10304  # 4096 + 6144 + 64
