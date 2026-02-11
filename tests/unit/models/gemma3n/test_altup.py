# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from fairseq2.models.gemma3n.altup import Gemma3nAltUp
from fairseq2.nn import RMSNorm
from tests.common import device


class TestGemma3nAltUp:
    """Test AltUp predict/correct mechanism."""

    def test_predict_input_shape_4d(self) -> None:
        """Verify predict accepts 4D input [num_inputs, B, S, D]."""
        model_dim, num_inputs = 64, 4
        batch_size, seq_len = 2, 8

        router_norm = RMSNorm(model_dim, bias=False, device=device)
        altup = Gemma3nAltUp(
            model_dim=model_dim,
            num_inputs=num_inputs,
            router_norm=router_norm,
            device=device,
        )

        # Input is 4D: [num_inputs, batch, seq_len, model_dim]
        hidden_states = torch.randn(
            num_inputs, batch_size, seq_len, model_dim, device=device
        )

        with torch.no_grad():
            predictions = altup(hidden_states)

        # Output should be same shape
        assert predictions.shape == (num_inputs, batch_size, seq_len, model_dim)

    def test_predict_output_shape_preserved(self) -> None:
        """Verify predict preserves 4D shape."""
        model_dim, num_inputs = 32, 4
        batch_size, seq_len = 1, 4

        router_norm = RMSNorm(model_dim, bias=False, device=device)
        altup = Gemma3nAltUp(
            model_dim=model_dim,
            num_inputs=num_inputs,
            router_norm=router_norm,
            device=device,
        )

        hidden_states = torch.randn(
            num_inputs, batch_size, seq_len, model_dim, device=device
        )

        with torch.no_grad():
            predictions = altup(hidden_states)

        assert predictions.shape == hidden_states.shape
        assert predictions.dtype == hidden_states.dtype

    def test_correct_combines_predictions_and_activated(self) -> None:
        """Verify correct combines 4D predictions with 3D activated output."""
        model_dim, num_inputs = 64, 4
        batch_size, seq_len = 2, 8

        router_norm = RMSNorm(model_dim, bias=False, device=device)
        altup = Gemma3nAltUp(
            model_dim=model_dim,
            num_inputs=num_inputs,
            active_idx=0,
            router_norm=router_norm,
            device=device,
        )

        # Predictions are 4D
        predictions = torch.randn(
            num_inputs, batch_size, seq_len, model_dim, device=device
        )

        # Activated output is 3D (from layer forward pass)
        activated = torch.randn(batch_size, seq_len, model_dim, device=device)

        with torch.no_grad():
            corrected = altup.correct(predictions, activated)

        # Output should be 4D
        assert corrected.shape == (num_inputs, batch_size, seq_len, model_dim)

    def test_correct_output_shape_4d(self) -> None:
        """Verify correct returns 4D output."""
        model_dim, num_inputs = 32, 4
        batch_size, seq_len = 1, 4

        router_norm = RMSNorm(model_dim, bias=False, device=device)
        altup = Gemma3nAltUp(
            model_dim=model_dim,
            num_inputs=num_inputs,
            router_norm=router_norm,
            device=device,
        )

        predictions = torch.randn(
            num_inputs, batch_size, seq_len, model_dim, device=device
        )
        activated = torch.randn(batch_size, seq_len, model_dim, device=device)

        with torch.no_grad():
            corrected = altup.correct(predictions, activated)

        assert corrected.shape == (num_inputs, batch_size, seq_len, model_dim)
        assert corrected.dtype == activated.dtype

    def test_coef_clip_parameter(self) -> None:
        """Verify coef_clip parameter is stored correctly."""
        model_dim, num_inputs = 64, 4
        coef_clip = 120.0

        router_norm = RMSNorm(model_dim, bias=False, device=device)
        altup = Gemma3nAltUp(
            model_dim=model_dim,
            num_inputs=num_inputs,
            router_norm=router_norm,
            coef_clip=coef_clip,
            device=device,
        )

        assert altup.coef_clip == coef_clip

    def test_active_idx_parameter(self) -> None:
        """Verify active_idx parameter controls which version is computed."""
        model_dim, num_inputs = 64, 4
        active_idx = 0

        router_norm = RMSNorm(model_dim, bias=False, device=device)
        altup = Gemma3nAltUp(
            model_dim=model_dim,
            num_inputs=num_inputs,
            active_idx=active_idx,
            router_norm=router_norm,
            device=device,
        )

        assert altup.active_idx == active_idx

    def test_predict_correct_cycle(self) -> None:
        """Verify full predict→layer→correct cycle."""
        model_dim, num_inputs = 64, 4
        batch_size, seq_len = 2, 8

        router_norm = RMSNorm(model_dim, bias=False, device=device)
        altup = Gemma3nAltUp(
            model_dim=model_dim,
            num_inputs=num_inputs,
            active_idx=0,
            router_norm=router_norm,
            device=device,
        )

        # 1. Start with 4D hidden states
        hidden_states = torch.randn(
            num_inputs, batch_size, seq_len, model_dim, device=device
        )

        with torch.no_grad():
            # 2. Predict generates 4D predictions
            predictions = altup(hidden_states)
            assert predictions.shape == (num_inputs, batch_size, seq_len, model_dim)

            # 3. Simulate layer processing active version (extract 3D)
            active_version = predictions[altup.active_idx]
            assert active_version.shape == (batch_size, seq_len, model_dim)

            # 4. Simulate layer forward (just identity for test)
            activated = active_version + torch.randn_like(active_version) * 0.1

            # 5. Correct updates all 4 versions
            corrected = altup.correct(predictions, activated)
            assert corrected.shape == (num_inputs, batch_size, seq_len, model_dim)

    def test_scale_corrected_output(self) -> None:
        """Verify scale_corrected_output applies learned scaling."""
        model_dim, num_inputs = 64, 4
        batch_size, seq_len = 2, 8

        router_norm = RMSNorm(model_dim, bias=False, device=device)
        altup = Gemma3nAltUp(
            model_dim=model_dim,
            num_inputs=num_inputs,
            router_norm=router_norm,
            device=device,
        )

        # 3D corrected output (after averaging or selecting)
        corrected = torch.randn(batch_size, seq_len, model_dim, device=device)

        with torch.no_grad():
            scaled = altup.scale_corrected_output(corrected)

        assert scaled.shape == corrected.shape
        assert scaled.dtype == corrected.dtype

    def test_router_modalities_computation(self) -> None:
        """Verify router modalities computed correctly."""
        model_dim, num_inputs = 64, 4
        batch_size, seq_len = 2, 8

        router_norm = RMSNorm(model_dim, bias=False, device=device)
        altup = Gemma3nAltUp(
            model_dim=model_dim,
            num_inputs=num_inputs,
            router_norm=router_norm,
            device=device,
        )

        x = torch.randn(batch_size, seq_len, model_dim, device=device)

        with torch.no_grad():
            modalities = altup.compute_router_modalities(x)

        # Modalities should be [batch, seq_len, num_inputs]
        assert modalities.shape == (batch_size, seq_len, num_inputs)
        # Should be tanh-bounded [-1, 1]
        assert modalities.min() >= -1.0
        assert modalities.max() <= 1.0

    def test_num_inputs_parameter(self) -> None:
        """Verify num_inputs parameter is typically 4."""
        model_dim = 64

        for num_inputs in [2, 4, 8]:
            router_norm = RMSNorm(model_dim, bias=False, device=device)
            altup = Gemma3nAltUp(
                model_dim=model_dim,
                num_inputs=num_inputs,
                router_norm=router_norm,
                device=device,
            )

            assert altup.num_inputs == num_inputs

            # Verify forward works
            batch_size, seq_len = 1, 4
            hidden_states = torch.randn(
                num_inputs, batch_size, seq_len, model_dim, device=device
            )

            with torch.no_grad():
                predictions = altup(hidden_states)

            assert predictions.shape[0] == num_inputs
