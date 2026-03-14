# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq2.models.gemma3n.audio.tower import Gemma3nAudioTower
from fairseq2.models.gemma3n.config import Gemma3nAudioConfig, get_gemma3n_e2b_config


class TestGemma3nAudioTower:
    def test_end_to_end_shape(self) -> None:
        """Test that full audio tower produces correct output shape."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        tower = Gemma3nAudioTower(audio_config, text_config)

        batch_size = 2
        time_steps = 100
        mel_features = torch.randn(batch_size, time_steps, audio_config.input_feat_size)

        output = tower(mel_features)

        # Output should have text model dimension and reduced time
        assert output.ndim == 3
        assert output.size(0) == batch_size
        assert output.size(2) == text_config.model_dim

    def test_temporal_reduction(self) -> None:
        """Test that temporal downsampling is consistent across inputs."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        tower = Gemma3nAudioTower(audio_config, text_config)

        batch_size = 1
        # Larger inputs should produce proportionally larger outputs
        prev_time = 0
        for time_steps in [64, 128, 256, 512]:
            mel_features = torch.randn(batch_size, time_steps, audio_config.input_feat_size)
            output = tower(mel_features)

            cur_time = output.size(1)
            assert cur_time > prev_time, \
                f"Larger input ({time_steps}) should produce larger output"
            prev_time = cur_time

    def test_projection_to_text_space(self) -> None:
        """Test that audio features are projected to text model dimension."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        tower = Gemma3nAudioTower(audio_config, text_config)

        batch_size = 2
        time_steps = 100
        mel_features = torch.randn(batch_size, time_steps, audio_config.input_feat_size)

        output = tower(mel_features)

        assert output.size(-1) == text_config.model_dim
        assert output.size(-1) == 2048

    def test_different_batch_sizes(self) -> None:
        """Test that tower handles different batch sizes correctly."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        tower = Gemma3nAudioTower(audio_config, text_config)

        time_steps = 100

        for batch_size in [1, 2, 4]:
            mel_features = torch.randn(batch_size, time_steps, audio_config.input_feat_size)
            output = tower(mel_features)

            assert output.size(0) == batch_size
            assert output.size(2) == text_config.model_dim

    def test_has_all_components(self) -> None:
        """Test that tower contains all expected components."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        tower = Gemma3nAudioTower(audio_config, text_config)

        assert hasattr(tower, "subsample")
        assert hasattr(tower, "encoder")
        assert hasattr(tower, "embedder")

        assert len(tower.encoder.layers) == 12

        embedder_params = list(tower.embedder.named_parameters())
        assert any("embedding.weight" in name for name, _ in embedder_params)
        assert any("embedding_projection.weight" in name for name, _ in embedder_params)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through entire tower."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        tower = Gemma3nAudioTower(audio_config, text_config)

        batch_size = 2
        time_steps = 100
        mel_features = torch.randn(batch_size, time_steps, audio_config.input_feat_size)

        output = tower(mel_features)

        loss = output.sum()
        loss.backward()

        subsample_has_grad = any(p.grad is not None for p in tower.subsample.parameters())
        encoder_has_grad = any(p.grad is not None for p in tower.encoder.parameters())
        embedder_has_grad = any(p.grad is not None for p in tower.embedder.parameters())

        assert subsample_has_grad, "Subsample should have gradients"
        assert encoder_has_grad, "Encoder should have gradients"
        assert embedder_has_grad, "Embedder should have gradients"
