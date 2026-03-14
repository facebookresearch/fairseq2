# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq2.models.gemma3n.config import Gemma3nAudioConfig, get_gemma3n_e2b_config
from fairseq2.models.gemma3n.audio.embedder import Gemma3nMultimodalEmbedder


class TestGemma3nMultimodalEmbedder:
    def test_soft_embedding_shape(self) -> None:
        """Test that soft embeddings are projected correctly."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        embedder = Gemma3nMultimodalEmbedder(audio_config, text_config)

        batch_size = 2
        seq_len = 24
        soft_features = torch.randn(batch_size, seq_len, audio_config.hidden_size)

        output = embedder(soft_features, is_soft=True)

        assert output.shape == (batch_size, seq_len, text_config.model_dim)

    def test_hard_embedding_shape(self) -> None:
        """Test that hard token IDs are embedded and projected correctly."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        embedder = Gemma3nMultimodalEmbedder(audio_config, text_config)

        batch_size = 2
        seq_len = 24
        token_ids = torch.randint(0, audio_config.vocab_size, (batch_size, seq_len))

        output = embedder(token_ids, is_soft=False)

        assert output.shape == (batch_size, seq_len, text_config.model_dim)

    def test_projection_dimensions(self) -> None:
        """Test that projection is from audio hidden_size to text model_dim."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        embedder = Gemma3nMultimodalEmbedder(audio_config, text_config)

        assert embedder.embedding_projection.weight.shape == (
            text_config.model_dim,
            audio_config.hidden_size,
        )

    def test_vocab_size(self) -> None:
        """Test that embedding table has correct vocab size."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        embedder = Gemma3nMultimodalEmbedder(audio_config, text_config)

        assert embedder.embedding.weight.shape == (
            audio_config.vocab_size,
            audio_config.hidden_size,
        )
        assert embedder.embedding.weight.shape[0] == 128

    def test_different_norms_for_hard_soft(self) -> None:
        """Test that hard and soft embeddings use different norms."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        embedder = Gemma3nMultimodalEmbedder(audio_config, text_config)

        assert embedder.hard_embedding_norm is not embedder.soft_embedding_norm

        batch_size = 1
        seq_len = 4

        soft_features = torch.randn(batch_size, seq_len, audio_config.hidden_size)
        token_ids = torch.randint(0, audio_config.vocab_size, (batch_size, seq_len))

        output_soft = embedder(soft_features, is_soft=True)
        output_hard = embedder(token_ids, is_soft=False)

        assert output_soft.shape == output_hard.shape
        assert not torch.allclose(output_soft, output_hard)
