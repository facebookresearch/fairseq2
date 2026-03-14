# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq2.models.gemma3n.config import Gemma3nAudioConfig
from fairseq2.models.gemma3n.audio.conformer import Gemma3nConformerEncoder
from fairseq2.nn import BatchLayout


class TestGemma3nConformerEncoder:
    def test_output_shape(self) -> None:
        """Test that conformer encoder produces correct output shape."""
        config = Gemma3nAudioConfig()

        encoder = Gemma3nConformerEncoder(config)

        batch_size = 2
        seq_len = 24
        features = torch.randn(batch_size, seq_len, config.hidden_size)

        layout = BatchLayout((batch_size, seq_len), seq_lens=[seq_len] * batch_size)

        output = encoder(features, layout)

        # Conformer applies reduction_factor striding
        expected_len = seq_len // config.conf_reduction_factor
        assert output.shape == (batch_size, expected_len, config.hidden_size)

    def test_layer_count(self) -> None:
        """Test that encoder has correct number of layers."""
        config = Gemma3nAudioConfig()

        encoder = Gemma3nConformerEncoder(config)

        assert len(encoder.layers) == config.conf_num_hidden_layers
        assert len(encoder.layers) == 12

    def test_parameter_count(self) -> None:
        """Test that encoder has expected parameter structure."""
        config = Gemma3nAudioConfig()

        encoder = Gemma3nConformerEncoder(config)

        total_params = sum(p.numel() for p in encoder.parameters())

        assert total_params > 0, "Encoder should have parameters"

        layer_params = list(encoder.layers[0].named_parameters())
        assert len(layer_params) > 0, "Each layer should have parameters"
