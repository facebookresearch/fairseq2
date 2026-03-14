# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq2.models.gemma3n.audio import Gemma3nSubsampleConvProjection
from fairseq2.models.gemma3n.config import Gemma3nAudioConfig


class TestSubsampleConvProjection:
    def test_output_shape(self) -> None:
        """Test that subsample projection produces correct output shape."""
        config = Gemma3nAudioConfig()

        module = Gemma3nSubsampleConvProjection(
            input_feat_size=config.input_feat_size,
            hidden_size=config.hidden_size,
            conv_channel_sizes=config.sscp_conv_channel_size,
            conv_kernel_sizes=config.sscp_conv_kernel_size,
            conv_strides=config.sscp_conv_stride_size,
            group_norm_eps=config.sscp_conv_group_norm_eps,
        )

        batch_size = 2
        time_steps = 100
        features = torch.randn(batch_size, time_steps, config.input_feat_size)

        output = module(features)

        expected_time = time_steps // 4
        assert output.shape == (batch_size, expected_time, config.hidden_size), \
            f"Expected shape {(batch_size, expected_time, config.hidden_size)}, got {output.shape}"

    def test_downsampling_factor(self) -> None:
        """Test that 4x downsampling is applied correctly."""
        config = Gemma3nAudioConfig()

        module = Gemma3nSubsampleConvProjection(
            input_feat_size=config.input_feat_size,
            hidden_size=config.hidden_size,
            conv_channel_sizes=config.sscp_conv_channel_size,
            conv_kernel_sizes=config.sscp_conv_kernel_size,
            conv_strides=config.sscp_conv_stride_size,
        )

        batch_size = 1
        for time_steps in [40, 80, 120]:
            features = torch.randn(batch_size, time_steps, config.input_feat_size)
            output = module(features)

            expected_time = time_steps // 4
            assert output.size(1) == expected_time, \
                f"Expected {expected_time} time steps, got {output.size(1)}"

    def test_parameter_count(self) -> None:
        """Test that module has expected number of parameters."""
        config = Gemma3nAudioConfig()

        module = Gemma3nSubsampleConvProjection(
            input_feat_size=config.input_feat_size,
            hidden_size=config.hidden_size,
            conv_channel_sizes=config.sscp_conv_channel_size,
            conv_kernel_sizes=config.sscp_conv_kernel_size,
            conv_strides=config.sscp_conv_stride_size,
        )

        total_params = sum(p.numel() for p in module.parameters())

        assert total_params > 0, "Module should have parameters"

        # conv0 (weight, no bias), norm0 (weight), conv1 (weight, no bias),
        # norm1 (weight), proj (weight, no bias)
        expected_params = 5
        assert len(list(module.named_parameters())) == expected_params, \
            f"Expected {expected_params} parameters, got {len(list(module.named_parameters()))}"
