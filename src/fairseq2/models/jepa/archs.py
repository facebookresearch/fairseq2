# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.jepa.factory import JepaConfig, jepa_arch


@jepa_arch("tiny")
def tiny() -> JepaConfig:
    config = base()

    config.encoder_config.model_dim = 192
    config.encoder_config.num_encoder_attn_heads = 3

    return config


@jepa_arch("small")
def small() -> JepaConfig:
    config = base()

    config.encoder_config.model_dim = 384
    config.encoder_config.num_encoder_attn_heads = 6

    return config


@jepa_arch("base")
def base() -> JepaConfig:
    return JepaConfig()


@jepa_arch("large")
def large() -> JepaConfig:
    config = base()

    config.encoder_config.model_dim = 1024
    config.encoder_config.num_encoder_layers = 24
    config.encoder_config.num_encoder_attn_heads = 16

    return config


@jepa_arch("huge")
def huge() -> JepaConfig:
    config = base()

    config.encoder_config.model_dim = 1280
    config.encoder_config.num_encoder_layers = 32
    config.encoder_config.num_encoder_attn_heads = 16

    return config


@jepa_arch("giant")
def giant() -> JepaConfig:
    config = base()

    config.encoder_config.model_dim = 1408
    config.encoder_config.num_encoder_layers = 40
    config.encoder_config.num_encoder_attn_heads = 16
    config.encoder_config.ffn_inner_dim_ratio = 48 / 11

    return config


@jepa_arch("gigantic")
def gigantic() -> JepaConfig:
    config = base()

    config.encoder_config.model_dim = 1664
    config.encoder_config.num_encoder_layers = 48
    config.encoder_config.num_encoder_attn_heads = 16
    config.encoder_config.ffn_inner_dim_ratio = 64 / 13

    return config
