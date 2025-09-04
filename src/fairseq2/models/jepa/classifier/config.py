# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Final

from fairseq2.models.jepa import JepaConfig, JepaEncoderConfig
from fairseq2.runtime.config_registry import ConfigRegistrar, get_config
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver

JEPA_CLASSIFIER_FAMILY: Final = "jepa_classifier"


@dataclass(kw_only=True)
class JepaClassifierConfig:
    encoder_config: JepaEncoderConfig = field(
        default_factory=lambda: JepaEncoderConfig()
    )
    """The configuration of the vision encoder."""

    pool_depth: int = 1
    """The pool depth."""

    decoder_projection: bool = True
    """If ``True``, the decoder will have an output projection on top."""

    num_queries: int = 1
    """The number of query tokens in the attention pool layer."""

    num_classes: int = 1000
    """The number of classes."""


def register_jepa_classifier_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, JepaClassifierConfig)

    @arch("base", advanced=True)
    def base(resolver: DependencyResolver) -> JepaClassifierConfig:
        jepa_config = get_config(resolver, JepaConfig, "base")

        return JepaClassifierConfig(encoder_config=jepa_config.encoder_config)

    @arch("large", advanced=True)
    def large(resolver: DependencyResolver) -> JepaClassifierConfig:
        jepa_config = get_config(resolver, JepaConfig, "large")

        return JepaClassifierConfig(encoder_config=jepa_config.encoder_config)

    @arch("huge", advanced=True)
    def huge(resolver: DependencyResolver) -> JepaClassifierConfig:
        jepa_config = get_config(resolver, JepaConfig, "huge")

        return JepaClassifierConfig(encoder_config=jepa_config.encoder_config)
