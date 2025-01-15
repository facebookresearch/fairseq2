# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq2.context import RuntimeContext
from fairseq2.models.jepa import JepaConfig, JepaEncoderConfig

JEPA_CLASSIFIER_MODEL_FAMILY = "jepa_classifier"


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


def register_jepa_classifier_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(JepaClassifierConfig)

    arch = registry.decorator

    jepa_registry = context.get_config_registry(JepaConfig)

    @arch("base")
    def base() -> JepaClassifierConfig:
        jepa_config = jepa_registry.get("base")

        return JepaClassifierConfig(encoder_config=jepa_config.encoder_config)

    @arch("large")
    def large() -> JepaClassifierConfig:
        jepa_config = jepa_registry.get("large")

        return JepaClassifierConfig(encoder_config=jepa_config.encoder_config)

    @arch("huge")
    def huge() -> JepaClassifierConfig:
        jepa_config = jepa_registry.get("huge")

        return JepaClassifierConfig(encoder_config=jepa_config.encoder_config)
