# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.jepa.archs import base as jepa_base
from fairseq2.models.jepa.archs import huge as jepa_huge
from fairseq2.models.jepa.archs import large as jepa_large
from fairseq2.models.jepa.classifier.factory import (
    JepaClassifierConfig,
    jepa_classifier_arch,
)


@jepa_classifier_arch("base")
def base() -> JepaClassifierConfig:
    pretrain_config = jepa_base()
    return JepaClassifierConfig(encoder_config=pretrain_config.encoder_config)


@jepa_classifier_arch("large")
def large() -> JepaClassifierConfig:
    pretrain_config = jepa_large()
    return JepaClassifierConfig(encoder_config=pretrain_config.encoder_config)


@jepa_classifier_arch("huge")
def huge() -> JepaClassifierConfig:
    pretrain_config = jepa_huge()
    return JepaClassifierConfig(encoder_config=pretrain_config.encoder_config)
