# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.models import register_model_family
from fairseq2.models.jepa.classifier._config import (
    JEPA_CLASSIFIER_MODEL_FAMILY,
    JepaClassifierConfig,
    register_jepa_classifier_configs,
)
from fairseq2.models.jepa.classifier._factory import JepaClassifierFactory
from fairseq2.models.jepa.classifier._model import JepaClassifierModel


def register_jepa_classifier_family(context: RuntimeContext) -> None:
    default_arch = "base"

    register_model_family(
        context,
        JEPA_CLASSIFIER_MODEL_FAMILY,
        JepaClassifierModel,
        JepaClassifierConfig,
        default_arch,
        create_jepa_classifier_model,
    )

    register_jepa_classifier_configs(context)


def create_jepa_classifier_model(config: JepaClassifierConfig) -> JepaClassifierModel:
    return JepaClassifierFactory(config).create_model()
