# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Module
from typing_extensions import override

from fairseq2.models.handler import AbstractModelHandler
from fairseq2.models.jepa.classifier.config import (
    JEPA_CLASSIFIER_MODEL_FAMILY,
    JepaClassifierConfig,
)
from fairseq2.models.jepa.classifier.factory import JepaClassifierFactory
from fairseq2.models.jepa.classifier.model import JepaClassifierModel
from fairseq2.typing import safe_cast


class JepaClassifierModelHandler(AbstractModelHandler):
    @override
    @property
    def family(self) -> str:
        return JEPA_CLASSIFIER_MODEL_FAMILY

    @override
    @property
    def kls(self) -> type[Module]:
        return JepaClassifierModel

    @override
    def _create_model(self, config: object) -> Module:
        config = safe_cast("config", config, JepaClassifierConfig)

        return JepaClassifierFactory(config).create_model()
