# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

from torch.nn import Module
from typing_extensions import override

from fairseq2.models import AbstractModelHandler
from fairseq2.models.jepa.classifier._config import (
    JEPA_CLASSIFIER_MODEL_FAMILY,
    JepaClassifierConfig,
)
from fairseq2.models.jepa.classifier._factory import JepaClassifierFactory
from fairseq2.models.jepa.classifier._model import JepaClassifierModel


class JepaClassifierModelHandler(AbstractModelHandler):
    @property
    @override
    def family(self) -> str:
        return JEPA_CLASSIFIER_MODEL_FAMILY

    @property
    @override
    def kls(self) -> type[Module]:
        return JepaClassifierModel

    @override
    def _create_model(self, config: object) -> Module:
        config = cast(JepaClassifierConfig, config)

        return JepaClassifierFactory(config).create_model()
