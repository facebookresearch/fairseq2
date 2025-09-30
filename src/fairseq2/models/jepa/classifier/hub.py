# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import ModelHubAccessor
from fairseq2.models.jepa.classifier.config import (
    JEPA_CLASSIFIER_FAMILY,
    JepaClassifierConfig,
)
from fairseq2.models.jepa.classifier.model import JepaClassifierModel

get_jepa_classifier_model_hub = ModelHubAccessor(
    JEPA_CLASSIFIER_FAMILY, JepaClassifierModel, JepaClassifierConfig
)
