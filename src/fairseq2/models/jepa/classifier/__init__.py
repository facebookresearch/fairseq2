# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.jepa.classifier._config import (
    JEPA_CLASSIFIER_MODEL_FAMILY as JEPA_CLASSIFIER_MODEL_FAMILY,
)
from fairseq2.models.jepa.classifier._config import (
    JepaClassifierConfig as JepaClassifierConfig,
)
from fairseq2.models.jepa.classifier._config import (
    register_jepa_classifier_configs as register_jepa_classifier_configs,
)
from fairseq2.models.jepa.classifier._factory import (
    JepaClassifierFactory as JepaClassifierFactory,
)
from fairseq2.models.jepa.classifier._handler import (
    JepaClassifierModelHandler as JepaClassifierModelHandler,
)
from fairseq2.models.jepa.classifier._model import AttentivePooler as AttentivePooler
from fairseq2.models.jepa.classifier._model import (
    CrossAttentionDecoderLayer as CrossAttentionDecoderLayer,
)
from fairseq2.models.jepa.classifier._model import (
    JepaClassifierModel as JepaClassifierModel,
)

# isort: split

from fairseq2.models import ModelHubAccessor

get_jepa_classifier_model_hub = ModelHubAccessor(
    JepaClassifierModel, JepaClassifierConfig
)
