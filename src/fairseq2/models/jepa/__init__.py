# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.jepa._config import JEPA_MODEL_FAMILY as JEPA_MODEL_FAMILY
from fairseq2.models.jepa._config import JepaConfig as JepaConfig
from fairseq2.models.jepa._config import JepaEncoderConfig as JepaEncoderConfig
from fairseq2.models.jepa._config import register_jepa_configs as register_jepa_configs
from fairseq2.models.jepa._factory import JepaEncoderFactory as JepaEncoderFactory
from fairseq2.models.jepa._factory import JepaFactory as JepaFactory
from fairseq2.models.jepa._handler import JepaModelHandler as JepaModelHandler
from fairseq2.models.jepa._handler import (
    convert_jepa_checkpoint as convert_jepa_checkpoint,
)
from fairseq2.models.jepa._model import JepaModel as JepaModel

# isort: split

from fairseq2.models import ModelHubAccessor

get_jepa_model_hub = ModelHubAccessor(JepaModel, JepaConfig)
