# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.jepa.config import JEPA_FAMILY as JEPA_FAMILY
from fairseq2.models.jepa.config import JepaConfig as JepaConfig
from fairseq2.models.jepa.config import JepaEncoderConfig as JepaEncoderConfig
from fairseq2.models.jepa.config import register_jepa_configs as register_jepa_configs
from fairseq2.models.jepa.factory import JepaEncoderFactory as JepaEncoderFactory
from fairseq2.models.jepa.factory import JepaFactory as JepaFactory
from fairseq2.models.jepa.factory import create_jepa_model as create_jepa_model
from fairseq2.models.jepa.hub import get_jepa_model_hub as get_jepa_model_hub
from fairseq2.models.jepa.interop import (
    convert_jepa_state_dict as convert_jepa_state_dict,
)
from fairseq2.models.jepa.model import JepaModel as JepaModel
