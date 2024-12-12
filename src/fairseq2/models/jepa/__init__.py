# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.jepa.factory import JEPA_FAMILY as JEPA_FAMILY
from fairseq2.models.jepa.factory import JepaBuilder as JepaBuilder
from fairseq2.models.jepa.factory import JepaConfig as JepaConfig
from fairseq2.models.jepa.factory import JepaEncoderBuilder as JepaEncoderBuilder
from fairseq2.models.jepa.factory import JepaEncoderConfig as JepaEncoderConfig
from fairseq2.models.jepa.factory import create_jepa_model as create_jepa_model
from fairseq2.models.jepa.factory import jepa_arch as jepa_arch
from fairseq2.models.jepa.factory import jepa_archs as jepa_archs
from fairseq2.models.jepa.loader import load_jepa_config as load_jepa_config
from fairseq2.models.jepa.loader import load_jepa_model as load_jepa_model

# isort: split

import fairseq2.models.jepa.archs  # Register architectures
