# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.jepa.factory import JEPA_FAMILY as JEPA_FAMILY
from fairseq2.models.jepa.factory import JepaConfig as JepaConfig
from fairseq2.models.jepa.factory import JepaEncoderConfig as JepaEncoderConfig
from fairseq2.models.jepa.factory import jepa_arch as jepa_arch
from fairseq2.models.jepa.factory import jepa_archs as jepa_archs

# isort: split

import fairseq2.models.jepa.archs  # Register architectures
