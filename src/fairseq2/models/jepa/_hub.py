# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import ModelHubAccessor

# isort: split

from fairseq2.models.jepa._config import JepaConfig
from fairseq2.models.jepa._model import JepaModel

get_jepa_model_hub = ModelHubAccessor(JepaModel, JepaConfig)
