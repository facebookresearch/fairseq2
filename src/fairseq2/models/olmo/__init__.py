# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.olmo.attention import (
    OLMOMultiheadAttention as OLMOMultiheadAttention,
)
from fairseq2.models.olmo.config import OLMO_FAMILY as OLMO_FAMILY
from fairseq2.models.olmo.config import OLMOConfig as OLMOConfig
from fairseq2.models.olmo.config import (
    register_olmo_configs as register_olmo_configs,
)
from fairseq2.models.olmo.factory import OLMOFactory as OLMOFactory
from fairseq2.models.olmo.factory import create_olmo_model as create_olmo_model
from fairseq2.models.olmo.hub import get_olmo_model_hub as get_olmo_model_hub
from fairseq2.models.olmo.interop import (
    convert_olmo_state_dict as convert_olmo_state_dict,
)

# Backward compatibility aliases for OLMO2
OLMO2_FAMILY = OLMO_FAMILY
OLMO2Config = OLMOConfig
OLMO2MultiheadAttention = OLMOMultiheadAttention
OLMO2Factory = OLMOFactory
create_olmo2_model = create_olmo_model
register_olmo2_configs = register_olmo_configs
get_olmo2_model_hub = get_olmo_model_hub
convert_olmo2_state_dict = convert_olmo_state_dict
