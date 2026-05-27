# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.nemotron.config import NEMOTRON_H_FAMILY as NEMOTRON_H_FAMILY
from fairseq2.models.nemotron.config import NemotronHConfig as NemotronHConfig
from fairseq2.models.nemotron.config import (
    register_nemotron_h_configs as register_nemotron_h_configs,
)
from fairseq2.models.nemotron.decoder_layer import NemotronHBlock as NemotronHBlock
from fairseq2.models.nemotron.factory import NemotronHFactory as NemotronHFactory
from fairseq2.models.nemotron.factory import (
    create_nemotron_h_model as create_nemotron_h_model,
)
from fairseq2.models.nemotron.hub import (
    get_nemotron_h_model_hub as get_nemotron_h_model_hub,
)
from fairseq2.models.nemotron.interop import (
    _NemotronHHuggingFaceConverter as _NemotronHHuggingFaceConverter,
)
from fairseq2.models.nemotron.interop import (
    convert_nemotron_h_state_dict as convert_nemotron_h_state_dict,
)
from fairseq2.models.nemotron.mamba2 import (
    NemotronHMamba2Mixer as NemotronHMamba2Mixer,
)
from fairseq2.models.nemotron.moe import NemotronHMoE as NemotronHMoE
from fairseq2.models.nemotron.sharder import (
    NemotronHMoESharder as NemotronHMoESharder,
)
from fairseq2.models.nemotron.sharder import (
    get_nemotron_h_shard_specs as get_nemotron_h_shard_specs,
)
