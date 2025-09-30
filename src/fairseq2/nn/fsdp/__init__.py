# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.nn.fsdp.common import FSDPApplier as FSDPApplier
from fairseq2.nn.fsdp.common import FSDPParameterInitializer as FSDPParameterInitializer
from fairseq2.nn.fsdp.common import FSDPWrapper as FSDPWrapper
from fairseq2.nn.fsdp.common import load_with_sdp_gang as load_with_sdp_gang
from fairseq2.nn.fsdp.fsdp1 import FSDP1Module as FSDP1Module
from fairseq2.nn.fsdp.fsdp1 import (
    fsdp1_load_local_state_dict as fsdp1_load_local_state_dict,
)
from fairseq2.nn.fsdp.fsdp1 import fsdp1_local_state_dict as fsdp1_local_state_dict
from fairseq2.nn.fsdp.fsdp1 import (
    fsdp1_summon_full_parameters as fsdp1_summon_full_parameters,
)
from fairseq2.nn.fsdp.fsdp1 import to_fsdp1 as to_fsdp1
from fairseq2.nn.fsdp.fsdp2 import FSDP2Module as FSDP2Module
from fairseq2.nn.fsdp.fsdp2 import (
    fsdp2_load_local_state_dict as fsdp2_load_local_state_dict,
)
from fairseq2.nn.fsdp.fsdp2 import fsdp2_local_state_dict as fsdp2_local_state_dict
from fairseq2.nn.fsdp.fsdp2 import fsdp2_no_sync as fsdp2_no_sync
from fairseq2.nn.fsdp.fsdp2 import (
    fsdp2_summon_full_parameters as fsdp2_summon_full_parameters,
)
from fairseq2.nn.fsdp.fsdp2 import to_fsdp2 as to_fsdp2
from fairseq2.nn.fsdp.unified import (
    fsdp_load_local_state_dict as fsdp_load_local_state_dict,
)
from fairseq2.nn.fsdp.unified import fsdp_local_state_dict as fsdp_local_state_dict
from fairseq2.nn.fsdp.unified import fsdp_no_sync as fsdp_no_sync
from fairseq2.nn.fsdp.unified import (
    fsdp_summon_full_parameters as fsdp_summon_full_parameters,
)
from fairseq2.nn.fsdp.unified import to_fsdp as to_fsdp
