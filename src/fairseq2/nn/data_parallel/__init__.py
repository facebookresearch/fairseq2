# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.nn.data_parallel._ddp import to_ddp as to_ddp
from fairseq2.nn.data_parallel._error import (
    DistributedSetupError as DistributedSetupError,
)
from fairseq2.nn.data_parallel._fsdp import (
    FSDP_LOW_MEMORY_POLICY as FSDP_LOW_MEMORY_POLICY,
)
from fairseq2.nn.data_parallel._fsdp import (
    FSDP_STANDARD_MEMORY_POLICY as FSDP_STANDARD_MEMORY_POLICY,
)
from fairseq2.nn.data_parallel._fsdp import (
    FSDP_VERY_LOW_MEMORY_POLICY as FSDP_VERY_LOW_MEMORY_POLICY,
)
from fairseq2.nn.data_parallel._fsdp import FSDPMemoryPolicy as FSDPMemoryPolicy
from fairseq2.nn.data_parallel._fsdp import (
    FSDPParameterInitializer as FSDPParameterInitializer,
)
from fairseq2.nn.data_parallel._fsdp import FSDPWrapPolicy as FSDPWrapPolicy
from fairseq2.nn.data_parallel._fsdp import (
    fsdp_local_state_dict as fsdp_local_state_dict,
)
from fairseq2.nn.data_parallel._fsdp import (
    fsdp_summon_full_parameters as fsdp_summon_full_parameters,
)
from fairseq2.nn.data_parallel._fsdp import to_fsdp as to_fsdp
from fairseq2.nn.data_parallel._sharded import load_with_sdp_gang as load_with_sdp_gang
