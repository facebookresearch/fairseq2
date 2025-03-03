# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.nn.data_parallel._common import FsdpApplier as FsdpApplier
from fairseq2.nn.data_parallel._common import FsdpGranularity as FsdpGranularity
from fairseq2.nn.data_parallel._common import (
    FsdpParameterInitializer as FsdpParameterInitializer,
)
from fairseq2.nn.data_parallel._common import FsdpWrapper as FsdpWrapper
from fairseq2.nn.data_parallel._common import load_with_sdp_gang as load_with_sdp_gang
from fairseq2.nn.data_parallel._ddp import DdpModule as DdpModule
from fairseq2.nn.data_parallel._ddp import to_ddp as to_ddp
from fairseq2.nn.data_parallel._error import (
    DistributedSetupError as DistributedSetupError,
)
from fairseq2.nn.data_parallel._fsdp import Fsdp1Module as Fsdp1Module
from fairseq2.nn.data_parallel._fsdp import Fsdp2Module as Fsdp2Module
from fairseq2.nn.data_parallel._fsdp import (
    fsdp1_local_state_dict as fsdp1_local_state_dict,
)
from fairseq2.nn.data_parallel._fsdp import (
    fsdp1_summon_full_parameters as fsdp1_summon_full_parameters,
)
from fairseq2.nn.data_parallel._fsdp import (
    fsdp2_local_state_dict as fsdp2_local_state_dict,
)
from fairseq2.nn.data_parallel._fsdp import fsdp2_no_sync as fsdp2_no_sync
from fairseq2.nn.data_parallel._fsdp import (
    fsdp2_summon_full_parameters as fsdp2_summon_full_parameters,
)
from fairseq2.nn.data_parallel._fsdp import (
    fsdp_local_state_dict as fsdp_local_state_dict,
)
from fairseq2.nn.data_parallel._fsdp import fsdp_no_sync as fsdp_no_sync
from fairseq2.nn.data_parallel._fsdp import (
    fsdp_summon_full_parameters as fsdp_summon_full_parameters,
)
from fairseq2.nn.data_parallel._fsdp import to_fsdp as to_fsdp
from fairseq2.nn.data_parallel._fsdp import to_fsdp1 as to_fsdp1
from fairseq2.nn.data_parallel._fsdp import to_fsdp2 as to_fsdp2
