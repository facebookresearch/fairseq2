# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.optim.lr_scheduler._cosine_annealing import (
    COSINE_ANNEALING_LR as COSINE_ANNEALING_LR,
)
from fairseq2.optim.lr_scheduler._cosine_annealing import (
    CosineAnnealingLR as CosineAnnealingLR,
)
from fairseq2.optim.lr_scheduler._cosine_annealing import (
    CosineAnnealingLRConfig as CosineAnnealingLRConfig,
)
from fairseq2.optim.lr_scheduler._cosine_annealing import (
    CosineAnnealingLRHandler as CosineAnnealingLRHandler,
)
from fairseq2.optim.lr_scheduler._error import (
    UnknownLRSchedulerError as UnknownLRSchedulerError,
)
from fairseq2.optim.lr_scheduler._error import (
    UnspecifiedNumberOfStepsError as UnspecifiedNumberOfStepsError,
)
from fairseq2.optim.lr_scheduler._handler import (
    LRSchedulerHandler as LRSchedulerHandler,
)
from fairseq2.optim.lr_scheduler._lr_scheduler import (
    AbstractLRScheduler as AbstractLRScheduler,
)
from fairseq2.optim.lr_scheduler._lr_scheduler import LRScheduler as LRScheduler
from fairseq2.optim.lr_scheduler._lr_scheduler import NoopLR as NoopLR
from fairseq2.optim.lr_scheduler._lr_scheduler import (
    get_effective_lr as get_effective_lr,
)
from fairseq2.optim.lr_scheduler._myle import MYLE_LR as MYLE_LR
from fairseq2.optim.lr_scheduler._myle import MyleLR as MyleLR
from fairseq2.optim.lr_scheduler._myle import MyleLRConfig as MyleLRConfig
from fairseq2.optim.lr_scheduler._myle import MyleLRHandler as MyleLRHandler
from fairseq2.optim.lr_scheduler._noam import NOAM_LR as NOAM_LR
from fairseq2.optim.lr_scheduler._noam import NoamLR as NoamLR
from fairseq2.optim.lr_scheduler._noam import NoamLRConfig as NoamLRConfig
from fairseq2.optim.lr_scheduler._noam import NoamLRHandler as NoamLRHandler
from fairseq2.optim.lr_scheduler._polynomial_decay import (
    POLYNOMIAL_DECAY_LR as POLYNOMIAL_DECAY_LR,
)
from fairseq2.optim.lr_scheduler._polynomial_decay import (
    PolynomialDecayLR as PolynomialDecayLR,
)
from fairseq2.optim.lr_scheduler._polynomial_decay import (
    PolynomialDecayLRConfig as PolynomialDecayLRConfig,
)
from fairseq2.optim.lr_scheduler._polynomial_decay import (
    PolynomialDecayLRHandler as PolynomialDecayLRHandler,
)
from fairseq2.optim.lr_scheduler._tri_stage import TRI_STAGE_LR as TRI_STAGE_LR
from fairseq2.optim.lr_scheduler._tri_stage import TriStageLR as TriStageLR
from fairseq2.optim.lr_scheduler._tri_stage import TriStageLRConfig as TriStageLRConfig
from fairseq2.optim.lr_scheduler._tri_stage import (
    TriStageLRHandler as TriStageLRHandler,
)
