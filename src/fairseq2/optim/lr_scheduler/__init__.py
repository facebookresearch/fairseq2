# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.optim.lr_scheduler.base import AbstractLRScheduler as AbstractLRScheduler
from fairseq2.optim.lr_scheduler.base import LRScheduler as LRScheduler
from fairseq2.optim.lr_scheduler.base import NoopLR as NoopLR
from fairseq2.optim.lr_scheduler.base import get_effective_lr as get_effective_lr
from fairseq2.optim.lr_scheduler.cosine import CosineAnnealingLR as CosineAnnealingLR
from fairseq2.optim.lr_scheduler.factory import (
    CosineAnnealingLRConfig as CosineAnnealingLRConfig,
)
from fairseq2.optim.lr_scheduler.factory import MyleLRConfig as MyleLRConfig
from fairseq2.optim.lr_scheduler.factory import NoamLRConfig as NoamLRConfig
from fairseq2.optim.lr_scheduler.factory import (
    PolynomialDecayLRConfig as PolynomialDecayLRConfig,
)
from fairseq2.optim.lr_scheduler.factory import TriStageLRConfig as TriStageLRConfig
from fairseq2.optim.lr_scheduler.factory import (
    create_cosine_annealing_lr as create_cosine_annealing_lr,
)
from fairseq2.optim.lr_scheduler.factory import (
    create_lr_scheduler as create_lr_scheduler,
)
from fairseq2.optim.lr_scheduler.factory import create_myle_lr as create_myle_lr
from fairseq2.optim.lr_scheduler.factory import create_noam_lr as create_noam_lr
from fairseq2.optim.lr_scheduler.factory import (
    create_polynomial_decay_lr as create_polynomial_decay_lr,
)
from fairseq2.optim.lr_scheduler.factory import (
    create_tri_stage_lr as create_tri_stage_lr,
)
from fairseq2.optim.lr_scheduler.factory import (
    lr_scheduler_factories as lr_scheduler_factories,
)
from fairseq2.optim.lr_scheduler.factory import (
    lr_scheduler_factory as lr_scheduler_factory,
)
from fairseq2.optim.lr_scheduler.myle import MyleLR as MyleLR
from fairseq2.optim.lr_scheduler.noam import NoamLR as NoamLR
from fairseq2.optim.lr_scheduler.polynomial import (
    PolynomialDecayLR as PolynomialDecayLR,
)
from fairseq2.optim.lr_scheduler.tri_stage import TriStageLR as TriStageLR
