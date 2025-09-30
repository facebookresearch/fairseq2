# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.optim.lr_schedulers.cosine_annealing import (
    CosineAnnealingLR as CosineAnnealingLR,
)
from fairseq2.optim.lr_schedulers.lr_scheduler import (
    AbstractLRScheduler as AbstractLRScheduler,
)
from fairseq2.optim.lr_schedulers.lr_scheduler import LRScheduler as LRScheduler
from fairseq2.optim.lr_schedulers.lr_scheduler import PassthroughLR as PassthroughLR
from fairseq2.optim.lr_schedulers.lr_scheduler import (
    get_effective_lr as get_effective_lr,
)
from fairseq2.optim.lr_schedulers.myle import MyleLR as MyleLR
from fairseq2.optim.lr_schedulers.noam import NoamLR as NoamLR
from fairseq2.optim.lr_schedulers.polynomial_decay import (
    PolynomialDecayLR as PolynomialDecayLR,
)
from fairseq2.optim.lr_schedulers.tri_stage import TriStageLR as TriStageLR
