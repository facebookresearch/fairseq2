# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.optim.lr_scheduler._cosine_annealing import register_cosine_annealing_lr
from fairseq2.optim.lr_scheduler._myle import register_myle_lr
from fairseq2.optim.lr_scheduler._noam import register_noam_lr
from fairseq2.optim.lr_scheduler._polynomial_decay import register_polynomial_decay_lr
from fairseq2.optim.lr_scheduler._tri_stage import register_tri_stage_lr


def register_lr_schedulers(context: RuntimeContext) -> None:
    register_cosine_annealing_lr(context)
    register_myle_lr(context)
    register_noam_lr(context)
    register_polynomial_decay_lr(context)
    register_tri_stage_lr(context)
