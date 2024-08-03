# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from torch.optim import Optimizer

from fairseq2.factory_registry import ConfigBoundFactoryRegistry
from fairseq2.optim.lr_scheduler.base import LRScheduler
from fairseq2.optim.lr_scheduler.cosine import CosineAnnealingLR
from fairseq2.optim.lr_scheduler.myle import MyleLR
from fairseq2.optim.lr_scheduler.noam import NoamLR
from fairseq2.optim.lr_scheduler.polynomial import PolynomialDecayLR
from fairseq2.optim.lr_scheduler.tri_stage import TriStageLR

if TYPE_CHECKING:  # compat: remove when Python 3.9 support is dropped.
    lr_scheduler_factories = ConfigBoundFactoryRegistry[
        [Optimizer, Optional[int]], LRScheduler
    ]()
else:
    lr_scheduler_factories = ConfigBoundFactoryRegistry()

lr_scheduler_factory = lr_scheduler_factories.decorator


@dataclass
class CosineAnnealingLRConfig:
    cycle_len: Optional[int] = None
    """The number of steps within the first cycle. If ``None``, will be set to
    ``max_num_steps - num_warmup_steps``."""

    num_warmup_steps: int = 0
    """The number of warmup steps."""

    cycle_mul: float = 1.0
    """The factor to grow the length of each cycle."""

    lr_mul: float = 1.0
    """The factor to scale the base and final learning rate at the end of each
    cycle."""

    start_lr: float = 0.0
    """The initial warmup learning rate."""

    final_lr: float = 0.0
    """The final learning rate."""


@lr_scheduler_factory("cosine-annealing")
def create_cosine_annealing_lr(
    config: CosineAnnealingLRConfig, optimizer: Optimizer, max_num_steps: Optional[int]
) -> CosineAnnealingLR:
    if config.cycle_len is None:
        if max_num_steps is None:
            raise ValueError(
                "`cycle_len` must be specified when `max_num_steps` is `None`."
            )

        cycle_len = max_num_steps - config.num_warmup_steps
    else:
        cycle_len = config.cycle_len

    return CosineAnnealingLR(
        optimizer,
        cycle_len,
        config.num_warmup_steps,
        cycle_mul=config.cycle_mul,
        lr_mul=config.lr_mul,
        start_lr=config.start_lr,
        final_lr=config.final_lr,
    )


@dataclass
class MyleLRConfig:
    """Holds the configuration of a :class:`MyleLR`."""

    num_warmup_steps: int = 0
    """The number of warmup steps."""

    start_lr: float = 0.0
    """The initial warmup learning rate."""


@lr_scheduler_factory("myle")
def create_myle_lr(
    config: MyleLRConfig, optimizer: Optimizer, max_num_steps: Optional[int]
) -> MyleLR:
    return MyleLR(optimizer, config.num_warmup_steps, start_lr=config.start_lr)


@dataclass
class NoamLRConfig:
    """Holds the configuration of a :class:`NoamLR`."""

    num_warmup_steps: int = 0
    """The number of warmup steps."""


@lr_scheduler_factory("noam")
def create_noam_lr(
    config: NoamLRConfig, optimizer: Optimizer, max_num_steps: Optional[int]
) -> NoamLR:
    return NoamLR(optimizer, config.num_warmup_steps)


@dataclass
class PolynomialDecayLRConfig:
    """Holds the configuration of a :class:`PolynomialDecayLR`."""

    num_steps: Optional[int] = None
    """The total number of steps, including, warmup, over which to decay the
    learning rate."""

    num_warmup_steps: int = 0
    """The number of warmup steps."""

    power: float = 1.0
    """The exponent of the polynomial used for decay."""

    start_lr: float = 0.0
    """The initial warmup learning rate."""

    final_lr: float = 0.0
    """The final learning rate."""


@lr_scheduler_factory("polynomial-decay")
def create_polynomial_decay_lr(
    config: PolynomialDecayLRConfig, optimizer: Optimizer, max_num_steps: Optional[int]
) -> PolynomialDecayLR:
    num_steps = config.num_steps
    if num_steps is None:
        if max_num_steps is None:
            raise ValueError(
                "`max_num_steps` must be specified when `num_steps` is None."
            )

        num_steps = max_num_steps

    return PolynomialDecayLR(
        optimizer,
        num_steps,
        config.num_warmup_steps,
        power=config.power,
        start_lr=config.start_lr,
        final_lr=config.final_lr,
    )


@dataclass
class TriStageLRConfig:
    """Holds the configuration of a :class:`TriStageLR`."""

    num_steps: Optional[int] = None
    """The total number of steps over which to adjust the learning rate."""

    stage_ratio: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """The ratios of warmup, hold, and decay stages. Must add up to 1."""

    start_lr_scale: float = 0.01
    """The scale of the initial warm-up learning rate."""

    final_lr_scale: float = 0.01
    """The scale of the final learning rate."""


@lr_scheduler_factory("tri-stage")
def create_tri_stage_lr(
    config: TriStageLRConfig, optimizer: Optimizer, max_num_steps: Optional[int]
) -> TriStageLR:
    num_steps = config.num_steps
    if num_steps is None:
        if max_num_steps is None:
            raise ValueError(
                "`max_num_steps` must be specified when `num_steps` is None."
            )

        num_steps = max_num_steps

    return TriStageLR(
        optimizer,
        num_steps,
        config.stage_ratio,
        start_lr_scale=config.start_lr_scale,
        final_lr_scale=config.final_lr_scale,
    )
