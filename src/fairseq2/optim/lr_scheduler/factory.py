# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from torch.optim import Optimizer

from fairseq2.factory_registry import ConfigBoundFactoryRegistry
from fairseq2.logging import get_log_writer
from fairseq2.optim.lr_scheduler.base import LRScheduler
from fairseq2.optim.lr_scheduler.cosine import CosineAnnealingLR
from fairseq2.optim.lr_scheduler.myle import MyleLR
from fairseq2.optim.lr_scheduler.noam import NoamLR
from fairseq2.optim.lr_scheduler.polynomial import PolynomialDecayLR
from fairseq2.optim.lr_scheduler.tri_stage import TriStageLR

lr_scheduler_factories = ConfigBoundFactoryRegistry[
    [Optimizer, int | None], LRScheduler
]()

lr_scheduler_factory = lr_scheduler_factories.decorator

log = get_log_writer(__name__)


def create_lr_scheduler(
    name: str,
    optimizer: Optimizer,
    unstructured_config: object = None,
    *,
    max_num_steps: int | None = None,
) -> LRScheduler:
    """Create a learning rate scheduler of type registered with ``name``.

    :param name:
        The name of the learning rate scheduler.
    :param optimizer:
        The optimizer to associate.
    :param unstructured_config:
        The configuration of the learning rate scheduler.
    :param max_num_steps:
        The maximum number of training steps.
    """
    factory = lr_scheduler_factories.get(name, unstructured_config)

    return factory(optimizer, max_num_steps)


@dataclass(kw_only=True)
class CosineAnnealingLRConfig:
    cycle_len: int | None = None
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

    final_lr: float | None = None
    """The final learning rate, ignored in favor of final_lr_scale if set to None. If this is None, then final_lr_scale has to be not None."""

    final_lr_scale: float | None = 0.2
    """Scale multipled with optimizer LR to set the final LR. If this is None, then final_lr has to be not None."""


@lr_scheduler_factory("cosine-annealing")
def create_cosine_annealing_lr(
    config: CosineAnnealingLRConfig, optimizer: Optimizer, max_num_steps: int | None
) -> CosineAnnealingLR:
    if config.cycle_len is None:
        if max_num_steps is None:
            raise ValueError(
                "`cycle_len` must be specified when `max_num_steps` is `None`."
            )

        cycle_len = max_num_steps - config.num_warmup_steps
    else:
        cycle_len = config.cycle_len

    # Validate config and set final_lr
    if (config.final_lr is not None) and (config.final_lr_scale is not None):
        raise ValueError(
            f"Invalid configuration: Both `final_lr` ({config.final_lr}) and `final_lr_scale` ({config.final_lr_scale}) are set. Please specify only one."
        )

    # Compute final_lr based on the configuration
    if config.final_lr_scale is not None:
        final_lr = optimizer.param_groups[0]["lr"] * config.final_lr_scale
    elif config.final_lr is not None:
        final_lr = config.final_lr
    else:
        raise ValueError(
            "Invalid configuration: Either `final_lr` or `final_lr_scale` must be specified."
        )

    if final_lr > optimizer.param_groups[0]["lr"]:
        log.warning(
            f"ATTENTION: Final LR scheduler value ({final_lr}) > Optimizer LR ({optimizer.param_groups[0]['lr']}). This means your learning rate will increase over the course of training."
        )

    return CosineAnnealingLR(
        optimizer,
        cycle_len,
        config.num_warmup_steps,
        cycle_mul=config.cycle_mul,
        lr_mul=config.lr_mul,
        start_lr=config.start_lr,
        final_lr=final_lr,
    )


@dataclass(kw_only=True)
class MyleLRConfig:
    """Holds the configuration of a :class:`MyleLR`."""

    num_warmup_steps: int = 0
    """The number of warmup steps."""

    start_lr: float = 0.0
    """The initial warmup learning rate."""


@lr_scheduler_factory("myle")
def create_myle_lr(
    config: MyleLRConfig, optimizer: Optimizer, max_num_steps: int | None
) -> MyleLR:
    return MyleLR(optimizer, config.num_warmup_steps, start_lr=config.start_lr)


@dataclass(kw_only=True)
class NoamLRConfig:
    """Holds the configuration of a :class:`NoamLR`."""

    num_warmup_steps: int = 0
    """The number of warmup steps."""


@lr_scheduler_factory("noam")
def create_noam_lr(
    config: NoamLRConfig, optimizer: Optimizer, max_num_steps: int | None
) -> NoamLR:
    return NoamLR(optimizer, config.num_warmup_steps)


@dataclass(kw_only=True)
class PolynomialDecayLRConfig:
    """Holds the configuration of a :class:`PolynomialDecayLR`."""

    num_steps: int | None = None
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
    config: PolynomialDecayLRConfig, optimizer: Optimizer, max_num_steps: int | None
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


@dataclass(kw_only=True)
class TriStageLRConfig:
    """Holds the configuration of a :class:`TriStageLR`."""

    num_steps: int | None = None
    """The total number of steps over which to adjust the learning rate."""

    stage_ratio: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """The ratios of warmup, hold, and decay stages. Must add up to 1."""

    start_lr_scale: float = 0.01
    """The scale of the initial warm-up learning rate."""

    final_lr_scale: float = 0.01
    """The scale of the final learning rate."""


@lr_scheduler_factory("tri-stage")
def create_tri_stage_lr(
    config: TriStageLRConfig, optimizer: Optimizer, max_num_steps: int | None
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
