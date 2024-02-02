# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, cast

import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.optim import Optimizer

from fairseq2.gang import Gang

logger = logging.getLogger(__name__)


class DynamicLossScaler:
    """Performs loss scaling during backward pass to prevent underflow of half
    precision gradients."""

    optimizer: Optimizer
    init_scale: float
    scale_factor: float
    scale_window: int
    min_scale: float
    enabled: bool

    _grad_scaler: GradScaler

    def __init__(
        self,
        optimizer: Optimizer,
        gang: Gang,
        *,
        init_scale: float = 2.0**15,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        min_scale: float = 0.0,
        enabled: bool = True,
    ) -> None:
        """
        :param optimizer:
            The optimizer that holds the gradients that will be unscaled.
        :param gang:
            The associated gang.
        :param init_scale:
            The initial scale.
        :param scale_factor:
            The factor by which the scale is multiplied if no inf/NaN gradients
            occur for ``scale_window`` consecutive optimizer steps.
        :param scale_window:
            The number of consecutive optimizer steps without inf/NaN gradients
            that must occur for the scale to be multiplied by ``scale_factor``.
        :param min_scale:
            The minimum allowed scale.
        :param enabled:
            If ``False``, disables loss scaling.
        """
        if enabled:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.dtype != torch.float16:
                        raise ValueError(
                            f"The parameters held by `optimizer` must be of type `torch.float16`, but at least one parameter is of type `{param.dtype}`."
                        )

                    if param.device.type != "cuda":
                        raise ValueError(
                            f"The parameters held by `optimizer` must be on a CUDA device, but at least one parameter is on a {param.device.type.upper()} device."
                        )

        if gang.size == 1:
            self._grad_scaler = GradScaler(
                init_scale, scale_factor, 1 / scale_factor, scale_window, enabled
            )
        else:
            pg = gang.as_process_group()

            # Yes, `growth_factor` and `backoff_factor` parameters are swapped.
            self._grad_scaler = ShardedGradScaler(
                init_scale, 1 / scale_factor, scale_factor, scale_window, enabled, pg
            )

        self.optimizer = optimizer
        self.init_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.enabled = enabled

    def state_dict(self) -> Dict[str, Any]:
        return {"grad_scaler": self._grad_scaler.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        try:
            self._grad_scaler.load_state_dict(state_dict["grad_scaler"])
        except KeyError as ex:
            raise ValueError(
                "`state_dict` must contain the state of the internal `GradScaler`."
            ) from ex

    def run_optimizer_step(
        self, step_nr: int, closure: Optional[Callable[[], float]] = None
    ) -> Tuple[Optional[float], "LossScaleResult"]:
        """Perform a single optimization step.

        :param step_nr:
            The number of the training step. Used for logging purposes.
        :param closure:
            A closure that reevaluates the model and returns the loss. Optional
            for most optimizers.

        :returns:
            - The return value of ``closure``.
            - The result of the loss scale operation.
        """
        if isinstance(self._grad_scaler, ShardedGradScaler):
            # As of PyTorch 2.0.1, `ShardedGradScaler` has a bug where it skips
            # calling `unscale_()` for optimizers that natively support gradient
            # scaling. Although this is the expected behavior for `GradScaler`,
            # in distributed settings this causes the scale to get out-of-sync
            # between processes. Here we force ranks to sync their inf/NaNs by
            # manually calling `unscale_()`.
            try:
                self._grad_scaler.unscale_(self.optimizer)  # type: ignore[arg-type]
            except RuntimeError as ex:
                if not str(ex).startswith("unscale_() has already been called"):
                    raise

        loss = self._grad_scaler.step(self.optimizer, closure)

        return loss, self._update_scale(step_nr)

    def _update_scale(self, step_nr: int) -> LossScaleResult:
        old_scale = self._grad_scaler.get_scale()

        self._grad_scaler.update()

        new_scale = self._grad_scaler.get_scale()

        if self._are_close(old_scale, new_scale):
            return LossScaleResult(old_scale, new_scale)

        if new_scale > old_scale:
            logger.info("No gradient overflow detected in the last %s step(s) after step %d, increasing loss scale from %s to %s.", self.scale_window, step_nr, old_scale, new_scale)  # fmt: skip

            return LossScaleResult(old_scale, new_scale)

        if self.min_scale > new_scale:
            self._grad_scaler.update(self.min_scale)

            if self._are_close(old_scale, self.min_scale):
                logger.warning("Overflow detected at step %d, ignoring gradient, loss scale is already at minimum (%s). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping, or increasing the batch size.", step_nr, self.min_scale)  # fmt: skip
            else:
                logger.warning("Overflow detected at step %d, ignoring gradient, decreasing loss scale from %s to %s (minimum). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping, or increasing the batch size.", step_nr, old_scale, self.min_scale)  # fmt: skip

            return LossScaleResult(
                old_scale, new_scale, overflow=True, min_reached=True
            )
        else:
            logger.info("Overflow detected at step %d, ignoring gradient, decreasing loss scale from %s to %s.", step_nr, old_scale, new_scale)  # fmt: skip

            return LossScaleResult(old_scale, new_scale, overflow=True)

    @staticmethod
    def _are_close(a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=1.3e-6, abs_tol=1e-5)

    def unscale_optimizer_grads_(self) -> None:
        """Unscale the associated optimizer's gradients by the current scale."""
        self._grad_scaler.unscale_(self.optimizer)

    def backward(self, loss: Tensor) -> None:
        """Compute the gradient of ``loss`` after scaling it to avoid underflow."""
        self._grad_scaler.scale(loss).backward()

    def get_scale(self) -> float:
        """Return the current scale, or 1.0 if loss scaling is disabled."""
        return cast(float, self._grad_scaler.get_scale())  # type: ignore[redundant-cast]


@dataclass
class LossScaleResult:
    """Holds the result of a loss scale operation."""

    old_scale: float
    """The scale before the optimizer step."""

    new_scale: float
    """The scale after the optimizer step."""

    overflow: bool = False
    """If ``True``, the loss has overflowed."""

    min_reached: bool = False
    """If ``True``, the scale has been decreased to its minimum value."""
