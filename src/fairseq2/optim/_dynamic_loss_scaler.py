# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast, final

import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.optim import Optimizer
from typing_extensions import override

from fairseq2.error import InvalidOperationError
from fairseq2.gang import Gang
from fairseq2.logging import log
from fairseq2.typing import Device


@final
class DynamicLossScaler:
    """Performs loss scaling during backward pass to prevent underflow of half
    precision gradients."""

    _optimizer: Optimizer
    _grad_scaler: GradScaler
    _scale_window: int
    _min_scale: float
    _enabled: bool

    def __init__(
        self,
        optimizer: Optimizer,
        gang: Gang,
        *,
        sharded: bool = True,
        init_scale: float = 2.0**15,
        scale_factor: float = 2.0,
        scale_window: int | None = None,
        min_scale: float = 0.0,
        gradient_accumulation: int = 1,
        enabled: bool = True,
    ) -> None:
        """
        :param optimizer:
            The optimizer that holds the gradients that will be unscaled.
        :param gang:
            The associated gang.
        :param sharded:
            If ``True``, assumes that the optimizer state is sharded.
        :param init_scale:
            The initial scale.
        :param scale_factor:
            The factor by which the scale is multiplied if no inf/NaN gradients
            occur for ``scale_window`` consecutive optimizer steps.
        :param scale_window:
            The number of consecutive optimizer steps without inf/NaN gradients
            that must occur for the scale to be multiplied by ``scale_factor``.
            If ``None``, the window size will be determined by a heuristic
            method.
        :param min_scale:
            The minimum scale.
        :param gradient_accumulation:
            The number of steps to accumulate gradients before an optimizer
            update. Used only when ``scale_window`` is ``None``.
        :param enabled:
            If ``False``, disables loss scaling.
        """
        if enabled:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.dtype != torch.float32 and param.dtype != torch.float16:
                        raise ValueError(
                            f"The parameters held by `optimizer` must be of type `torch.float32` or `torch.float16`, but at least one parameter is of type `{param.dtype}` instead."
                        )

        if scale_window is None:
            if enabled:
                # The same formula as in fairseq.
                scale_window = max(int(2**14 / gang.size / gradient_accumulation), 1)

                log.info("float16 loss scale window set to {}.", scale_window)
            else:
                scale_window = 1

        if not enabled or not sharded or gang.size == 1:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message=r".*torch\.cuda\.amp\.GradScaler is enabled.*"  # fmt: skip
                )

                self._grad_scaler = InternalGradScaler(
                    init_scale=init_scale,
                    growth_factor=scale_factor,
                    backoff_factor=1 / scale_factor,
                    growth_interval=scale_window,
                    enabled=enabled,
                )
        else:
            if not supports_manual_gradient_scaling(optimizer):
                raise ValueError(
                    "`optimizer` must support manual gradient scaling via `torch.amp.GradScaler`, but supports only implicit scaling in its step function (i.e. `_step_supports_amp_scaling == True`) which is not supported in a distributed setting."
                )

            pg = gang.as_process_group()

            self._grad_scaler = ShardedGradScaler(
                init_scale=init_scale,
                growth_factor=scale_factor,
                backoff_factor=1 / scale_factor,
                growth_interval=scale_window,
                enabled=enabled,
                process_group=pg,
            )

        self._optimizer = optimizer
        self._scale_window = scale_window
        self._min_scale = min_scale
        self._enabled = enabled

    def state_dict(self) -> dict[str, object]:
        return {"grad_scaler": self._grad_scaler.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        try:
            gs_state_dict = state_dict["grad_scaler"]
        except KeyError:
            raise ValueError("`state_dict` must contain a 'grad_scaler' key.") from None

        if not isinstance(gs_state_dict, dict):
            raise TypeError(
                f"`state_dict['grad_scaler']` must be of type `dict`, but is of type `{type(gs_state_dict)}` instead."
            )

        try:
            self._grad_scaler.load_state_dict(gs_state_dict)
        except (RuntimeError, ValueError) as ex:
            raise ValueError(
                f"`state_dict['grad_scaler']` is not a valid `{type(self._grad_scaler)}` state. See the nested exception for details."
            ) from ex

    def run_optimizer_step(
        self, step_nr: int, closure: Callable[[], float] | None = None
    ) -> tuple[float | None, LossScaleResult]:
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
        loss = self._grad_scaler.step(self._optimizer, closure)

        return loss, self._update_scale(step_nr)

    def _update_scale(self, step_nr: int) -> LossScaleResult:
        old_scale = self._grad_scaler.get_scale()

        self._grad_scaler.update()

        new_scale = self._grad_scaler.get_scale()

        if self._are_close(old_scale, new_scale):
            return LossScaleResult(old_scale, new_scale)

        if new_scale > old_scale:
            log.info("No gradient overflow detected in the last {} step(s) after step {}, increasing loss scale from {:g} to {:g}.", self._scale_window, step_nr, old_scale, new_scale)  # fmt: skip

            return LossScaleResult(old_scale, new_scale)

        if self._min_scale > new_scale:
            self._grad_scaler.update(self._min_scale)

            if self._are_close(old_scale, self._min_scale):
                log.error("Overflow detected at step {}, ignoring gradient, loss scale is already at minimum ({:g}). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping, or increasing the batch size.", step_nr, self._min_scale)  # fmt: skip
            else:
                log.error("Overflow detected at step {}, ignoring gradient, decreasing loss scale from {:g} to {:g} (minimum). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping, or increasing the batch size.", step_nr, old_scale, self._min_scale)  # fmt: skip

            return LossScaleResult(
                old_scale, new_scale, overflow=True, min_reached=True
            )
        else:
            log.info("Overflow detected at step {}, ignoring gradient, decreasing loss scale from {:g} to {:g}.", step_nr, old_scale, new_scale)  # fmt: skip

            return LossScaleResult(old_scale, new_scale, overflow=True)

    @staticmethod
    def _are_close(a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=1.3e-6, abs_tol=1e-5)

    def unscale_gradients_(self) -> None:
        """Unscale the associated optimizer's gradients by the current scale."""
        if self._enabled and not supports_manual_gradient_scaling(self._optimizer):
            raise InvalidOperationError(
                "`optimizer` must support manual gradient scaling via `torch.amp.GradScaler`, but supports only implicit scaling in its step function (i.e. `_step_supports_amp_scaling == True`)."
            )

        self._grad_scaler.unscale_(self._optimizer)

    def backward(self, loss: Tensor) -> None:
        """Compute the gradient of ``loss`` after scaling it to avoid underflow."""
        self._grad_scaler.scale(loss).backward()

    def get_scale(self) -> float:
        """Return the current scale, or 1.0 if loss scaling is disabled."""
        return cast(float, self._grad_scaler.get_scale())  # type: ignore[redundant-cast]

    @property
    def is_enabled(self) -> bool:
        """``True`` if the loss scaling is enabled."""
        return self._enabled


@dataclass(frozen=True)
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


def supports_manual_gradient_scaling(optimizer: Optimizer) -> bool:
    """
    Returns ``True`` if ``optimizer`` supports manual gradient scaling.
    """
    return not getattr(optimizer, "_step_supports_amp_scaling", False)


class InternalGradScaler(GradScaler):
    @override
    def _unscale_grads_(
        self,
        optimizer: Optimizer,
        inv_scale: Tensor,
        found_inf: Tensor,
        allow_fp16: bool,
    ) -> dict[Device, Tensor]:
        # `GradScaler` artificially limits fp16 gradients only to optimizers
        # that natively support AMP. Here, we hijack `_unscale_grads_()` and
        # always pass `allow_fp16=True` to the real function.
        return super()._unscale_grads_(optimizer, inv_scale, found_inf, allow_fp16=True)  # type: ignore[no-any-return]
