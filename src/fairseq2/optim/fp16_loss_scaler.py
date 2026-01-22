# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, final, overload

from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.optim import Optimizer
from typing_extensions import override

from fairseq2.device import Device
from fairseq2.error import InternalError, StateDictError
from fairseq2.gang import Gangs
from fairseq2.typing import Stateful


class Float16LossScaler(ABC, Stateful):
    """
    Performs loss scaling during backward pass to prevent underflow of half
    precision gradients.
    """

    @overload
    def run_optimizer_step(self, optimizer: Optimizer) -> Float16LossScaleResult: ...

    @overload
    def run_optimizer_step(
        self, optimizer: Optimizer, closure: Callable[[], float] | None
    ) -> tuple[float, Float16LossScaleResult]: ...

    @abstractmethod
    def run_optimizer_step(
        self, optimizer: Optimizer, closure: Callable[[], float] | None = None
    ) -> Float16LossScaleResult | tuple[float, Float16LossScaleResult]:
        """
        Performs an optimization step.

        :param optimizer: Optimizer that holds the gradients to scale or unscale.
        :param closure: Closure that reevaluates the model and returns the loss.
            Optional for most optimizers.

        :returns:
            - Return value of ``closure``.
            - Result of the loss scale operation.
        """

    @abstractmethod
    def unscale_grads_(self, optimizer: Optimizer) -> None:
        """Unscales the gradients of the optimizer by the current scale."""

    @abstractmethod
    def backward(self, loss: Tensor) -> None:
        """Computes the gradient of ``loss`` after scaling it to avoid underflow."""

    @abstractmethod
    def state_dict(self) -> dict[str, object]: ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, object]) -> None: ...

    @property
    @abstractmethod
    def scale_window(self) -> int: ...


@final
class _NoopFloat16LossScaler(Float16LossScaler):
    @override
    @overload
    def run_optimizer_step(self, optimizer: Optimizer) -> Float16LossScaleResult: ...

    @override
    @overload
    def run_optimizer_step(
        self, optimizer: Optimizer, closure: Callable[[], float] | None
    ) -> tuple[float, Float16LossScaleResult]: ...

    def run_optimizer_step(
        self, optimizer: Optimizer, closure: Callable[[], float] | None = None
    ) -> Float16LossScaleResult | tuple[float, Float16LossScaleResult]:
        loss = optimizer.step(closure)

        scale_result = Float16LossScaleResult(old_scale=1.0, new_scale=1.0)

        if closure is not None:
            if loss is None:
                raise InternalError("`closure` returned `None`")

            return loss, scale_result

        return scale_result

    @override
    def unscale_grads_(self, optimizer: Optimizer) -> None:
        pass

    @override
    def backward(self, loss: Tensor) -> None:
        loss.backward()

    @override
    def state_dict(self) -> dict[str, object]:
        return {}

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        StateDictError.raise_if_not_empty(state_dict)

    @property
    @override
    def scale_window(self) -> int:
        return 0


NOOP_FP16_LOSS_SCALER: Final = _NoopFloat16LossScaler()


@final
class StandardFloat16LossScaler(Float16LossScaler):
    def __init__(
        self,
        gangs: Gangs,
        *,
        init_scale: float = 2.0**16,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        min_scale: float = 0.0001,
    ) -> None:
        """
        :param gangs: Gangs.
        :param init_scale: Initial scale.
        :param scale_factor: Factor by which the scale is multiplied if no
            inf/NaN gradients occur for ``scale_window`` consecutive optimizer
            steps.
        :param scale_window: Number of consecutive optimizer steps without
            inf/NaN gradients that must occur for the scale to be multiplied by
            ``scale_factor``.  If ``None``, the window size will be determined
            heuristically.
        :param min_scale: Minimum scale.
        """
        # Means either FSDP or non-data parallelism. In both cases, we have to
        # use `ShardedGradScaler` to ensure that grad scales are synced.
        sharded = gangs.root.size != gangs.rdp.size

        grad_scaler: GradScaler

        if sharded:
            root_pg = gangs.root.as_process_group()

            grad_scaler = ShardedGradScaler(
                init_scale=init_scale,
                growth_factor=scale_factor,
                backoff_factor=1 / scale_factor,
                growth_interval=scale_window,
                process_group=root_pg,
            )
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message=r".*torch\.cuda\.amp\.GradScaler is enabled.*"  # fmt: skip
                )

                grad_scaler = _InternalGradScaler(
                    init_scale=init_scale,
                    growth_factor=scale_factor,
                    backoff_factor=1 / scale_factor,
                    growth_interval=scale_window,
                )

        self._grad_scaler = grad_scaler
        self._sharded = sharded
        self._scale_window = scale_window
        self._min_scale = min_scale

    @override
    @overload
    def run_optimizer_step(self, optimizer: Optimizer) -> Float16LossScaleResult: ...

    @override
    @overload
    def run_optimizer_step(
        self, optimizer: Optimizer, closure: Callable[[], float] | None
    ) -> tuple[float, Float16LossScaleResult]: ...

    def run_optimizer_step(
        self, optimizer: Optimizer, closure: Callable[[], float] | None = None
    ) -> Float16LossScaleResult | tuple[float, Float16LossScaleResult]:
        loss = self._grad_scaler.step(optimizer, closure)

        scale_result = self._update_scale()

        if closure is not None:
            if loss is None:
                raise InternalError("`closure` returned `None`")

            return loss, scale_result

        return scale_result

    def _update_scale(self) -> Float16LossScaleResult:
        old_scale = self._grad_scaler.get_scale()

        self._grad_scaler.update()

        new_scale = self._grad_scaler.get_scale()

        if math.isclose(new_scale, old_scale, rel_tol=1.3e-6, abs_tol=1e-5):
            return Float16LossScaleResult(old_scale, new_scale)

        if new_scale > old_scale:
            return Float16LossScaleResult(old_scale, new_scale, scaled=True)

        if self._min_scale > new_scale:
            self._grad_scaler.update(self._min_scale)

            return Float16LossScaleResult(
                old_scale, self._min_scale, overflowed=True, exploded=True
            )
        else:
            return Float16LossScaleResult(old_scale, new_scale, overflowed=True)

    @override
    def unscale_grads_(self, optimizer: Optimizer) -> None:
        if self._sharded:
            if not supports_manual_grad_scaling(optimizer):
                raise ValueError(
                    "`optimizer` must support manual gradient scaling via `torch.amp.GradScaler`, but supports only implicit scaling in its step function (i.e. `_step_supports_amp_scaling == True`)."
                )

        self._grad_scaler.unscale_(optimizer)

    @override
    def backward(self, loss: Tensor) -> None:
        self._grad_scaler.scale(loss).backward()

    @property
    @override
    def scale_window(self) -> int:
        return self._scale_window

    @override
    def state_dict(self) -> dict[str, object]:
        return {"grad_scaler": self._grad_scaler.state_dict()}

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        state_dict = dict(state_dict)

        try:
            gs_state_dict = state_dict.pop("grad_scaler")
        except KeyError:
            raise StateDictError(
                "`state_dict` is expected to contain a 'grad_scaler' key."
            ) from None

        if not isinstance(gs_state_dict, dict):
            raise StateDictError(
                f"`state_dict['grad_scaler']` is expected to be of type `{dict}`, but is of type `{type(gs_state_dict)}` instead."
            )

        try:
            self._grad_scaler.load_state_dict(gs_state_dict)
        except (RuntimeError, ValueError, TypeError) as ex:
            raise StateDictError(
                f"`state_dict['grad_scaler']` does not represent a valid `{type(self._grad_scaler)}` state."
            ) from ex

        StateDictError.raise_if_not_empty(state_dict)


@dataclass(frozen=True)
class Float16LossScaleResult:
    old_scale: float
    """Scale before the optimizer step."""

    new_scale: float
    """Scale after the optimizer step."""

    scaled: bool = False

    overflowed: bool = False
    """If ``True``, loss has overflowed."""

    exploded: bool = False


def supports_manual_grad_scaling(optimizer: Optimizer) -> bool:
    """
    Returns ``True`` if ``optimizer`` supports manual gradient scaling.
    """
    return not getattr(optimizer, "_step_supports_amp_scaling", False)


class _InternalGradScaler(GradScaler):
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
