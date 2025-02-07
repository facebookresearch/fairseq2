# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Any, Final, Literal, cast, final

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.adamw import adamw  # type: ignore[attr-defined]
from typing_extensions import override

from fairseq2.error import NotSupportedError
from fairseq2.optim._handler import OptimizerHandler
from fairseq2.optim._optimizer import AbstractOptimizer, ParameterCollection
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@final
class AdamW(AbstractOptimizer):
    """Represents an AdamW optimizer.

    This class internally calls the same functional AdamW implementation as
    :class:`torch.optim.AdamW`. The main difference is that it also supports
    memory efficient mixed precision training via its ``use_fp32`` parameter.
    """

    def __init__(
        self,
        params: ParameterCollection,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        impl: Literal["auto", "foreach", "fused", "naive"] = "auto",
        use_fp32: bool = False,
    ) -> None:
        """
        :param params:
            The parameters to optimize.
        :param lr:
            The learning rate.
        :param betas:
            The coefficients used for computing running averages of gradient and
            its square.
        :param eps:
            The term added to the denominator to improve numerical stability.
        :param weight_decay:
            The weight decay coefficient.
        :param amsgrad:
            If ``True``, uses the AMSGrad variant.
        :param maximize:
            If ``True``, maximizes the parameters instead of minimizing.
        :param capturable:
            If ``True``, it is safe to capture this instance in a CUDA graph.
        :param differentiable:
            If ``True``, runs the optimizer step under autograd.
        :param impl:
            The implementation variant. See :class:`torch.optim.AdamW` for
            details.
        :param use_fp32:
            If ``True``, stores the optimizer state in single precision and
            converts gradients on-the-fly to single precision for numerical
            stability.
        """
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "maximize": maximize,
            "capturable": capturable,
            "differentiable": differentiable,
            "impl": impl,
            "use_fp32": use_fp32,
        }

        super().__init__(params, defaults)

        if impl == "fused":
            if differentiable:
                raise NotSupportedError(
                    "`fused` implementation does not support `differentiable`."
                )

            for pg in self.param_groups:
                for p in pg["params"]:
                    if not torch.is_floating_point(p) or p.device.type != "cuda":
                        raise NotSupportedError(
                            "`fused` implementation requires all parameters to be float CUDA tensors."
                        )

            self._step_supports_amp_scaling = True

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)

        state_keys = ["step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"]

        params = chain.from_iterable(
            (pg["params"] for pg in self.param_groups if pg["use_fp32"])
        )

        saved_params = chain.from_iterable(
            (pg["params"] for pg in state_dict["param_groups"] if pg["use_fp32"])
        )

        param_map = {saved_p: p for saved_p, p in zip(saved_params, params)}
        if not param_map:
            return

        # This is a workaround where we override `Optimizer`'s state restore
        # handling to ensure that our state stays in single precision.
        #
        # Note that we use the state tensors in `state_dict` instead of the ones
        # already set in the optimizer since we want to avoid the loss of
        # precision caused by the downcasting in `Optimizer`.
        for saved_param, saved_state in state_dict["state"].items():
            param = param_map[saved_param]

            if param.dtype == torch.float32:
                continue

            state = self.state[param]

            # The base `Optimizer` always casts state tensors to the data type
            # of their corresponding parameter.
            for key in state_keys:
                try:
                    state[key] = saved_state[key].to(
                        device=param.device, dtype=torch.float32
                    )
                except KeyError:
                    pass

    @override
    def _do_step(self) -> None:
        self._cuda_graph_capture_health_check()  # type: ignore[attr-defined]

        for pg in self.param_groups:
            use_fp32: bool = pg["use_fp32"]
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            steps: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            amsgrad = pg["amsgrad"]
            beta1, beta2 = pg["betas"]

            for p in pg["params"]:
                self._init_param(
                    p,
                    pg,
                    use_fp32,
                    params_with_grad,
                    grads,
                    steps,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    amsgrad,
                )

            kwargs: dict[str, object] = {}

            if pg["differentiable"]:
                kwargs["differentiable"] = True

            if (impl := pg["impl"]) != "auto":
                if impl == "naive":
                    # Disables both 'foreach' and 'fused'.
                    kwargs["foreach"] = False
                else:
                    kwargs[impl] = True

            # These two attributes are set by `GradScaler` only for the 'fused'
            # implementaiton which natively supports AMP gradient scaling.
            for attr in ["grad_scale", "found_inf"]:
                if (value := getattr(self, attr, None)) is not None:
                    kwargs[attr] = value

            # Mitigates a shape issue specific to PyTorch 2.0.1.
            if isinstance(found_inf := kwargs.get("found_inf"), Tensor):
                kwargs["found_inf"] = found_inf.squeeze()

            adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=pg["lr"],
                weight_decay=pg["weight_decay"],
                eps=pg["eps"],
                maximize=pg["maximize"],
                capturable=pg["capturable"],
                **kwargs,
            )

            if use_fp32:
                params = (p for p in pg["params"] if p.grad is not None)

                # Cast parameters back to their original data type.
                for original_param, param in zip(params, params_with_grad):
                    if original_param.dtype != torch.float32:
                        original_param.copy_(param)

    def _init_param(
        self,
        param: Tensor,
        param_group: dict[str, object],
        use_fp32: bool,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        steps: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        max_exp_avg_sqs: list[Tensor],
        amsgrad: bool,
    ) -> None:
        grad = param.grad
        if grad is None:
            return

        if grad.is_sparse:
            raise NotSupportedError("`AdamW` does not support sparse gradients.")

        state = cast(dict[str, Tensor], self.state[param])  # type: ignore[index]

        if use_fp32:
            if param.dtype != torch.float32:
                param = param.float()

            if grad.dtype != torch.float32:
                grad = grad.float()

        params_with_grad.append(param)

        grads.append(grad)

        if len(state) == 0:
            if param_group["capturable"] or param_group["impl"] == "fused":
                step_device = param.device
            else:
                step_device = None

            # Step counter.
            state["step"] = torch.zeros((), device=step_device, dtype=torch.float32)

            # Exponential moving average of gradient values.
            state["exp_avg"] = torch.zeros_like(param)

            # Exponential moving average of squared gradient values.
            state["exp_avg_sq"] = torch.zeros_like(param)

            if amsgrad:
                state["max_exp_avg_sq"] = torch.zeros_like(param)

        steps.append(state["step"])

        exp_avgs.append(state["exp_avg"])
        exp_avg_sqs.append(state["exp_avg_sq"])

        if amsgrad:
            max_exp_avg_sqs.append(state["max_exp_avg_sq"])


ADAMW_OPTIMIZER: Final = "adamw"


@dataclass(kw_only=True)
class AdamWConfig:
    lr: float = 1e-3
    """The learning rate."""

    betas: tuple[float, float] = (0.9, 0.999)
    """The coefficients used for computing running averages of gradient and its
    square."""

    eps: float = 1e-8
    """The term added to the denominator to improve numerical stability."""

    weight_decay: float = 0.0
    """The weight decay coefficient."""

    amsgrad: bool = False
    """If ``True``, uses the AMSGrad variant."""

    maximize: bool = False
    """If ``True``, maximizes the parameters instead of minimizing."""

    capturable: bool = False
    """If ``True``, it is safe to capture this instance in a CUDA graph."""

    differentiable: bool = False
    """If ``True``, runs the optimizer step under autograd."""

    impl: Literal["auto", "foreach", "fused", "naive"] = "auto"
    """The implementation variant. See :class:`torch.optim.AdamW` for details."""

    use_fp32: bool = False
    """If ``True``, stores the optimizer state in single precision and converts
    gradients on-the-fly to single precision for numerical stability."""


@final
class AdamWHandler(OptimizerHandler):
    @override
    def create(self, params: ParameterCollection, config: object) -> Optimizer:
        config = structure(config, AdamWConfig)

        validate(config)

        return AdamW(
            params,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad,
            maximize=config.maximize,
            capturable=config.capturable,
            differentiable=config.differentiable,
            impl=config.impl,
            use_fp32=config.use_fp32,
        )

    @property
    @override
    def config_kls(self) -> type[object]:
        return AdamWConfig
