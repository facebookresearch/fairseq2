# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from fairseq2.optim.optimizer import AbstractOptimizer, ParameterCollection


class SophiaG(AbstractOptimizer):
    """Represents a SophiaG optimizer."""

    def __init__(
        self,
        params: ParameterCollection,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.965, 0.99),
        k: int = 10,
        rho: float = 0.04,
        weight_decay: float = 1e-1,
        *,
        maximize: bool = False,
        capturable: bool = False,
    ) -> None:
        """
        :param params:
            The parameters to optimize.
        :param lr:
            The learning rate.
        :param betas:
            The coefficients used for computing running averages of gradient and
            its square.
        :param rho:
            The parameter clipping threshold.
        :param k:
            The number of optimizer steps before updating the parameter Hessian values.
        :param weight_decay:
            The weight decay coefficient.
        :param maximize:
            If ``True``, maximizes the parameters instead of minimizing.
        :param capturable:
            If ``True``, it is safe to capture this instance in a CUDA graph.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay parameter: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "rho": rho,
            "k": k,
            "weight_decay": weight_decay,
            "maximize": maximize,
            "capturable": capturable,
            "differentiable": False,
        }

        super().__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("capturable", False)

        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )

        if not step_is_tensor:
            for value in state_values:
                value["step"] = torch.tensor(float(value["step"]))

    def _do_step(self, bs: int = 5120) -> None:
        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            hessian: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError("Hero does not support sparse gradients")

                grads.append(p.grad)
                state = self.state[p]

                # State initialization.
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if self.defaults["capturable"]
                        else torch.tensor(0.0)
                    )
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                if "hessian" not in state.keys():
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                # Hessian update.
                if state["step"] % group["k"] == 0:
                    state["hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                exp_avgs.append(state["exp_avg"])
                state_steps.append(state["step"])
                hessian.append(state["hessian"])

                if self.defaults["capturable"]:
                    bs = torch.ones((1,), dtype=torch.float, device=p.device) * bs

            sophiag(
                params_with_grad,
                grads,
                exp_avgs,
                hessian,
                state_steps,
                bs=bs,
                beta1=beta1,
                beta2=beta2,
                rho=group["rho"],
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"],
                capturable=group["capturable"],
            )


def sophiag(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    hessian: List[Tensor],
    state_steps: List[Tensor],
    capturable: bool = False,
    *,
    bs: int,
    beta1: float,
    beta2: float,
    rho: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
) -> None:
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda and bs.is_cuda

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)

        step_t += 1

        # Perform stepweight decay.
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient.
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        if capturable:
            step_size = lr
            step_size_neg = step_size.neg()

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
        else:
            step_size_neg = -lr

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
