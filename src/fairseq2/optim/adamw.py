# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import chain
from typing import Any, Dict, Iterable, List, Literal, Tuple, Union, cast, final

import torch
from torch import Tensor
from torch.optim.adamw import adamw  # type: ignore[attr-defined]

from fairseq2.optim.optimizer_base import OptimizerBase
from fairseq2.typing import finaloverride


@final
class AdamW(OptimizerBase):
    """Implements AdamW algorithm.

    This class uses the same functional AdamW implementation as
    :class:`torch.optim.AdamW`. The main difference is that it also supports
    mixed precision training when ``use_fp32`` parameter is set.
    """

    def __init__(
        self,
        params: Union[Iterable[Tensor], Iterable[Dict[str, Any]]],
        *,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        impl: Literal["auto", "foreach", "fused"] = "auto",
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
            If ``True``, maximizes the parameters based on the objective,
            instead of minimizing.
        :param capturable:
            If ``True``, it is safe to capture this instance in a CUDA graph.
        :param differentiable:
            If ``True``, runs the optimizer step under autograd.
        :param impl:
            The implementation variant. See :class:`torch.optim.AdamW` for
            details.
        :param use_fp32:
            If ``True``, stores the optimizer state in single precision (i.e.
            ``torch.float32``) for better numerical stability at the cost of
            higher memory consumption.
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
                raise RuntimeError(
                    "`fused` implementation does not support `differentiable`."
                )

            for pg in self.param_groups:
                for p in pg["params"]:
                    if not torch.is_floating_point(p) or p.device.type != "cuda":
                        raise RuntimeError(
                            "`fused` implementation requires all parameters to be float CUDA tensors."
                        )

            self._step_supports_amp_scaling = True

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)

        state_keys = ["step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"]

        fp32_params = chain.from_iterable(
            (pg["params"] for pg in self.param_groups if pg["use_fp32"])
        )

        saved_fp32_params = chain.from_iterable(
            (pg["params"] for pg in state_dict["param_groups"] if pg["use_fp32"])
        )

        param_map = {saved_p: p for saved_p, p in zip(saved_fp32_params, fp32_params)}

        # If we don't have any parameter group with `use_fp32`, skip the rest.
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
            # of their correspondig parameter.
            for key in state_keys:
                try:
                    state[key] = saved_state[key].to(
                        device=param.device, dtype=torch.float32
                    )
                except KeyError:
                    pass

    @finaloverride
    def _do_step(self) -> None:
        self._cuda_graph_capture_health_check()  # type: ignore[attr-defined]

        for pg in self.param_groups:
            use_fp32 = pg["use_fp32"]
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            steps: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            max_exp_avg_sqs: List[Tensor] = []
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

            kwargs: Dict[str, Any] = {}

            if pg["differentiable"]:
                kwargs["differentiable"] = True

            if (impl := pg["impl"]) != "auto":
                kwargs[impl] = True

            for attr in ["grad_scale", "found_inf"]:
                if (value := getattr(self, attr, None)) is not None:
                    kwargs[attr] = value

            # Mitigates a shape issue that is specific to PyTorch 2.0.1.
            if (found_inf := kwargs.get("found_inf")) is not None:
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
        param_group: Dict[str, Any],
        use_fp32: bool,
        params_with_grad: List[Tensor],
        grads: List[Tensor],
        steps: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        amsgrad: bool,
    ) -> None:
        grad = param.grad
        if grad is None:
            return

        if grad.is_sparse:
            raise RuntimeError("`AdamW` does not support sparse gradients.")

        state = cast(Dict[str, Tensor], self.state[param])  # type: ignore[index]

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
