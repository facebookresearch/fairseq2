# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterable, List, Literal, Tuple, Union, cast, final

import torch
from torch import Tensor
from torch.optim.adamw import adamw  # type: ignore[attr-defined]

from fairseq2.optim.optimizer_base import OptimizerBase
from fairseq2.typing import finaloverride
from fairseq2.utils.version import is_pt2_or_greater


@final
class AdamW(OptimizerBase):
    """Implements AdamW algorithm.

    This class internally uses the same functional implementation as
    :class:`torch.optim.AdamW`. The main difference is that it casts parameters
    and gradients to single precision (i.e. ``torch.float32``) during an
    optimization step to have better numerical stability. The updated parameters
    are then cast back to their original data type at the end of the step.
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
            If ``True``, indicates that it is safe to capture this instance in a
            CUDA graph.
        :param differentiable:
            If ``True``, runs the optimizer step under autograd.
        :param impl:
            The implementation variant. See :class:`torch.optim.AdamW` for the
            details.
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
        }

        super().__init__(params, defaults)

        if differentiable and not is_pt2_or_greater():
            raise RuntimeError("`differentiable` requires PyTorch 2.0 or greater.")

        if impl == "fused":
            if not is_pt2_or_greater():
                raise RuntimeError(
                    "`fused` implementation requires PyTorch 2.0 or greater."
                )

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

        # The base optimizer casts all state tensors to the data type of their
        # parameter.
        for state in self.state.values():
            for name in ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]:
                try:
                    state[name] = state[name].float()
                except KeyError:
                    pass

    @finaloverride
    def _do_step(self) -> None:
        self._cuda_graph_capture_health_check()  # type: ignore[attr-defined]

        for pg in self.param_groups:
            params_with_grad: List[Tensor] = []
            fp32_params_with_grad: List[Tensor] = []
            fp32_grads: List[Tensor] = []
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
                    params_with_grad,
                    fp32_params_with_grad,
                    fp32_grads,
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
                    try:
                        kwargs[attr] = getattr(self, attr)
                    except AttributeError:
                        pass

            # Mitigates a shape issue that is specific to PyTorch 2.0.1.
            try:
                kwargs["found_inf"] = kwargs["found_inf"].squeeze()
            except KeyError:
                pass

            adamw(
                fp32_params_with_grad,
                fp32_grads,
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

            # Cast parameters back to fp16/bf16.
            for p, fp32_p in zip(params_with_grad, fp32_params_with_grad):
                if p.dtype == torch.float16 or p.dtype == torch.bfloat16:
                    p.copy_(fp32_p)

    def _init_param(
        self,
        param: Tensor,
        param_group: Dict[str, Any],
        params_with_grad: List[Tensor],
        fp32_params_with_grad: List[Tensor],
        fp32_grads: List[Tensor],
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

        params_with_grad.append(param)

        # fp32 parameter
        if param.dtype == torch.float16 or param.dtype == torch.bfloat16:
            fp32_param = param.float()
        else:
            fp32_param = param

        fp32_params_with_grad.append(fp32_param)

        # fp32 grad
        if grad.dtype == torch.float16 or grad.dtype == torch.bfloat16:
            fp32_grads.append(grad.float())
        else:
            fp32_grads.append(grad)

        state = cast(Dict[str, Tensor], self.state[param])  # type: ignore[index]

        if len(state) == 0:
            if param_group["capturable"] or param_group["impl"] == "fused":
                device = param.device
            else:
                device = None

            # Step counter.
            state["step"] = torch.zeros((), device=device, dtype=torch.float32)

            # Exponential moving average of gradient values.
            state["exp_avg"] = torch.zeros_like(fp32_param)

            # Exponential moving average of squared gradient values.
            state["exp_avg_sq"] = torch.zeros_like(fp32_param)

            if amsgrad:
                state["max_exp_avg_sq"] = torch.zeros_like(fp32_param)

        steps.append(state["step"])

        exp_avgs.append(state["exp_avg"])

        exp_avg_sqs.append(state["exp_avg_sq"])

        if amsgrad:
            max_exp_avg_sqs.append(state["max_exp_avg_sq"])
