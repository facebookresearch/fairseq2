from dataclasses import dataclass
from typing import List, Optional, final

import math
import torch
from torch import Tensor
import torch.nn as nn

from abc import ABC, abstractmethod
from fairseq2.nn import Embedding, Projection
from torch.nn import Dropout
from torch.nn.functional import linear
from torch.nn.parameter import Parameter

from fairseq2.typing import DataType, Device


@dataclass
class LoRAConfig:
    r: int
    alpha: float
    dropout_p: float
    keys: List[str]


class LoRALayer(ABC):
    def __init__(self, config: LoRAConfig):
        self.r = config.r
        self.alpha = config.alpha
        self.scaling = self.alpha / self.r
        self.dropout_p = config.dropout_p

    @property
    @abstractmethod
    def wrapped_module(self) -> nn.Module:
        ...

    @abstractmethod
    def merge(self) -> None:
        ...

    @abstractmethod
    def unmerge(self) -> None:
        ...


@final
class LoRALinear(Projection, LoRALayer):

    weight: Parameter
    bias: Optional[Parameter]
    lora_A: Parameter
    lora_B: Parameter
    dropout: Optional[Dropout]
    skip_init: bool
    merged: bool

    def __init__(
            self,
            wrapped: Projection,
            config: LoRAConfig,
            skip_init: bool = False,
            device: Device=None,
            dtype: DataType=None
        ) -> None:
        Projection.__init__(self, wrapped.input_dim, wrapped.output_dim)

        LoRALayer.__init__(self, config)

        self.weight = Parameter(
            wrapped.weight, requires_grad=False
        )

        if wrapped.bias is not None:
            self.bias = Parameter(
                wrapped.bias, requires_grad=False
            )
        else:
            self.register_module("bias", None)

        self.lora_A = Parameter(
            torch.empty((self.r, self.input_dim), device=device, dtype=dtype)
        )

        self.lora_B = Parameter(
            torch.empty((self.output_dim, self.r), device=device, dtype=dtype)
        )

        if self.dropout_p > 0.:
            self.dropout = Dropout(self.dropout_p)

        else:
            self.register_module('dropout', None)

        self.merged = False

        self.skip_init = skip_init

        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        if self.merged:
            return linear(x, self.weight, self.bias)

        else:
            h1 = linear(x, self.weight, self.bias)

            if self.dropout is not None:
                h2 = linear(self.dropout(x), self.lora_B @ self.lora_A * self.scaling)

            else:
                h2 = linear(x, self.lora_B @ self.lora_A * self.scaling)

            return h1 + h2
    
    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if not self.skip_init:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

    def wrapped_module(self) -> nn.Module:
        return self

    def merge(self) -> None:
        if self.merged:
            return
        self.weight.data += torch.matmul(self.lora_B.data, self.lora_A.data) * self.scaling

        self.merged = True
    
    def unmerge(self) -> None:
        if not self.merged:
            return
        else:
            self.weight.data -= torch.matmul(self.lora_B.data, self.lora_A.data) * self.scaling

            self.merged = False
