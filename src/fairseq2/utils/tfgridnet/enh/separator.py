from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from fairseq2.utils.tfgridnet.enh.tcn import TemporalConvNet

def is_torch_complex_tensor(c):
    return not isinstance(c, ComplexTensor) and torch.is_complex(c)


def is_complex(c):
    return isinstance(c, ComplexTensor) or is_torch_complex_tensor(c)

class AbsSeparator(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:
        raise NotImplementedError

    def forward_streaming(
        self,
        input_frame: torch.Tensor,
        buffer=None,
    ):
        raise NotImplementedError

    @property
    @abstractmethod
    def num_spk(self):
        raise NotImplementedError

class RNNSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        rnn_type: str = "blstm",
        num_spk: int = 2,
        predict_noise: bool = False,
        nonlinear: str = "sigmoid",
        layer: int = 3,
        unit: int = 512,
        dropout: float = 0.0,
    ):
        """RNN Separator

        Args:
            input_dim: input feature dimension
            rnn_type: string, select from 'blstm', 'lstm' etc.
            bidirectional: bool, whether the inter-chunk RNN layers are bidirectional.
            num_spk: number of speakers
            predict_noise: whether to output the estimated noise signal
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
            layer: int, number of stacked RNN layers. Default is 3.
            unit: int, dimension of the hidden state.
            dropout: float, dropout ratio. Default is 0.
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

        self.rnn = RNN(
            idim=input_dim,
            elayers=layer,
            cdim=unit,
            hdim=unit,
            dropout=dropout,
            typ=rnn_type,
        )

        num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
        self.linear = torch.nn.ModuleList(
            [torch.nn.Linear(unit, input_dim) for _ in range(num_outputs)]
        )

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model
                NOTE: not used in this model

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """

        # if complex spectrum,
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input

        x, ilens, _ = self.rnn(feature, ilens)

        masks = []

        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            masks.append(y)

        if self.predict_noise:
            *masks, mask_noise = masks

        masked = [input * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input * mask_noise

        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk

    def forward_streaming(self, input_frame: torch.Tensor, states=None):
        # input_frame # B, 1, N

        # if complex spectrum,
        if is_complex(input_frame):
            feature = abs(input_frame)
        else:
            feature = input_frame

        ilens = torch.ones(feature.shape[0], device=feature.device)

        x, _, states = self.rnn(feature, ilens, states)

        masks = []

        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            masks.append(y)

        if self.predict_noise:
            *masks, mask_noise = masks

        masked = [input_frame * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input * mask_noise

        return masked, states, others

class TCNSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        predict_noise: bool = False,
        layer: int = 8,
        stack: int = 3,
        bottleneck_dim: int = 128,
        hidden_dim: int = 512,
        kernel: int = 3,
        causal: bool = False,
        norm_type: str = "gLN",
        nonlinear: str = "relu",
    ):
        """Temporal Convolution Separator

        Args:
            input_dim: input feature dimension
            num_spk: number of speakers
            predict_noise: whether to output the estimated noise signal
            layer: int, number of layers in each stack.
            stack: int, number of stacks
            bottleneck_dim: bottleneck dimension
            hidden_dim: number of convolution channel
            kernel: int, kernel size.
            causal: bool, defalut False.
            norm_type: str, choose from 'BN', 'gLN', 'cLN'
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.tcn = TemporalConvNet(
            N=input_dim,
            B=bottleneck_dim,
            H=hidden_dim,
            P=kernel,
            X=layer,
            R=stack,
            C=num_spk + 1 if predict_noise else num_spk,
            norm_type=norm_type,
            causal=causal,
            mask_nonlinear=nonlinear,
        )

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model
                NOTE: not used in this model

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """
        # if complex spectrum
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input
        B, L, N = feature.shape

        feature = feature.transpose(1, 2)  # B, N, L

        masks = self.tcn(feature)  # B, num_spk, N, L
        masks = masks.transpose(2, 3)  # B, num_spk, L, N
        if self.predict_noise:
            *masks, mask_noise = masks.unbind(dim=1)  # List[B, L, N]
        else:
            masks = masks.unbind(dim=1)  # List[B, L, N]

        masked = [input * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input * mask_noise

        return masked, ilens, others

    def forward_streaming(self, input_frame: torch.Tensor, buffer=None):
        # input_frame: B, 1, N

        B, _, N = input_frame.shape

        receptive_field = self.tcn.receptive_field

        if buffer is None:
            buffer = torch.zeros((B, receptive_field, N), device=input_frame.device)

        buffer = torch.roll(buffer, shifts=-1, dims=1)
        buffer[:, -1, :] = input_frame[:, 0, :]

        masked, ilens, others = self.forward(buffer, None)

        masked = [m[:, -1, :].unsqueeze(1) for m in masked]

        return masked, buffer, others

    @property
    def num_spk(self):
        return self._num_spk
