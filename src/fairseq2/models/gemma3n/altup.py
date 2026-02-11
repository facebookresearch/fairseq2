# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn import LayerNorm
from fairseq2.nn.projection import Linear


@final
class Gemma3nAltUp(Module):
    """Alternating Updates (AltUp) for Gemma3n.

    AltUp wraps transformer layers to enable sparse computation via predict/correct.
    The predict step modifies the input to the layer, and the correct step
    propagates the output to sparsely updated dimensions.

    Reference: https://proceedings.neurips.cc/paper_files/paper/2023/file/f2059277ac6ce66e7e5543001afa8bb5-Paper-Conference.pdf
    """

    model_dim: int
    num_inputs: int
    active_idx: int
    correct_output_scale: Parameter
    correction_coefs: Linear
    prediction_coefs: Linear
    modality_router: Linear
    router_norm: LayerNorm
    coef_clip: float | None

    def __init__(
        self,
        model_dim: int,
        num_inputs: int,
        *,
        active_idx: int = 0,
        router_norm: LayerNorm,
        coef_clip: float | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim: Model dimensionality.
        :param num_inputs: Number of AltUp inputs (typically 4).
        :param active_idx: Index of the actively computed version (default 0).
        :param router_norm: Layer normalization for router inputs.
        :param coef_clip: Optional clipping value for prediction/correction coefficients.
        """
        super().__init__()

        self.model_dim = model_dim
        self.num_inputs = num_inputs
        self.active_idx = active_idx
        self.coef_clip = coef_clip

        self.correct_output_scale = Parameter(torch.zeros(model_dim, device=device, dtype=dtype))

        self.correction_coefs = Linear(
            num_inputs, num_inputs, bias=False, device=device, dtype=dtype
        )
        self.prediction_coefs = Linear(
            num_inputs, num_inputs * num_inputs, bias=False, device=device, dtype=dtype
        )
        self.modality_router = Linear(
            model_dim, num_inputs, bias=False, device=device, dtype=dtype
        )

        self.router_norm = router_norm
        self.register_buffer(
            "router_input_scale",
            torch.tensor(model_dim**-1.0, device=device, dtype=dtype),
            persistent=False,
        )

    def compute_router_modalities(self, x: Tensor) -> Tensor:
        """Compute routing modalities from input tensor."""
        router_inputs = self.router_norm(x) * self.router_input_scale
        routed = self.modality_router(router_inputs)
        return torch.tanh(routed.float()).type_as(x)

    @override
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Predict step: predicts layer output using trainable map.

        :param hidden_states: 4D tensor [num_inputs, batch, seq_len, model_dim].
        :returns: 4D predictions tensor [num_inputs, batch, seq_len, model_dim].
        """
        modalities = self.compute_router_modalities(hidden_states[self.active_idx])

        if self.training and self.coef_clip is not None:
            self.prediction_coefs.weight.data.clamp_(-self.coef_clip, self.coef_clip)

        # Project and reshape to get coefficient matrices
        all_coefs = (
            self.prediction_coefs(modalities)
            .reshape(*modalities.shape[:-1], self.num_inputs, self.num_inputs)
            .permute(0, 1, 3, 2)
        )

        # Apply predictions: matmul with hidden states
        predictions = torch.matmul(hidden_states.permute(1, 2, 3, 0), all_coefs)
        predictions = predictions.permute(3, 0, 1, 2)  # Back to [num_inputs, B, S, M]
        predictions += hidden_states  # Residual connection
        return predictions.contiguous().type_as(hidden_states)

    def correct(self, predictions: Tensor, activated: Tensor) -> Tensor:
        """Correct step: corrects predictions using activated output.

        :param predictions: 4D predictions [num_inputs, batch, seq_len, model_dim].
        :param activated: 3D activated output [batch, seq_len, model_dim].
        :returns: 4D corrected tensor [num_inputs, batch, seq_len, model_dim].
        """
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.active_idx]  # Prediction error
        innovation = innovation.repeat(self.num_inputs, 1, 1, 1)  # Match shape

        if self.training and self.coef_clip is not None:
            self.correction_coefs.weight.data.clamp_(-self.coef_clip, self.coef_clip)

        # Compute correction coefficients
        all_coefs = self.correction_coefs(modalities) + 1.0
        all_coefs = all_coefs.permute(2, 0, 1).unsqueeze(-1)  # Shape for broadcasting

        # Apply corrections
        corrected = torch.mul(innovation, all_coefs)
        corrected += predictions  # Add back predictions
        return corrected.contiguous().type_as(activated)

    def scale_corrected_output(self, corrected: Tensor) -> Tensor:
        """Scales the corrected output tensor.

        :param corrected: 3D tensor [batch, seq_len, model_dim].
        :returns: Scaled tensor [batch, seq_len, model_dim].
        """
        return (corrected.type_as(self.correct_output_scale) * self.correct_output_scale).type_as(corrected)
