import math
from abc import ABC, abstractmethod
from functools import reduce

import torch
import torch.nn.functional as F
from packaging.version import parse as V

from espnet2.enh.layers.complex_utils import complex_norm, is_complex, new_complex_like
from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")

EPS = torch.finfo(torch.get_default_dtype()).eps


class FrequencyDomainMSE(FrequencyDomainLoss):
    def __init__(
        self,
        compute_on_mask=False,
        mask_type="IBM",
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        if name is not None:
            _name = name
        elif compute_on_mask:
            _name = f"MSE_on_{mask_type}"
        else:
            _name = "MSE_on_Spec"
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self._compute_on_mask = compute_on_mask
        self._mask_type = mask_type

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    def forward(self, ref, inf) -> torch.Tensor:
        """time-frequency MSE loss.

        Args:
            ref: (Batch, T, F) or (Batch, T, C, F)
            inf: (Batch, T, F) or (Batch, T, C, F)
        Returns:
            loss: (Batch,)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)

        diff = ref - inf
        if is_complex(diff):
            mseloss = diff.real**2 + diff.imag**2
        else:
            mseloss = diff**2
        if ref.dim() == 3:
            mseloss = mseloss.mean(dim=[1, 2])
        elif ref.dim() == 4:
            mseloss = mseloss.mean(dim=[1, 2, 3])
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )
        return mseloss

class SNRLoss(TimeDomainLoss):
    def __init__(
        self,
        eps=EPS,
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        _name = "snr_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self.eps = float(eps)

    def forward(self, ref: torch.Tensor, inf: torch.Tensor) -> torch.Tensor:
        # the return tensor should be shape of (batch,)

        noise = inf - ref

        snr = 20 * (
            torch.log10(torch.norm(ref, p=2, dim=1).clamp(min=self.eps))
            - torch.log10(torch.norm(noise, p=2, dim=1).clamp(min=self.eps))
        )
        return -snr