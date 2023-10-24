"""Beamformer module."""
from typing import Sequence, Tuple, Union

import torch
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

EPS = torch.finfo(torch.double).eps


def new_complex_like(
    ref: Union[torch.Tensor, ComplexTensor],
    real_imag: Tuple[torch.Tensor, torch.Tensor],
):
    if isinstance(ref, ComplexTensor):
        return ComplexTensor(*real_imag)
    elif is_torch_complex_tensor(ref):
        return torch.complex(*real_imag)
    else:
        raise ValueError(
            "Please update your PyTorch version to 1.9+ for complex support."
        )


def is_torch_complex_tensor(c):
    return not isinstance(c, ComplexTensor) and torch.is_complex(c)


def is_complex(c):
    return isinstance(c, ComplexTensor) or is_torch_complex_tensor(c)


def to_complex(c):
    # Convert to torch native complex
    if isinstance(c, ComplexTensor):
        c = c.real + 1j * c.imag
        return c
    elif torch.is_complex(c):
        return c
    else:
        return torch.view_as_complex(c)

def complex_norm(
    c: Union[torch.Tensor, ComplexTensor], dim=-1, keepdim=False
) -> torch.Tensor:
    if not is_complex(c):
        raise TypeError("Input is not a complex tensor.")
    if is_torch_complex_tensor(c):
        return torch.norm(c, dim=dim, keepdim=keepdim)
    else:
        if dim is None:
            return torch.sqrt((c.real**2 + c.imag**2).sum() + EPS)
        else:
            return torch.sqrt(
                (c.real**2 + c.imag**2).sum(dim=dim, keepdim=keepdim) + EPS
            )
