from typing import TYPE_CHECKING, NamedTuple

from .utils import RoundRobin

if TYPE_CHECKING:
    import torch


class Batch(NamedTuple):
    source: "torch.Tensor"
    target: "torch.Tensor"
    num_tokens: int
    # TODO: add batch statisticts


class Translation(NamedTuple):
    source: str
    target: str
    predicted: str


__all__ = ["Batch", "RoundRobin", "Translation"]
