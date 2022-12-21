from typing import TYPE_CHECKING, List, NamedTuple

from .utils import RoundRobin

if TYPE_CHECKING:
    import torch


class Seq2SeqBatch(NamedTuple):
    source: "torch.Tensor"
    target: "torch.Tensor"
    num_tokens: int
    # TODO: add batch source


class Seq2SeqStr(NamedTuple):
    source: str
    target: str
    predicted: str


class Text2TextBatch(NamedTuple):
    src: List[str]
    tgt: List[str]


class Audio2Text(NamedTuple):
    source: str
    target: str
    predicted: str


__all__ = ["RoundRobin", "Seq2SeqBatch", "Seq2SeqStr"]
