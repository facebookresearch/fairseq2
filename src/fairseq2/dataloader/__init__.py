from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Sequence

from .utils import RoundRobin as RoundRobin

if TYPE_CHECKING:
    import torch


class Seq2SeqBatch(NamedTuple):
    source: "torch.Tensor"
    target: "torch.Tensor"
    num_tokens: int
    metadata: Sequence[Dict[str, Any]] = []


class Seq2SeqStr(NamedTuple):
    source: str
    target: str
    predicted: str


class Text2TextBatch(NamedTuple):
    src: Sequence[str]
    tgt: Sequence[str]


class Audio2Text(NamedTuple):
    source: str
    target: str
    predicted: str
