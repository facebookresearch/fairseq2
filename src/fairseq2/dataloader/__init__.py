from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Sequence

from fairseq2.data import StringLike

from .utils import RoundRobin as RoundRobin

if TYPE_CHECKING:
    import torch


class Seq2SeqBatch(NamedTuple):
    source: "torch.Tensor"
    src_seq_lens: "torch.Tensor"
    target: "torch.Tensor"
    tgt_seq_lens: "torch.Tensor"
    metadata: Sequence[Dict[str, Any]] = []


class Seq2SeqStr(NamedTuple):
    source: StringLike
    target: StringLike
    predicted: StringLike


class Text2TextBatch(NamedTuple):
    src: Sequence[str]
    tgt: Sequence[str]


class Audio2Text(NamedTuple):
    source: str
    target: str
    predicted: str
