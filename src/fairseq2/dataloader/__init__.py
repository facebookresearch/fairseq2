from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Protocol, Sequence

from fairseq2.data import StringLike

from .utils import RoundRobin as RoundRobin

if TYPE_CHECKING:
    from torch import Tensor


class Seq2SeqBatch(NamedTuple):
    source: "Tensor"
    src_seq_lens: "Tensor"
    target: "Tensor"
    tgt_seq_lens: "Tensor"
    metadata: Sequence[Dict[str, Any]] = []


class Seq2SeqStr(NamedTuple):
    source: StringLike
    target: StringLike
    predicted: StringLike


class Text2TextBatch(NamedTuple):
    src: Sequence[str]
    tgt: Sequence[str]
