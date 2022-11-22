from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import sentencepiece
import torch
from torch import Tensor

if TYPE_CHECKING:
    import fairseq


class Tokenizer:

    UNK = 0
    BOS = 1
    EOS = 2
    PAD = 3

    def __init__(self, batch_first: bool):
        self.batch_first = batch_first
        self.special_tokens: Dict[str, int] = {}

    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode_batch(self, sentences: List[str], bos: int = -1) -> Tensor:
        raise NotImplementedError

    def decode_batch(self, tokens: Tensor) -> List[str]:
        raise NotImplementedError

    def num_tokens(self, tokens: Tensor) -> int:
        return int((tokens != self.PAD).sum())

    def add_special_token(self, token: str) -> int:
        if token in self.special_tokens:
            return self.special_tokens[token]

        n = self.vocab_size()
        self.special_tokens[token] = n
        return n


@dataclass(frozen=True)
class TokenMeta:
    """Description of a tokenizer vocabulary."""

    vocab_size: int
    UNK: int
    BOS: int
    EOS: int
    PAD: int

    @staticmethod
    def from_tokenizer(tokenizer: Tokenizer) -> "TokenMeta":
        return TokenMeta(
            vocab_size=tokenizer.vocab_size(),
            UNK=tokenizer.UNK,
            BOS=tokenizer.BOS,
            EOS=tokenizer.EOS,
            PAD=tokenizer.PAD,
        )


class SpmTokenizer(Tokenizer):
    @staticmethod
    def from_file(file: Path, batch_first: bool = True) -> "SpmTokenizer":
        spm = sentencepiece.SentencePieceProcessor()
        spm.load(str(file))
        return SpmTokenizer(spm, batch_first=batch_first)

    def __init__(self, spm: sentencepiece.SentencePieceProcessor, batch_first: bool):
        super().__init__(batch_first)
        self.spm = spm
        self.batch_first = batch_first

        # Spm models already have unk/bos/eos assigned to 0/1/2.
        assert spm.unk_id() >= 0
        self.UNK = spm.unk_id()
        assert spm.bos_id() >= 0
        self.BOS = spm.bos_id()
        assert spm.eos_id() >= 0
        self.EOS = spm.eos_id()
        self.PAD = spm.pad_id() if spm.pad_id() >= 0 else self.add_special_token("PAD")

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "spm": self.spm.serialized_model_proto(),
            "special_tokens": self.special_tokens,
            "batch_first": self.batch_first,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        spm = sentencepiece.SentencePieceProcessor()
        spm.load_from_serialized_proto(state["spm"])
        SpmTokenizer.__init__(self, spm, state["batch_first"])
        self.special_tokens = state["special_tokens"]

    def vocab_size(self) -> int:
        return int(self.spm.GetPieceSize()) + len(self.special_tokens)

    def encode_batch(self, sentences: List[str], bos: int = -1) -> Tensor:
        tokens: List[List[int]] = [
            self.spm.encode_as_ids(
                # TODO: the sampling should be configurable
                sentence,
                add_bos=True,
                add_eos=True,
                enable_sampling=True,
            )
            for sentence in sentences
        ]
        bos = self.BOS if bos < 0 else bos
        return _make_batch(tokens, self.PAD, self.batch_first, special_bos=bos)

    def decode_batch(self, tokens: Tensor) -> List[str]:
        # Replace special tokens with BOS.
        tokens[tokens >= self.spm.GetPieceSize()] = self.BOS
        if self.batch_first:
            return [self._decode(tokens[i, :].tolist()) for i in range(tokens.size(0))]
        else:
            return [self._decode(tokens[:, i].tolist()) for i in range(tokens.size(1))]

    def _decode(self, tokens: List[int]) -> str:
        # TODO: encode
        if tokens[-1] == self.PAD:
            first_pad = tokens.index(self.PAD)
        else:
            first_pad = len(tokens)
        return self.spm.decode(tokens[:first_pad])  # type: ignore


class DictTokenizer(Tokenizer):
    """Dict and spaces based tokenizer like in legacy Fairseq."""

    @staticmethod
    def from_fairseq_dict_txt(file: Path, batch_first: bool = True) -> "DictTokenizer":
        import fairseq.data

        src_dict = fairseq.data.Dictionary.load(str(file))
        return DictTokenizer(src_dict, batch_first)

    def __init__(self, vocab: "fairseq.data.Dictionary", batch_first: bool):
        super().__init__(batch_first)
        self.vocab = vocab
        self.BOS = vocab.bos_index
        self.PAD = vocab.pad_index
        self.EOS = vocab.eos_index
        self.UNK = vocab.unk_index

    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode_batch(self, sentences: List[str], bos: int = -1) -> Tensor:
        tokens: List[List[int]] = [
            self.vocab.encode_line(sentence) for sentence in sentences
        ]
        bos = self.BOS if bos < 0 else bos
        return _make_batch(tokens, self.PAD, self.batch_first, special_bos=bos)

    def decode_batch(self, tokens: Tensor) -> List[str]:
        if self.batch_first:
            return [self._decode(tokens[i, :].tolist()) for i in range(tokens.size(0))]
        else:
            return [self._decode(tokens[:, i].tolist()) for i in range(tokens.size(1))]

    def _decode(self, tokens: List[int]) -> str:
        if tokens[-1] == self.PAD:
            first_pad = tokens.index(self.PAD)
        else:
            first_pad = len(tokens)
        return self.vocab.string(tokens[:first_pad])  # type: ignore


# TODO do this in C++
def _make_batch(
    values: List[List[int]],
    pad_id: int,
    batch_first: bool,
    pad_to_length: int = 0,
    pad_to_multiple: int = 1,
    batch_size: int = 0,
    left_pad: bool = False,
    # TODO: use int16 when possible
    dtype: torch.dtype = torch.int64,
    special_bos: int = 0,
) -> Tensor:
    """Convert a list of token-index list into a padded 2d tensor.

    Note: eos/bos are supposed to be already added by sentencepiece
    """
    size = max(len(v) for v in values)
    size = max(size, pad_to_length)
    if size % pad_to_multiple != 0:
        size = (size - size % pad_to_multiple) + pad_to_multiple

    batch_size = max(len(values), batch_size)
    res = torch.zeros((batch_size, size), dtype=dtype).fill_(pad_id)
    for i, v in enumerate(values):
        if left_pad:
            res[i, size - len(v) :] = torch.tensor(v, dtype=dtype)
        else:
            res[i, : len(v)] = torch.tensor(v, dtype=dtype)
    res[:, 0] = special_bos
    if not batch_first:
        return res.transpose(1, 0)
    return res
