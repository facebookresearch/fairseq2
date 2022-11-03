from pathlib import Path
from typing import Any, Dict, List

import sentencepiece
import torch
from torch import Tensor


class Tokenizer:

    UNK = 0
    BOS = 1
    EOS = 2
    PAD = 3

    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode_batch(self, sentences: List[str]) -> Tensor:
        raise NotImplementedError

    def decode_batch(self, tokens: Tensor) -> List[str]:
        raise NotImplementedError


class SpmTokenizer(Tokenizer):
    @staticmethod
    def from_file(file: Path, batch_first: bool = True) -> "SpmTokenizer":
        spm = sentencepiece.SentencePieceProcessor()
        spm.load(str(file))
        return SpmTokenizer(spm, batch_first=batch_first)

    def __init__(self, spm: sentencepiece.SentencePieceProcessor, batch_first: bool):
        self.spm = spm
        self._vocab_size = int(spm.GetPieceSize())
        self.batch_first = batch_first

        # Typically spm models already have unk/bos/eos assigned to 0/1/2.
        self.UNK = spm.unk_id() if spm.unk_id() >= 0 else self._add_special_token()
        self.BOS = spm.bos_id() if spm.bos_id() >= 0 else self._add_special_token()
        self.EOS = spm.eos_id() if spm.eos_id() >= 0 else self._add_special_token()
        self.PAD = spm.pad_id() if spm.pad_id() >= 0 else self._add_special_token()

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "spm": self.spm.serialized_model_proto(),
            "batch_first": self.batch_first,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        spm = sentencepiece.SentencePieceProcessor()
        spm.load_from_serialized_proto(state["spm"])
        SpmTokenizer.__init__(self, spm, state["batch_first"])

    def vocab_size(self) -> int:
        return self._vocab_size

    def _add_special_token(self) -> int:
        # TODO edit the spm model itself
        n = self._vocab_size
        self._vocab_size += 1
        return n

    def encode_batch(self, sentences: List[str]) -> Tensor:
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
        return _make_batch(tokens, self.PAD, self.batch_first)

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
        return self.spm.decode(tokens[:first_pad])  # type: ignore


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

    if not batch_first:
        return res.transpose(1, 0)
    return res
