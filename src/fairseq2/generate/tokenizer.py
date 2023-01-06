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

    def add_special_token(self, token: str, idx: int = -1) -> int:
        if token in self.special_tokens:
            n = self.special_tokens[token]
            if idx >= 0:
                assert (
                    idx == n
                ), f"{token} is already assigned to {n}, can't remap to {idx}"
            return n

        n = idx if idx >= 0 else self.vocab_size()
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
    def from_file(
        file: Path, batch_first: bool = True, _pad_shift_hack: bool = False
    ) -> "SpmTokenizer":
        spm = sentencepiece.SentencePieceProcessor()
        spm.load(str(file))
        return SpmTokenizer(
            spm, batch_first=batch_first, _pad_shift_hack=_pad_shift_hack
        )

    def __init__(
        self,
        spm: sentencepiece.SentencePieceProcessor,
        batch_first: bool,
        sampling: bool = False,
        _pad_shift_hack: bool = False,
    ):
        super().__init__(batch_first)
        self.spm = spm
        self.batch_first = batch_first
        self.sampling = sampling
        # HACK to reproduce Fairseq1 behavior
        # Fairseq1 is not using tokens returned by spm, but convert them to string then back to index.
        # The results is to shift each word token by one.
        self._pad_shift_hack = _pad_shift_hack

        # Typically UNK = 0, BOS = 1, EOS = 2, PAD = VOCAB_SIZE
        # With _pad_shift_hack: PAD = 0, UNK =1, BOS = 2, EOS = 3
        self.UNK = self.add_special_token("<UNK>", spm.unk_id() + _pad_shift_hack)
        self.BOS = self.add_special_token("<BOS>", spm.bos_id() + _pad_shift_hack)
        self.EOS = self.add_special_token("<EOS>", spm.eos_id() + _pad_shift_hack)
        self.PAD = self.add_special_token("<PAD>", 0 if _pad_shift_hack else -1)

    def state_dict(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["spm"] = self.spm.serialized_model_proto()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        spm = sentencepiece.SentencePieceProcessor()
        spm.load_from_serialized_proto(state["spm"])
        state["spm"] = spm
        self.__dict__.update(state)

    def vocab_size(self) -> int:
        # unk, bos, and eos are both in spm.GetPieceSize() and special_tokens
        return int(self.spm.GetPieceSize()) - 3 + len(self.special_tokens)

    def encode_batch(self, sentences: List[str], bos: int = -1) -> Tensor:
        tokens: List[List[int]] = [
            self.spm.encode_as_ids(
                # TODO: the sampling should be configurable
                sentence,
                add_bos=True,
                add_eos=True,
                enable_sampling=self.sampling,
            )
            for sentence in sentences
        ]
        bos = self.BOS if bos < 0 else bos
        return _make_batch(
            tokens,
            self.PAD,
            self.batch_first,
            rewrite_bos=bos,
            shift_tokens=1 if self._pad_shift_hack else 0,
        )

    def decode_batch(self, tokens: Tensor) -> List[str]:
        # Replace special tokens with BOS.
        # TODO: allow to print special tokens (again, we should probably modify the underlying spm)
        tokens = tokens.clone().detach()
        tokens[tokens >= self.spm.GetPieceSize()] = self.BOS
        # SPM doesn't now PAD.
        tokens[tokens == self.PAD] = self.EOS
        if self._pad_shift_hack:
            tokens = tokens - 1
        if not self.batch_first:
            tokens = tokens.transpose(1, 0)

        return [self._decode(tokens[i, :].tolist()) for i in range(tokens.size(0))]

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
        self.UNK = self.add_special_token("<UNK>", vocab.unk_index)
        self.BOS = self.add_special_token("<BOS>", vocab.bos_index)
        self.EOS = self.add_special_token("<EOS>", vocab.eos_index)
        self.PAD = self.add_special_token("<PAD>", vocab.pad_index)

    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode_batch(self, sentences: List[str], bos: int = -1) -> Tensor:
        tokens: List[List[int]] = [
            self.vocab.encode_line(sentence, append_eos=True) for sentence in sentences
        ]
        bos = self.BOS if bos < 0 else bos
        # Fairseq is adding BOS after tokenization. Let's add it now.
        return _make_batch(tokens, self.PAD, self.batch_first, prepend_bos=bos)

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
    rewrite_bos: int = -1,
    prepend_bos: int = -1,
    shift_tokens: int = 0,
) -> Tensor:
    """Convert a list of token-index list into a padded 2d tensor.

    Note: eos/bos are supposed to be already added by sentencepiece
    """
    size = max(len(v) for v in values)
    size = max(size, pad_to_length)

    offset = 0
    if prepend_bos >= 0:
        assert not left_pad, "TODO: left_pad isn't compatible with prepend_bos."
        assert rewrite_bos < 0, "Can't use both rewrite_bos and prepend_bos."
        size += 1
        offset = 1
        rewrite_bos = prepend_bos

    if size % pad_to_multiple != 0:
        size = (size - size % pad_to_multiple) + pad_to_multiple

    batch_size = max(len(values), batch_size)
    res = torch.zeros((batch_size, size), dtype=dtype).fill_(pad_id)
    for i, v in enumerate(values):
        if left_pad:
            # TODO: make left_pad work with prepend_bos (who is using left_pad ?)
            res[i, size - len(v) :] = torch.tensor(v, dtype=dtype) + shift_tokens
        else:
            res[i, offset : len(v) + offset] = (
                torch.tensor(v, dtype=dtype) + shift_tokens
            )
    if rewrite_bos >= 0:
        assert not left_pad, "TODO: left_pad isn't compatible with rewrite_bos."
        res[:, 0] = rewrite_bos
    if not batch_first:
        return res.transpose(1, 0)
    return res
