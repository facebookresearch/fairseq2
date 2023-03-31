import functools
from typing import List, Optional, Sequence, Union

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer  # type: ignore[import]

from fairseq2.data import StringLike
from fairseq2.data.text import TokenDecoder, TokenEncoder, Tokenizer, VocabularyInfo


# TODO: This is a temporary solution. I plan to write a more complete
# implementation that converts HG tokenizer json files containing BPE, unigram
# vocabs on the fly to in-memory protobuf dicts that we can feed to our own
# SentencePiece API.
class HfTokenizer(Tokenizer):
    def __init__(self, tokenizer: "PreTrainedTokenizer"):
        self.tokenizer = tokenizer

        vocab_size = int(self.tokenizer.vocab_size)

        vocab_info = VocabularyInfo(
            size=vocab_size,
            unk_idx=tokenizer.unk_token_id,
            bos_idx=tokenizer.bos_token_id,
            eos_idx=tokenizer.eos_token_id,
            pad_idx=tokenizer.pad_token_id,
        )

        super().__init__(vocab_info)

    def create_encoder(
        self,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        pin_memory: bool = False,
        dtype: torch.dtype = torch.int32,
        disable_parallelism: bool = False,
    ) -> TokenEncoder:
        return functools.partial(self._encode, device=device)  # type: ignore[return-value]

    def create_decoder(self) -> TokenDecoder:
        return self._decode  # type: ignore[return-value]

    def _encode(
        self,
        sentences: Union[StringLike, Sequence[StringLike]],
        device: Optional[torch.device],
    ) -> Tensor:
        t = self.tokenizer(sentences, return_tensors="pt", padding=True).to(device)

        return t["input_ids"]  # type: ignore[no-any-return]

    def _decode(self, token_indices: Tensor) -> List[StringLike]:
        return self.tokenizer.decode(token_indices, skip_special_tokens=True)  # type: ignore[no-any-return]
