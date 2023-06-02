from typing import Optional, Sequence

import torch
from overrides import final

from fairseq2.data.string import StringLike
from fairseq2.data.text.dict import DictDecoder, DictEncoder, DictModel
from fairseq2.data.text.tokenizer import (
    TokenDecoder,
    TokenEncoder,
    Tokenizer,
    VocabularyInfo,
)


@final
class DictTokenizer(Tokenizer):
    """Represents a simple tokenizer that splits on space and replace word by token found in dict.
    Warning: only int64 as token index works for now. TODO: Implement integer types with more options."""

    model: DictModel
    dim: int

    def __init__(self, dim: int, vocab: Sequence[StringLike]) -> None:
        self.dim = dim
        self.model = DictModel(vocab)
        vocab_info = VocabularyInfo(
            self.model.vocab_size,
            self.model.unk_idx,
            self.model.bos_idx,
            self.model.eos_idx,
            self.model.pad_idx,
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
    ) -> "TokenEncoder":
        """Create a token encoder.

        The valid arguments for the ``task``, ``lang``, and ``mode`` parameters
        are implementation specific. Refer to concrete ``Tokenizer`` subclasses
        for more information.

        :param task:
            An optional implementation-specific task such as 'translation' or
            'transcription' for which to generate token indices.
        :param lang:
            An optional identifier indicating the language of generated token
            indices. Typically used by multilingual tokenizers for
            distinguishing between different source and target languages.
        :param mode:
            An optional implementation-specific mode in which to generate token
            indices. Typically used by translation tasks to indicate whether the
            encoding is done for source or target sentences.
        :param batch_size:
            If the number of sentences to encode is less than ``batch_size``,
            the output will be padded.
        :param device:
            The device on which to initialize token indices.
        :param pin_memory:
            If ``True``, uses pinned memory before copying token indices to the
            target device. (only supported by CUDA devices)
        :param dtype:
            The integral data type of generated token indices.
        :param disabled_parallelism:
            If ``True``, disables parallelism and uses the calling thread only.
        """
        return DictEncoder(self.model, self.dim)

    def create_decoder(self) -> "TokenDecoder":
        """Create a token decoder."""
        return DictDecoder(self.model)
