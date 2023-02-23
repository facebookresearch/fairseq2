from typing import TYPE_CHECKING, List, Sequence

import torch
from torch import Tensor

from .tokenizer import Tokenizer, _make_batch

if TYPE_CHECKING:
    import transformers  # type: ignore


class HfTokenizer(Tokenizer):
    def __init__(self, tokenizer: "transformers.PreTrainedTokenizer"):
        super().__init__()
        self.tokenizer = tokenizer

        self.UNK = self.add_special_token("<UNK>", tokenizer.unk_token_id)
        self.BOS = self.add_special_token("<BOS>", tokenizer.bos_token_id)
        self.EOS = self.add_special_token("<EOS>", tokenizer.eos_token_id)
        self.PAD = self.add_special_token("<PAD>", tokenizer.pad_token_id)

        special_tokens = tokenizer._additional_special_tokens
        special_ids = tokenizer.additional_special_tokens_ids
        for tok, idx in zip(special_tokens, special_ids):
            self.add_special_token(tok, idx)

    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size) + len(self.special_tokens)

    def encode_batch(self, sentences: Sequence[str], bos: int = -1) -> Tensor:
        tokens: List[List[int]] = [
            self.tokenizer.encode(sentence) for sentence in sentences
        ]
        bos = self.BOS if bos < 0 else bos
        return _make_batch(tokens, self.PAD, prepend_bos=bos)

    def decode_batch(self, tokens: Tensor) -> List[str]:
        return [self._decode(tokens[i, :].tolist()) for i in range(tokens.size(0))]

    def _decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)  # type: ignore


class SpeechToTextTokenizer(HfTokenizer):
    @classmethod
    def from_pretrained(
        cls,
        name: str = "openai/whisper-small",
    ) -> "SpeechToTextTokenizer":
        import transformers

        # TODO: there are probably other tokenizer than whisper.
        processor = transformers.WhisperProcessor.from_pretrained(name)
        return SpeechToTextTokenizer(
            processor.tokenizer,
            processor.feature_extractor,
        )

    def __init__(
        self,
        tokenizer: "transformers.PreTrainedTokenizer",
        feature_extractor: "transformers.WhisperFeatureExtractor",
    ):
        # Note: Audio features are typically 2-D: (batch, feature, time)
        # what does non batch-first mean in this case ?
        super().__init__(tokenizer)
        self.feature_extractor = feature_extractor
        self.sampling_rate = feature_extractor.sampling_rate

    def encode_audio(
        self,
        raw_speech: List[torch.Tensor],
        *,
        sampling_rate: int,
        pad_to_multiple: int = 128,
    ) -> Tensor:
        # TODO: resample if needed
        assert self.sampling_rate == sampling_rate
        features = self.feature_extractor(
            raw_speech=raw_speech,
            pad_to_multiple_of=pad_to_multiple,
            return_attention_mask=False,
            return_tensors="pt",
            sampling_rate=self.sampling_rate,
        )["input_features"]

        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == len(raw_speech)
        return features

    def decode_batch(self, tokens: Tensor) -> List[str]:
        if tokens.dtype != torch.long:
            return self.decode_audio(tokens)

        return super().decode_batch(tokens)

    def decode_audio(self, tokens: Tensor) -> List[str]:
        return ["<audio>"] * tokens.shape[0]
