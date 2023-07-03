"""
Fine tune Whisper model on an HuggingFace dataset.

Example cli:

fairseq2 train examples/cli/finetune_whisper.py help
fairseq2 train examples/cli/finetune_whisper.py -w /checkpoint/$USER/fairseq2/whisper lang=hi
"""

import datetime
import functools
import itertools
import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Optional, Tuple

import torch
from torch import Tensor
from torchtnt.framework.state import State
from transformers import (  # type: ignore[import]
    SequenceFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

import fairseq2
from fairseq2.cli.api import Env, Seq2Seq, Seq2SeqStr
from fairseq2.data import Collater, StringLike
from fairseq2.data.text import TokenDecoder, TokenEncoder, Tokenizer, VocabularyInfo
from fairseq2.metrics import Metrics
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.typing import DataType, Device

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer  # type: ignore[import]

try:
    import datasets  # type: ignore[import]
except ImportError:
    print(
        "`finetune_whisper` requires HuggingFace Datasets library. You can install it with: `pip install datasets`"
    )
    raise

REQUIREMENTS = [
    "datasets>=2.6.1",
    "git+https://github.com/huggingface/transformers",
    "librosa",
    "evaluate>=0.30",
    "jiwer",
    "gradio",
]


log = logging.getLogger(__name__)


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
        device: Optional[Device] = None,
        pin_memory: bool = False,
    ) -> TokenEncoder:
        return functools.partial(self._encode, device=device)  # type: ignore[return-value]

    def create_decoder(self) -> TokenDecoder:
        return self._decode  # type: ignore[return-value]

    def _encode(self, sentence: StringLike, device: Optional[Device]) -> Tensor:
        t = self.tokenizer(sentence, return_tensors="pt", padding=True).to(device)

        return t["input_ids"]  # type: ignore[no-any-return]

    def _decode(self, token_indices: Tensor) -> List[StringLike]:
        return self.tokenizer.decode(token_indices, skip_special_tokens=True)  # type: ignore[no-any-return]


class AsrBatch:
    def __init__(self, sampling_rate: int):
        # Actually those are numpy array
        self.audio_features: List[Tensor] = []
        self.sentences: List[str] = []
        self.batch_len: int = 0
        self.sampling_rate: int = sampling_rate

    def append(self, audio: Tensor, sentence: str) -> int:
        self.audio_features.append(audio)
        self.sentences.append(sentence)
        self.batch_len += len(audio)
        return self.batch_len


class AsrDataloader(Iterable[Seq2SeqBatch]):
    def __init__(
        self,
        name: str,
        language: str,
        split: str,
        feature_extractor: Any,
        tokenizer: Tokenizer,
        *,
        batch_size: int = 0,
        batch_duration: Optional[datetime.timedelta] = None,
        env: Env,
        dtype: DataType,
    ):
        """
        Load ASR dataset from HF hub, and yields examples compatible with Seq2Seq task.

        - feature_extractor: This should follow the API of transformers.SequenceFeatureExtractor
        https://huggingface.co/transformers/v4.4.2/main_classes/feature_extractor.html#transformers.SequenceFeatureExtractor
        """
        self.feature_extractor = feature_extractor
        self.sampling_rate = feature_extractor.sampling_rate
        self.env = env
        self.device = env.device
        self.dtype = dtype

        self.pad_idx = tokenizer.vocab_info.pad_idx

        self.token_encoder = tokenizer.create_encoder(
            task="transcribe",
            lang=language,
            mode="target",
            device=env.device,
            pin_memory=True,
        )

        if batch_size > 0:
            self.batch_size = batch_size
        else:
            assert (
                batch_duration is not None
            ), "Need to specify either batch_size or batch_duration"
            self.batch_size = int(batch_duration.total_seconds() * self.sampling_rate)

        self.data = (
            datasets.load_dataset(name, language, split=split).remove_columns(
                [
                    "accent",
                    "age",
                    "client_id",
                    "down_votes",
                    "gender",
                    "locale",
                    "path",
                    "segment",
                    "up_votes",
                ]
            )
            # Downsampling
            .cast_column("audio", datasets.Audio(sampling_rate=(self.sampling_rate)))
        )
        self.epoch = 0

        self.collater = Collater(self.pad_idx)

    def _finalize_batch(self, batch: AsrBatch) -> Seq2SeqBatch:
        device = self.env.device

        target_tokens = []

        # TODO: expose parallel_for
        for s in batch.sentences:
            target_tokens.append(self.token_encoder(s))

        target = self.collater(target_tokens)
        source, src_seq_lens = self._encode_audio(batch.audio_features)
        return Seq2SeqBatch(
            source,
            src_seq_lens,
            target.to(device),
            self._num_tokens(target),
        )

    def _encode_audio(self, raw_speech: List[torch.Tensor]) -> Tuple[Tensor, Tensor]:
        features = [
            self.feature_extractor(
                raw_speech=example,
                return_tensors="pt",
                sampling_rate=self.sampling_rate,
            )["input_features"]
            for example in raw_speech
        ]
        src_seq_lens = torch.tensor([f.size(-1) for f in features], device=self.device)
        source = self.feature_extractor.pad(
            {"input_features": [f.squeeze(0) for f in features]},
            padding=True,
            pad_to_multiple_of=128,
            return_attention_mask=False,
            return_tensors="pt",
        )["input_features"]

        assert isinstance(source, Tensor)
        assert source.shape[0] == len(raw_speech)
        source = source.to(device=self.device, dtype=self.dtype)
        return source, src_seq_lens

    def _num_tokens(self, tokens: Tensor) -> Tensor:
        return (tokens != self.pad_idx).sum(dim=-1)

    def __iter__(self) -> Iterator[Seq2SeqBatch]:
        batch = AsrBatch(self.sampling_rate)
        env = self.env

        for i, sample in enumerate(self.data):
            if i % env.world_size != env.global_rank:
                continue

            audio = sample["audio"]
            # We already resampled earlier
            assert audio["sampling_rate"] == self.sampling_rate
            batch_len = batch.append(audio["array"], sample["sentence"])

            if batch_len >= self.batch_size:
                yield self._finalize_batch(batch)
                batch = AsrBatch(self.sampling_rate)
        if batch.audio_features:
            yield self._finalize_batch(batch)

        log.info(f"End of epoch {self.epoch}, with {i * self.batch_size} samples")
        self.epoch += 1


class AsrTask(Seq2Seq):
    def init_metrics(self, mode: str) -> Metrics:
        metrics = super().init_metrics(mode)
        self.best_metric = "wer" if self.eval_gen else "loss"

        if mode == "eval":
            metrics["wer"] = fairseq2.metrics.WER()
        return metrics

    def eval_step(self, state: State, data: Seq2SeqBatch) -> Any:
        super().eval_step(state, data)

        assert state.eval_state is not None

        if self.eval_gen:
            # TODO: upgrade once torchtnt has builtin support for metric
            wer = state.eval_state.metrics["wer"]  # type: ignore
            generations = self.generate_batch(data)
            for asr in generations:
                wer.update(asr.predicted, reference=asr.target)

            return generations

    def generate_batch(self, data: Seq2SeqBatch) -> List[Seq2SeqStr]:
        token_decoder = self.tokenizer.create_decoder()

        target = token_decoder(data.target_seqs)
        # Use HF beam search, assuming we have an HF model
        # TODO: Can we use fairseq2 beamsearch ?
        predicted_tokens = self.model.generate(  # type: ignore
            inputs=data.source_seqs,
            num_beams=1,
            max_length=int(data.target_seqs.size(1) * 1.2),
        )
        predicted = token_decoder(predicted_tokens.squeeze(1))
        # TODO: upload some generation to W&B
        return [
            Seq2SeqStr(*x)
            for x in itertools.zip_longest(
                ["<audio>" for _ in data.target_seqs], target, predicted
            )
        ]


task = AsrTask


def module(
    env: Env,
    version: str = "openai/whisper-small",
    fp16: bool = True,
) -> Any:
    """Chose which huggingface model to use"""
    m = WhisperForConditionalGeneration.from_pretrained(version)
    m = m.to(env.device)
    if fp16:
        m = m.half()
    return m


def _load_dataset(
    name: str,
    lang: str,
    split: str,
    feature_extractor: "SequenceFeatureExtractor",
    tokenizer: Tokenizer,
    env: Env,
    batch_duration: timedelta,
    fp16: bool,
) -> AsrDataloader:
    return AsrDataloader(
        name,
        lang,
        split,
        feature_extractor,
        tokenizer,
        batch_duration=batch_duration,
        env=env,
        dtype=torch.float16 if fp16 else torch.float32,
    )


def train_data(
    lang: str,
    feature_extractor: "SequenceFeatureExtractor",
    tokenizer: Tokenizer,
    env: Env,
    dataset: str = "mozilla-foundation/common_voice_11_0",
    batch_duration: timedelta = timedelta(seconds=10),
    fp16: bool = True,
) -> Iterable[Seq2SeqBatch]:
    return _load_dataset(
        dataset,
        lang,
        "train",
        feature_extractor,
        tokenizer,
        env,
        batch_duration,
        fp16,
    )


def valid_data(
    lang: str,
    feature_extractor: "SequenceFeatureExtractor",
    tokenizer: Tokenizer,
    env: Env,
    dataset: str = "mozilla-foundation/common_voice_11_0",
    batch_duration: timedelta = timedelta(seconds=10),
    fp16: bool = True,
) -> AsrDataloader:
    return _load_dataset(
        dataset,
        lang,
        "validation",
        feature_extractor,
        tokenizer,
        env,
        batch_duration,
        fp16,
    )


def processor(version: str = "openai/whisper-small") -> "WhisperProcessor":
    return WhisperProcessor.from_pretrained(version)


def tokenizer(processor: "WhisperProcessor") -> Tokenizer:
    return HfTokenizer(processor.tokenizer)


def feature_extractor(processor: "WhisperProcessor") -> "SequenceFeatureExtractor":
    return processor.feature_extractor


# This is important, it tells torch.hub how to reload our "task"
fairseq2_hub = fairseq2.cli.fairseq2_hub


if __name__ == "__main__":
    import fairseq2.cli.commands

    fairseq2.cli.commands.main(__file__)
