"""
Fine tune Whisper model on an HuggingFace dataset.

Example cli:

python examples/tune_whisper.py help
python examples/tune_whisper.py -w /checkpoint/$USER/fairseq2/whisper langs=hi
"""

import itertools
from datetime import timedelta
from typing import Any, Iterable, List

import torch
from torchtnt.framework.state import State
from transformers import (  # type: ignore[import]
    SequenceFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

import fairseq2
import fairseq2.data.huggingface
import fairseq2.tasks
from fairseq2.cli import Env
from fairseq2.data import Seq2SeqBatch, Seq2SeqStr
from fairseq2.data.huggingface import AsrDataloader
from fairseq2.data.text import Tokenizer
from fairseq2.generate import HfTokenizer
from fairseq2.metrics import Metrics

REQUIREMENTS = [
    "datasets>=2.6.1",
    "git+https://github.com/huggingface/transformers",
    "librosa",
    "evaluate>=0.30",
    "jiwer",
    "gradio",
]


class AsrTask(fairseq2.tasks.Seq2Seq):
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

        target = token_decoder(data.target)
        # Use HF beam search, assuming we have an HF model
        # TODO: Can we use fairseq2 beamsearch ?
        predicted_tokens = self.model.generate(  # type: ignore
            inputs=data.source,
            num_beams=1,
            max_length=int(data.target.size(1) * 1.2),
        )
        predicted = token_decoder(predicted_tokens.squeeze(1))
        # TODO: upload some generation to W&B
        return [
            Seq2SeqStr(*x)
            for x in itertools.zip_longest(
                ["<audio>" for _ in data.target], target, predicted
            )
        ]


task = AsrTask


def model(
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
    langs: str,
    feature_extractor: "SequenceFeatureExtractor",
    tokenizer: Tokenizer,
    env: Env,
    dataset: str = "mozilla-foundation/common_voice_11_0",
    batch_duration: timedelta = timedelta(seconds=10),
    fp16: bool = True,
) -> Iterable[Seq2SeqBatch]:
    _langs = langs.split(",")
    if len(_langs) > 1:
        return fairseq2.data.huggingface.RoundRobin(
            [
                _load_dataset(
                    dataset,
                    lang,
                    "train",
                    feature_extractor,
                    tokenizer,
                    env,
                    batch_duration,
                    fp16,
                )
                for lang in _langs
            ]
        )
    else:
        return _load_dataset(
            dataset,
            _langs[0],
            "train",
            feature_extractor,
            tokenizer,
            env,
            batch_duration,
            fp16,
        )


def valid_data(
    langs: str,
    feature_extractor: "SequenceFeatureExtractor",
    tokenizer: Tokenizer,
    env: Env,
    dataset: str = "mozilla-foundation/common_voice_11_0",
    batch_duration: timedelta = timedelta(seconds=10),
    fp16: bool = True,
) -> AsrDataloader:
    _langs = langs.split(",")
    # Only evaluate on the first lang
    return _load_dataset(
        dataset,
        _langs[0],
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
