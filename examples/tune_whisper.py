import itertools
from datetime import timedelta
from typing import Any, Iterable, List

import torch
import transformers  # type: ignore
from torchtnt.framework.state import State

import fairseq2
import fairseq2.dataloader.huggingface
import fairseq2.distributed
import fairseq2.tasks
from fairseq2.callbacks import Metrics
from fairseq2.dataloader import Seq2SeqBatch, Seq2SeqStr
from fairseq2.dataloader.huggingface import AsrDataloader
from fairseq2.distributed import Env
from fairseq2.generate import SpeechToTextTokenizer

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
            metrics["wer"] = fairseq2.callbacks.WER()
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
        target = self.tokenizer.decode_batch(data.target)
        # Use HF beam search, assuming we have an HF model
        # TODO: Can we use fairseq2 beamsearch ?
        predicted_tokens = self.model.generate(  # type: ignore
            inputs=data.source,
            num_beams=1,
            max_length=int(data.target.size(1) * 1.2),
        )
        predicted = self.tokenizer.decode_batch(predicted_tokens.squeeze(1))
        # TODO: upload some generation to W&B
        return [
            Seq2SeqStr(*x)
            for x in itertools.zip_longest(
                ["<audio>" for _ in data.target], target, predicted
            )
        ]


task = AsrTask


def model(
    version: str,
    env: Env,
    fp16: bool = True,
) -> Any:
    m = transformers.WhisperForConditionalGeneration.from_pretrained(version)
    m = m.to(env.device)
    if fp16:
        m = m.half()
    return m


def _load_dataset(
    name: str,
    lang: str,
    split: str,
    tokenizer: SpeechToTextTokenizer,
    env: Env,
    batch_duration: timedelta,
    fp16: bool,
) -> AsrDataloader:
    return AsrDataloader(
        name,
        lang,
        split,
        tokenizer,
        batch_duration=batch_duration,
        env=env,
        dtype=torch.float16 if fp16 else torch.float32,
    )


def train_data(
    langs: str,
    tokenizer: SpeechToTextTokenizer,
    env: Env,
    dataset: str = "mozilla-foundation/common_voice_11_0",
    batch_duration: timedelta = timedelta(seconds=10),
    fp16: bool = True,
) -> Iterable[Seq2SeqBatch]:
    _langs = langs.split(",")
    if len(_langs) > 1:
        return fairseq2.dataloader.RoundRobin(
            [
                _load_dataset(
                    dataset, lang, "train", tokenizer, env, batch_duration, fp16
                )
                for lang in _langs
            ]
        )
    else:
        return _load_dataset(
            dataset, _langs[0], "train", tokenizer, env, batch_duration, fp16
        )


def valid_data(
    langs: str,
    tokenizer: SpeechToTextTokenizer,
    env: Env,
    dataset: str = "mozilla-foundation/common_voice_11_0",
    batch_duration: timedelta = timedelta(seconds=10),
    fp16: bool = True,
) -> AsrDataloader:
    _langs = langs.split(",")
    # Only evaluate on the first lang
    return _load_dataset(
        dataset, _langs[0], "validation", tokenizer, env, batch_duration, fp16
    )


def tokenizer(version: str) -> SpeechToTextTokenizer:
    return fairseq2.generate.SpeechToTextTokenizer.from_pretrained(version)
