import datetime
import itertools
from pathlib import Path
from typing import Any, Iterable, List

import torch
import torchtnt.framework as tnt
import transformers  # type: ignore
from torchtnt.framework.state import State

import fairseq2
import fairseq2.dataloader.huggingface
import fairseq2.distributed
import fairseq2.tasks
from fairseq2.callbacks import Metrics
from fairseq2.dataloader import Seq2SeqBatch, Seq2SeqStr
from fairseq2.typing import DataType, Device

REQUIREMENTS = [
    "datasets>=2.6.1",
    "git+https://github.com/huggingface/transformers",
    "librosa",
    "evaluate>=0.30",
    "jiwer",
    "gradio",
]


class AsrTask(fairseq2.tasks.Seq2Seq):
    def __init__(
        self,
        builder: "WhisperBuilder",
        tokenizer: fairseq2.generate.SpeechToTextTokenizer,
        device: Device,
    ):
        super().__init__(builder, tokenizer, device)  # type: ignore

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


# TODO: make helper builder for transformers.* models.
class WhisperBuilder:
    def __init__(self, name: str, device: Device, dtype: DataType):
        self.name = name
        self.max_seq_len = 0
        self.config: Any = None
        self.device = device
        self.dtype = dtype

    def __call__(self) -> Any:
        # Ideally we should be able to load directly on the correct device.
        model = transformers.WhisperForConditionalGeneration.from_pretrained(self.name)
        model = model.to(self.device)
        if self.dtype is torch.float16:
            model = model.half()
        self.max_seq_len = model.config.max_target_positions
        self.config = model.config
        return model


def main(
    workdir: Path,
    dataset: str = "mozilla-foundation/common_voice_11_0",
    langs: str = "hi",
    batch_duration: float = 10,
    pretrained: str = "openai/whisper-small",
    partition: str = "debug",
    num_gpus: int = 1,
    wandb_project: str = "nllb/tune_whisper",
    eval_freq: int = 10_000,
    reset: bool = False,
) -> None:
    """
    Fine tune whisper on the given dataset (only tested with mozilla-foundation/common_voice_11_0)

    workdir: where to save the training snapshot
    dataset: name of HF ASR dataset to use
    langs: lang to select
    batch_duration: duration of batch in seconds
    pretrained: name of pretrained model
    partition: partition to use (needed for num_gpus > 1)
    num_gpus: GPUs to use, will use Slurm if num_gpus > 1
    wandb_project: project to log experiments too. Disable wandb by setting to ""
    eval_freq: freq to evaluate, the model is pretty slow for evaluation
    reset: ignore existing snapshots and restart from pretrained model
    """
    workdir = Path(str(workdir).format(langs=langs))
    workdir.mkdir(exist_ok=True)
    env = fairseq2.distributed.init(workdir, partition, num_gpus)

    tokenizer = fairseq2.generate.SpeechToTextTokenizer.from_pretrained(
        "openai/whisper-small"
    )

    task = AsrTask(
        WhisperBuilder(pretrained, device=env.device, dtype=torch.float16),
        tokenizer,
        env.device,
    )

    def load_data(lang: str, split: str) -> Iterable[Seq2SeqBatch]:
        return fairseq2.dataloader.huggingface.AsrDataloader(
            dataset,
            lang,
            split,
            tokenizer,
            batch_duration=datetime.timedelta(seconds=batch_duration),
            env=env,
            dtype=torch.float16,
        )

    _langs = langs.split(",")
    if len(_langs) > 1:
        train: Iterable[Seq2SeqBatch] = fairseq2.dataloader.RoundRobin(
            [load_data(lang, "train") for lang in _langs],
        )
    else:
        train = load_data(_langs[0], "train")
    # Only evaluate on the first lang pair
    valid = load_data(_langs[0], "validation")

    train_state = tnt.init_fit_state(train, valid, evaluate_every_n_steps=eval_freq)

    callbacks = fairseq2.callbacks.default_callbacks(
        task, env, reload_model=not reset, wandb_project=wandb_project
    )

    tnt.fit(train_state, task, callbacks=callbacks)


if __name__ == "__main__":
    import func_argparse

    func_argparse.single_main(main)
