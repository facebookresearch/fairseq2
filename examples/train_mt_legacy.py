import enum
import functools
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, List

import torch
import torchtnt.framework
import torchtnt.framework.callbacks
import torchtnt.utils

import fairseq2.callbacks
import fairseq2.dataloader.huggingface
import fairseq2.dataloader.legacy
import fairseq2.distributed
import fairseq2.hub
import fairseq2.nn
import fairseq2.optim.lr_scheduler
from fairseq2.dataloader import Seq2SeqBatch
from fairseq2.generate.tokenizer import DictTokenizer
from fairseq2.nn import transformer
from fairseq2.tasks import TranslationTask
from fairseq2.typing import Device

log = logging.getLogger(__name__)

BATCH_FIRST = True


class Mode(enum.Enum):
    TRAINING = 0
    EVALUATE = 1
    INFERENCE = 2
    TORCHHUB = 3


def train(
    task: TranslationTask,
    env: fairseq2.distributed.Env,
    lang_pairs: List[str],
    data_dir: Path,
    reset: bool,
    eval_freq: int,
    wandb_project: str,
) -> None:
    task.__hubstate__.gen_hubconf(env.workdir)

    data_dir = Path(
        f"/private/home/guw/github/fairseq/data-bin/iwslt14.tokenized.{lang_pairs[0]}"
    )
    load_data = functools.partial(
        fairseq2.dataloader.legacy.BilingualDataloader,
        data_dir,
        device=env.device,
    )

    if len(lang_pairs) > 1:
        train: Iterable[Seq2SeqBatch] = fairseq2.dataloader.RoundRobin(
            [load_data(*pair.split("-"), "train") for pair in lang_pairs],
            batch_first=BATCH_FIRST,
        )
    else:
        train = load_data(*lang_pairs[0].split("-"), "train")
    # Only evaluate on the first lang pair
    valid = load_data(*lang_pairs[0].split("-"), "valid")

    callbacks = fairseq2.callbacks.default_callbacks(
        task, env, wandb_project=wandb_project, reload_model=not reset
    )
    train_state = torchtnt.framework.init_fit_state(
        train, valid, evaluate_every_n_steps=eval_freq
    )
    torchtnt.framework.fit(train_state, task, callbacks=callbacks)


def evaluate(
    task: TranslationTask,
    env: fairseq2.distributed.Env,
    lang_pairs: List[str],
    batch_size: int,
) -> None:
    load_data = functools.partial(
        fairseq2.dataloader.huggingface.NllbDataLoader,
        tokenizer=task.tokenizer,
        batch_size=batch_size,
        env=env,
    )
    callbacks = fairseq2.callbacks.default_callbacks(task, env, reload_model=True)

    for lang_pair in lang_pairs:
        # Only evaluate on the first lang pair
        valid = load_data(*lang_pairs[0].split("-"), "valid")
        eval_state = torchtnt.framework.init_fit_state([], valid)
        # eval_state = torchtnt.framework.init_eval_state(dataloader=valid)
        log.info(f"Evaluating on {lang_pair} ...")
        torchtnt.framework.evaluate(eval_state, task, callbacks=callbacks)


def inference(
    task: TranslationTask,
    lang_pairs: List[str],
    batch_size: int,
    beam_search: str,
) -> None:
    import fairseq2.inference

    for lang_pair in lang_pairs:
        src, tgt = lang_pair.split("-")
        fairseq2.inference.inference(
            task,
            batch_size=batch_size,
            src_bos=src,
            tgt_bos=tgt,
            **json.loads(beam_search),
        )


DEFAULT_BEAM_SEARCH = json.dumps({"beam_size": 5, "max_len": 128, "unk_penalty": 1.0})


class LegacyBuilder(transformer.TransformerBuilder):
    # Just override some defaults.
    def __init__(
        self,
        num_tokens: int,
        num_enc_attn_heads: int = 4,
        num_dec_attn_heads: int = 4,
        max_seq_len: int = 1024,
        ffn_inner_dim: int = 1024,
        **kwargs: Any,
    ):
        super().__init__(
            num_tokens,
            num_enc_attn_heads=num_enc_attn_heads,
            num_dec_attn_heads=num_dec_attn_heads,
            ffn_inner_dim=ffn_inner_dim,
            max_seq_len=max_seq_len,
            **kwargs,
        )

    def build(self) -> transformer.Transformer:
        """Build on CPU then push to GPU. This allows to use the CPU RNG seed, like fairseq1."""
        device = self.device
        dtype = self.dtype
        try:
            self.device = Device("cpu")
            self.dtype = torch.float32
            self._fct_kwargs["device"] = self.device
            self._fct_kwargs["dtype"] = self.dtype

            model = super().build()
            model.to(device=device, dtype=dtype)
            return model
        finally:
            self.device = device
            self.dtype = dtype
            self._fct_kwargs["device"] = self.device
            self._fct_kwargs["dtype"] = self.dtype


def main(
    workdir: Path,
    langs: str = "de-en",
    small: bool = False,
    wandb_project: str = "nllb/fairseq2",
    batch_size: int = 16,
    partition: str = "debug",
    eval_freq: int = 10_000,
    num_gpus: int = 1,
    reset: bool = False,
    mode: Mode = Mode.TRAINING,
    beam_search: str = DEFAULT_BEAM_SEARCH,
) -> TranslationTask:
    workdir = Path(str(workdir).format(langs=langs))
    workdir.mkdir(exist_ok=True)
    # TODO: we should allow downloading the first time
    # os.environ["HF_DATASETS_OFFLINE"] = "1"
    # os.environ.update(os.environ)

    env = fairseq2.distributed.init(workdir, partition, num_gpus, one_file=True)
    torchtnt.utils.seed(1)
    torch.cuda.manual_seed(1)

    data_dir = Path("/private/home/guw/github/fairseq/data-bin/iwslt14.tokenized.de-en")

    lang_pairs = langs.split(",")
    src_langs = set(pair.split("-")[0] for pair in lang_pairs)
    tgt_langs = set(pair.split("-")[1] for pair in lang_pairs)

    src_0 = lang_pairs[0].split("-")[0]
    tokenizer = DictTokenizer.from_fairseq_dict_txt(data_dir / f"dict.{src_0}.txt")
    for lang in sorted(src_langs | tgt_langs):
        tokenizer.add_special_token(lang)

    builder = LegacyBuilder(
        tokenizer.vocab_size(),
        tokenizer.PAD,
        batch_first=BATCH_FIRST,
        dropout_p=0,
        device=env.device,
    )

    task = TranslationTask(builder, tokenizer, env.device)
    if mode == Mode.TRAINING:
        train(task, env, lang_pairs, data_dir, reset, eval_freq, wandb_project)
    elif mode == Mode.EVALUATE:
        evaluate(task, env, lang_pairs, batch_size)
    elif mode == Mode.INFERENCE:
        inference(task, lang_pairs, batch_size, beam_search)
    elif mode == Mode.TORCHHUB:
        return task
    else:
        raise Exception(f"Unknown enum value: {mode}")

    sys.exit(0)


if __name__ == "__main__":
    import func_argparse

    func_argparse.single_main(main)
