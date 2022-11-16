import itertools
import logging
import os
import sys
import typing as tp
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import torcheval.metrics
import torchtnt.runner
import torchtnt.runner.callbacks
import torchtnt.utils
from torch import Tensor
from torchtnt.loggers.logger import MetricLogger
from torchtnt.runner.state import State
from torchtnt.runner.unit import EvalUnit, PredictUnit, TrainUnit

import fairseq2.callbacks
import fairseq2.dataloader.huggingface
import fairseq2.distributed
import fairseq2.nn
import fairseq2.optim.lr_scheduler
from fairseq2.dataloader import Batch
from fairseq2.generate import BeamSearch, SpmTokenizer, Tokenizer, generate
from fairseq2.nn import transformer

REQUIREMENTS = [
    "sentencepiece",
    # TODO: make optional
    "wandb",
]

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# For now, required for generation
BATCH_FIRST = True


class Translation(tp.NamedTuple):
    source: str
    target: str
    predicted: str


class MachineTranslationTask(TrainUnit[Batch], EvalUnit[Batch], PredictUnit[Batch]):
    def __init__(
        self,
        model: transformer.Transformer,
        tokenizer: Tokenizer,
        logger: MetricLogger,
    ):
        # TODO: take the spm as input
        super().__init__()
        # initialize module & optimizer

        self.model = model
        self.tokenizer = tokenizer
        # TODO: we should take optim, scheduler as inputs
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1.0,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.0001,
        )
        self.lr_scheduler = fairseq2.optim.lr_scheduler.InverseSquareRootLR(
            self.optimizer, lr=5e-4
        )

        self.train_loss = torcheval.metrics.Sum()
        self.train_tokens = torcheval.metrics.Sum()
        self.eval_loss = torcheval.metrics.Sum()
        self.eval_tokens = torcheval.metrics.Sum()
        self.log_frequency_steps = 10
        self.predict_frequency_steps = 1000
        self.lr_frequency_steps = 1
        self.logger = logger

    def replicated_keys(self) -> List[str]:
        return ["logger/**"]

    def train_step(self, state: State, data: Batch) -> None:
        assert state.train_state
        seed = state.train_state.progress.num_steps_completed
        torchtnt.utils.seed(seed)

        steps = state.train_state.progress.num_steps_completed + 1
        loss = self.loss(data)

        # TODO: allow accumulating gradients over several batch
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if steps % self.lr_frequency_steps == 0:
            self.lr_scheduler.step()

        # update metrics & logs
        self.train_loss.update(loss.detach().cpu())
        self.train_tokens.update(torch.tensor(data.num_tokens))

        if steps % self.log_frequency_steps == 0:
            self.log_metrics(state, "train")

    def on_train_start(self, state: State) -> None:
        pass

    def on_train_end(self, state: State) -> None:
        self.logger.close()

    def loss(self, data: Batch) -> Tensor:
        # TODO: nll loss requires longs ? Why ?
        # net_output = self.model.forward(data.source, data.target)
        net_output = self.model.extract_features(data.source, data.target[:, :-1])
        lprobs = F.log_softmax(net_output, dim=-1).transpose(2, 1)
        loss = F.nll_loss(
            lprobs, data.target[:, 1:], reduction="sum", ignore_index=self.tokenizer.PAD
        )
        return loss

    def log_metrics(self, state: State, prefix: str) -> None:
        assert state.train_state is not None
        step = state.train_state.progress.num_steps_completed
        if prefix == "train":
            loss = self.train_loss.compute()
            self.train_loss.reset()
            num_tokens = self.train_tokens.compute()
            self.train_tokens.reset()
        elif prefix == "eval":
            loss = self.eval_loss.compute()
            self.eval_loss.reset()
            num_tokens = self.eval_tokens.compute()
            self.eval_tokens.reset()
        else:
            raise ValueError(f"unknown stage: {prefix}")
        loss = float(loss.detach().cpu()) / int(num_tokens.detach().cpu())
        # This isn't really accurate.
        # This is the perplexity of the average loss, not the average perplexity.
        # But AFAICT Fairseq also computed it this way.
        try:
            ppl = round(2**loss, 3)
        except OverflowError:
            ppl = float("inf")

        # TODO use tnt.logger
        metrics = {"step": step, f"{prefix}/loss": loss, f"{prefix}/ppl": ppl}
        if prefix == "train":
            metrics[f"{prefix}/lr"] = self.lr_scheduler.get_last_lr()[0]
        self.logger.log_dict(metrics, step)

    def state_dict(self) -> Dict[str, Any]:
        return {k: v.state_dict() for k, v in self.app_state().items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for k, v in state_dict.items():
            self.__getattr__(k).load_state_dict(v)

    def state_dict_for_inference(self) -> Dict[str, Any]:
        # TODO: return a minimized stated dict for inference.
        ...

    def on_eval_start(self, state: State) -> None:
        pass

    def on_eval_end(self, state: State) -> None:
        self.log_metrics(state, "eval")

    def eval_step(self, state: State, data: Batch) -> None:
        assert state.eval_state

        self.eval_loss.update(self.loss(data).detach().cpu())
        self.eval_tokens.update(torch.tensor(data.num_tokens))

        eval_step = state.eval_state.progress.num_steps_completed_in_epoch
        if eval_step == 0:
            # Always translate the same sample from the eval data loader
            self.log_translated_sample(state, data)

    def log_translated_sample(self, state: State, data: Batch) -> None:
        assert state.train_state

        # TODO: move to callback
        translations = self.translate_batch(data)
        if torchtnt.utils.get_global_rank() != 0:
            return

        step = state.train_state.progress.num_steps_completed
        if isinstance(self.logger, fairseq2.callbacks.WandbLogger):
            import wandb

            self.logger.prepare()
            table = wandb.Table(
                columns=["source", "target", "predicted"], data=translations
            )
            self.logger._wandb.log({"train/predictions": table}, step=step)

        log.info(f"Translations at step {step}")
        for row in translations:
            log.info(f"Source: {row.source}")
            log.info(f"Target: {row.target}")
            log.info(f"Predicted: {row.predicted}")

    def predict_step(self, state: State, data: Batch) -> List[Translation]:
        return self.translate_batch(data)

    def translate_batch(self, data: Batch) -> List[Translation]:
        source = self.tokenizer.decode_batch(data.source)
        target = self.tokenizer.decode_batch(data.target)
        search = BeamSearch(self.tokenizer, beam_size=2, max_len=128)
        predicted_tokens = generate(self.model, search, data.source, top=1)
        print(f"{predicted_tokens[0] = }")
        predicted = self.tokenizer.decode_batch(predicted_tokens.squeeze(1))
        return list(
            map(
                lambda x: Translation(*x),
                itertools.zip_longest(source, target, predicted),
            )
        )


def main(
    workdir: Path,
    spm_path: Path = Path("/private/home/kevinheffernan/nllb.200.models/laser2.spm"),
    src_lang: str = "cat_Latn",
    tgt_lang: str = "eng_Latn",
    small: bool = False,
    wandb_project: str = "nllb/fairseq2",
    batch_size: int = 16,
    partition: str = "debug",
    num_gpus: int = 1,
    reset: bool = False,
) -> None:
    workdir = Path(str(workdir).format(src=src_lang, tgt=tgt_lang))
    workdir.mkdir(exist_ok=True)
    # TODO: we should allow downloading the first time
    # os.environ["HF_DATASETS_OFFLINE"] = "1"
    # os.environ.update(os.environ)

    env = fairseq2.distributed.init(workdir, partition, num_gpus, one_file=True)
    torchtnt.utils.seed(0)
    torch.cuda.manual_seed(0)

    # import fairseq2.dataloader.legacy
    # data_dir = Path("/private/home/guw/github/fairseq/data-bin/iwslt14.tokenized.de-en")
    # tokenizer = DictTokenizer.from_fairseq_dict_txt(data_dir / f"dict.{src_lang}.txt")
    # builder = transformer.TransformerBuilder(
    #     tokenizer.vocab_size(), tokenizer.PAD, batch_first=True, dropout_p=0, device=env.device
    # )
    # model = builder.build()
    # model.to(env.device)
    # train_dataloader = fairseq2.dataloader.legacy.BilingualDataloader(
    #     data_dir,
    #     src=src_lang,
    #     tgt=tgt_lang,
    #     split="train",
    #     device=env.device,
    # )
    # valid_dataloader = fairseq2.dataloader.legacy.BilingualDataloader(
    #     data_dir,
    #     src=src_lang,
    #     tgt=tgt_lang,
    #     split="valid",
    #     device=env.device,
    # )

    tokenizer = SpmTokenizer.from_file(spm_path, batch_first=BATCH_FIRST)
    builder = transformer.TransformerBuilder(
        tokenizer.vocab_size(),
        tokenizer.PAD,
        batch_first=True,
        dropout_p=0,
        device=env.device,
    )
    model = builder.build()
    train_dataloader = fairseq2.dataloader.huggingface.NllbDataLoader(
        train=True,
        src=src_lang,
        tgt=tgt_lang,
        tokenizer=tokenizer,
        batch_size=batch_size,
        global_rank=env.global_rank,
        world_size=env.world_size,
        device=env.device,
    )
    valid_dataloader = fairseq2.dataloader.huggingface.NllbDataLoader(
        train=False,
        src=src_lang,
        tgt=tgt_lang,
        tokenizer=tokenizer,
        batch_size=batch_size,
        global_rank=env.global_rank,
        world_size=env.world_size,
        device=env.device,
    )

    if wandb_project:
        logger: MetricLogger = fairseq2.callbacks.WandbLogger(wandb_project, {})
    else:
        logger = fairseq2.callbacks.StdoutLogger()

    task = MachineTranslationTask(model, tokenizer, logger)

    train_state = torchtnt.runner.init_fit_state(
        train_dataloader,
        valid_dataloader,
        max_epochs=None,
        max_steps=1_000_000,
        evaluate_every_n_steps=10_000,
    )
    callbacks = [
        # Synchronize GC runs across all nodes
        torchtnt.runner.callbacks.GarbageCollector(step_interval=1),
        torchtnt.runner.callbacks.TorchSnapshotSaver(
            str(workdir),
            save_every_n_train_steps=10_000,
            replicated=task.replicated_keys(),
        ),
    ]
    if not reset:
        callbacks.append(
            fairseq2.callbacks.TorchSnapshotLoader(
                str(workdir), replicated=task.replicated_keys()
            )
        )
    if os.isatty(sys.stdout.fileno()):
        callbacks.append(fairseq2.callbacks.Debugger())

    torchtnt.runner.fit(train_state, task, callbacks=callbacks)


if __name__ == "__main__":
    import func_argparse

    func_argparse.single_main(main)
