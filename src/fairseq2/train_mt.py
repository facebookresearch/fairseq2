import itertools
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, NamedTuple, cast

import datasets
import torch
import torch.nn.functional as F
import torchtnt.runner
import torchtnt.runner.callbacks
import torchtnt.utils
from datasets import Dataset
from torch import Tensor
from torcheval.metrics import Mean
from torchtnt.loggers.logger import MetricLogger
from torchtnt.runner.state import State
from torchtnt.runner.unit import EvalUnit, PredictUnit, TrainUnit

import fairseq2.callbacks
import fairseq2.nn
from fairseq2.generate import BeamSearch, SpmTokenizer, Tokenizer, generate
from fairseq2.nn import transformer
from fairseq2.typing import Device

REQUIREMENTS = [
    "sentencepiece",
    "torcheval",
    "omegaconf",
    "fairscale",
    # TODO: make optional
    "datasets",
    "wandb",
]

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# For now, required for generation
BATCH_FIRST = True


class Batch(NamedTuple):
    source: torch.Tensor
    target: torch.Tensor


class Translation(NamedTuple):
    source: str
    target: str
    predicted: str


class MachineTranslationTask(TrainUnit[Batch], EvalUnit[Batch], PredictUnit[Batch]):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
        logger: MetricLogger,
    ):
        # TODO: take the spm as input
        super().__init__()
        # initialize module & optimizer

        # TODO: we should take all of this as inputs
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.9
        )

        self.train_loss = Mean()
        self.eval_loss = Mean()
        self.log_frequency_steps = 10
        self.predict_frequency_steps = 1000
        self.lr_frequency_steps = 10_000
        self.logger = logger

    def replicated_keys(self) -> List[str]:
        return ["logger/**"]

    def train_step(self, state: State, data: Batch) -> None:
        assert state.train_state
        steps = state.train_state.progress.num_steps_completed + 1
        loss = self._nll_loss(data)

        # TODO: allow accumulating gradients over several batch
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # update metrics & logs
        self.train_loss.update(loss.detach().cpu())

        if steps % self.log_frequency_steps == 0:
            self._flush_loss(state, "train", self.train_loss)

        if steps % self.lr_frequency_steps == 0:
            self.lr_scheduler.step()

    def on_train_start(self, state: State) -> None:
        pass

    def on_train_end(self, state: State) -> None:
        self.logger.close()

    def _nll_loss(self, data: Batch) -> Tensor:
        # TODO: nll loss requires longs ? Why ?
        net_output = self.model.forward(data.source, data.target)
        lprobs = F.log_softmax(net_output, dim=-1).transpose(2, 1)
        loss = F.nll_loss(
            lprobs, data.target, reduction="mean", ignore_index=self.tokenizer.PAD
        )
        return loss

    def _flush_loss(
        self,
        state: State,
        prefix: str,
        metric: Any,
    ) -> None:
        assert state.train_state
        step = state.train_state.progress.num_steps_completed
        loss = metric.compute()
        metric.reset()
        # This isn't really accurate.
        # This is the perplexity of the average loss, not the average perplexity.
        # But AFAICT Fairseq also computed it this way.
        ppl = math.exp(loss)
        # TODO use tnt.logger
        metrics = {f"{prefix}/loss": loss, f"{prefix}/ppl": ppl}
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
        self._flush_loss(state, "eval", self.eval_loss)

    def eval_step(self, state: State, data: Batch) -> None:
        assert state.eval_state

        self.eval_loss.update(self._nll_loss(data).detach().cpu())

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


# TODO: Implement ColumnParallelLinear, RowParallelLinear
class ModelBuilder(transformer.TransformerBuilder):
    pass


#    def to_yaml(self, file: Path) -> None:
#        config = dataclasses.asdict(self)
#        cls = type(self)
#        config["_target_"] = f"{cls.__module__}.{cls.__qualname__}"
#        omegaconf.OmegaConf.save(config=config, f=file)


def get_large_model_builder(
    num_tokens: int, padding_token_idx: int, device: Device
) -> ModelBuilder:
    return ModelBuilder(
        num_tokens,
        padding_token_idx,
        model_dim=128,
        num_enc_layers=6,
        num_dec_layers=6,
        num_enc_attn_heads=16,
        num_dec_attn_heads=16,
        ffn_inner_dim=512,
        dropout_p=0.1,
        device=device,
    )


def get_small_model_builder(
    num_tokens: int, padding_token_idx: int, device: Device
) -> ModelBuilder:
    return ModelBuilder(
        num_tokens,
        padding_token_idx,
        model_dim=16,
        num_enc_layers=2,
        num_dec_layers=2,
        num_enc_attn_heads=4,
        num_dec_attn_heads=4,
        ffn_inner_dim=16,
        dropout_p=0.1,
        device=device,
    )


def _finalize_batch(
    tokenizer: Tokenizer,
    src_batch: List[str],
    tgt_batch: List[str],
    device: Device,
) -> Batch:
    # TODO: add batch statistic
    return Batch(
        source=tokenizer.encode_batch(src_batch).to(device),
        target=tokenizer.encode_batch(tgt_batch).to(device),
    )


class DatasetLoader(Iterable[Batch]):
    def __init__(
        self,
        train: bool,
        src: str,
        tgt: str,
        tokenizer: Tokenizer,
        batch_size: int,
        global_rank: int,
        world_size: int,
        device: Device,
    ):
        self.src = src
        self.tgt = tgt
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.global_rank = global_rank
        self.world_size = world_size
        self.device = device

        if train:
            try:
                data = cast(
                    Mapping[str, Dataset],
                    datasets.load_dataset(
                        "allenai/nllb",
                        f"{src}-{tgt}",
                        save_infos=True,
                        streaming=True,
                    ),
                )
            except Exception:
                # TODO: why is this run twice? Is this meant to fallback to another dataset?
                data = cast(
                    Mapping[str, Dataset],
                    datasets.load_dataset(
                        "allenai/nllb",
                        f"{tgt}-{src}",
                        save_infos=True,
                        streaming=True,
                    ),
                )
            self.data = data["train"]
            self.extract_src = lambda sample: sample["translation"][src]
            self.extract_tgt = lambda sample: sample["translation"][tgt]
        else:
            data = cast(
                Mapping[str, Dataset],
                datasets.load_dataset("facebook/flores", f"{src}-{tgt}"),
            )
            self.data = data["dev"]
            self.extract_src = lambda sample: sample["sentence_" + src]
            self.extract_tgt = lambda sample: sample["sentence_" + tgt]

    def __iter__(self) -> Iterator[Batch]:
        src_batch, tgt_batch = [], []
        for i, sample in enumerate(self.data):
            if i % self.world_size != self.global_rank:
                continue

            src_batch.append(self.extract_src(sample))
            tgt_batch.append(self.extract_tgt(sample))
            if len(src_batch) == self.batch_size:
                yield _finalize_batch(self.tokenizer, src_batch, tgt_batch, self.device)
                src_batch, tgt_batch = [], []
        if src_batch:
            yield _finalize_batch(self.tokenizer, src_batch, tgt_batch, self.device)


def init_env() -> None:
    local_rank = os.environ.get("LOCAL_RANK", 0)
    rank = os.environ.get("RANK", 0)
    group_rank = os.environ.get("GROUP_RANK", 0)
    world_size = os.environ.get("WORLD_SIZE", 1)
    master_port = os.environ.get("MASTER_PORT", 2000)
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    log.info(
        f"LOCAL_RANK: {local_rank}\n"
        f"RANK: {rank}\n"
        f"GROUP_RANK: {group_rank}\n"
        f"WORLD_SIZE: {world_size}"
    )
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(rank)
    os.environ["GROUP_RANK"] = str(group_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ.update(os.environ)


def main(
    workdir: Path = Path("/checkpoint/guw/fairseq2/mt.{src}-{tgt}"),
    spm_path: Path = Path("/private/home/kevinheffernan/nllb.200.models/laser2.spm"),
    src_lang: str = "eng_Latn",
    tgt_lang: str = "cat_Latn",
    small: bool = False,
    wandb_project: str = "nllb/fairseq2",
    batch_size: int = 16,
) -> None:
    workdir = Path(str(workdir).format(src=src_lang, tgt=tgt_lang))
    workdir.mkdir(exist_ok=True)
    init_env()
    device = torchtnt.utils.init_from_env()

    tokenizer = SpmTokenizer.from_file(spm_path, batch_first=BATCH_FIRST)
    vocab_size = tokenizer.vocab_size()

    padding_token_idx = tokenizer.PAD

    device = Device("cuda:0")

    if small:
        builder_fn = get_small_model_builder
    else:
        builder_fn = get_large_model_builder

    builder = builder_fn(vocab_size, padding_token_idx, device)

    model = builder.build()

    if wandb_project:
        logger: MetricLogger = fairseq2.callbacks.WandbLogger(wandb_project, builder)
    else:
        logger = torchtnt.loggers.TensorBoardLogger(str(workdir / "tensorboard"))

    task = MachineTranslationTask(model, tokenizer, logger)
    train_dataloader = DatasetLoader(
        train=True,
        src=src_lang,
        tgt=tgt_lang,
        tokenizer=tokenizer,
        batch_size=batch_size,
        global_rank=int(os.environ.get("GROUP_RANK", 0)),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        device=device,
    )
    valid_dataloader = DatasetLoader(
        train=False,
        src=src_lang,
        tgt=tgt_lang,
        tokenizer=tokenizer,
        batch_size=batch_size,
        global_rank=int(os.environ.get("GROUP_RANK", 0)),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        device=device,
    )

    train_state = torchtnt.runner.init_fit_state(
        train_dataloader,
        valid_dataloader,
        max_epochs=2**30,
        max_steps=1_000_000,
        evaluate_every_n_steps=1000,
    )
    callbacks = [
        # Synchronize GC runs across all nodes
        torchtnt.runner.callbacks.GarbageCollector(step_interval=1),
        torchtnt.runner.callbacks.TorchSnapshotSaver(
            str(workdir),
            save_every_n_train_steps=100,
            replicated=task.replicated_keys(),
        ),
        fairseq2.callbacks.TorchSnapshotLoader(
            str(workdir), replicated=task.replicated_keys()
        ),
    ]
    if os.isatty(sys.stdout.fileno()):
        callbacks.append(fairseq2.callbacks.Debugger())

    torchtnt.runner.fit(train_state, task, callbacks=callbacks)


if __name__ == "__main__":
    # test_spm()
    import func_argparse

    func_argparse.single_main(main)
