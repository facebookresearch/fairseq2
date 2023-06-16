import functools
import inspect
import itertools
import math
import typing as tp
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Type

import sacrebleu  # type: ignore
import torch
import torch.optim.lr_scheduler
import torcheval.metrics
import torchtnt.framework as tnt
from torch import Tensor
from torchtnt.framework.state import State
from torchtnt.utils import TLRScheduler

import fairseq2.nn
import fairseq2.optim.lr_scheduler
from fairseq2.data import StringLike
from fairseq2.data.text import Tokenizer
from fairseq2.generate import BeamSearchStrategy, SearchStrategy
from fairseq2.metrics import Metrics
from fairseq2.optim.lr_scheduler import LRScheduler

if tp.TYPE_CHECKING:
    from fairseq2.models.encoder_decoder import EncoderDecoderModel


@functools.lru_cache(maxsize=1)
def auto_unit_kwargs() -> List[str]:
    return inspect.getfullargspec(tnt.AutoUnit.__init__).kwonlyargs


class Env(NamedTuple):
    """Represents the distributed environment we are currently running in."""

    world_size: int
    """Total number of worker process working together"""

    global_rank: int
    """Unique id of this worker. Workers are numbered from 0 to ``world_size - 1``"""

    device: torch.device
    """Cuda device this worker should use."""


class Seq2SeqBatch(NamedTuple):
    """The default batch type for :py:class:`fairseq2.tasks.Seq2Seq` task"""

    source: "Tensor"
    """Source tokens: Tensor[long] for text input, Tensor[float] for waveform input."""

    src_seq_lens: "Tensor"
    """Lengths of each source sequence, allowing to mask the padding tokens. Tensor[long]"""

    target: "Tensor"
    """Target tokens: Tensor[long]"""

    tgt_seq_lens: "Tensor"
    """Lengths of each target sequence, allowing to mask the padding tokens. Tensor[long]"""

    metadata: Sequence[Dict[str, Any]] = []


class Seq2SeqStr(NamedTuple):
    source: StringLike
    target: StringLike
    predicted: StringLike


class Seq2Seq(tnt.AutoUnit[Seq2SeqBatch]):
    """Default seq2seq task based on tnt.AutoUnit.

    Compared to AutoUnit this provides:
    * a default NLL loss
    * some default metrics like lr, token per second and loss
    * inference using fairseq2.generate

    If your data type is too different from Seq2SeqBatch it's best to directly
    inherit from tnt.AutoUnit.
    """

    data_type: Type[Any] = Seq2SeqBatch

    def __init__(
        self,
        module: torch.nn.Module,
        tokenizer: Tokenizer,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[LRScheduler],
        env: Env,
        step_lr_interval: tp.Literal["step", "epoch"] = "step",
        precision: Optional[torch.dtype] = None,
        gradient_accumulation_steps: int = 1,
        detect_anomaly: bool = False,
        clip_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
    ):
        device = env.device
        kwargs = locals()
        # Pass to AutoUnit the intersection of args we have and the one they want.
        super().__init__(**{k: kwargs[k] for k in auto_unit_kwargs() if k in kwargs})
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        # tnt and us both have a type alias for LRScheduler
        self.lr_scheduler = lr_scheduler  # type: ignore[assignment]
        self.pad_idx = self.tokenizer.vocab_info.pad_idx if self.tokenizer else 0

        self.eval_gen = False

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
        return self.optimizer, self.lr_scheduler  # type: ignore

    def on_train_start(self, state: State) -> None:
        super().on_train_start(state)
        assert state.train_state
        # TODO: upgrade once torchtnt has builtin support for metric
        state.train_state.metrics = self.init_metrics("train")  # type: ignore

    def init_metrics(self, mode: str) -> Metrics:
        device = self.device
        metrics = Metrics(
            loss=torcheval.metrics.Mean(device=device),
            src_num_tokens=torcheval.metrics.Sum(device=device),
            tgt_num_tokens=torcheval.metrics.Sum(device=device),
            tps=torcheval.metrics.Throughput(device=device),
        )
        if mode == "eval":
            metrics["loss_min"] = torcheval.metrics.Min(device=device)
            metrics["best"] = False
        return metrics

    def replicated_keys(self) -> Set[str]:
        if isinstance(self.tokenizer, Tokenizer):
            # This is an optimization to avoid having several copies of the tokenizer
            # in the checkpoint. This may not always be possible,
            # in particular if the tokenizer is a torch.nn.Module
            return {"tokenizer"}

        return set()

    def move_data_to_device(self, state: State, data: Seq2SeqBatch) -> Seq2SeqBatch:
        data = super().move_data_to_device(state, data)
        # Pybind11 converts NamedTuple to list, we restore the NamedTuple here.
        if isinstance(data, list):
            data = self.data_type(*data)
        return data

    def train_step(self, state: State, data: Seq2SeqBatch) -> Tuple[Tensor, Any]:
        # This is the same than AutoUnit train_state, except we compute metric
        # last so that the throughput take into account the backward time.
        data = self.move_data_to_device(state, data)

        assert state.train_state
        train_state = state.train_state
        should_update_weights = (
            train_state.progress.num_steps_completed_in_epoch + 1
        ) % self.gradient_accumulation_steps == 0 or train_state.is_last_batch

        loss, outputs = self._forward_and_backward(state, data, should_update_weights)

        if should_update_weights:
            # TODO try to use dynamo here
            self._run_optimizer_lr_scheduler_step(state)

            # log metrics only after an optimizer step
            if self.num_optimizer_steps_completed % self.log_frequency_steps == 0:
                self.log_metrics(state, self.num_optimizer_steps_completed - 1, "step")

        # users can override this, by default this is a no-op
        self.update_metrics(state, data, loss, outputs)
        return loss, outputs

    def compute_loss(self, state: State, data: Seq2SeqBatch) -> Tuple[Tensor, Any]:
        """Default loss for Seq2Seq is nll_loss."""
        net_output = self.module(
            data.source, data.src_seq_lens, data.target[:, :-1], data.tgt_seq_lens - 1
        )

        loss = net_output.compute_loss(data.target[:, 1:])

        return loss, net_output

    def update_metrics(
        self, state: State, data: Seq2SeqBatch, loss: Tensor, outputs: Any
    ) -> None:
        """Track loss normalized by number of tokens and token per second"""
        metrics = self.active_metrics(state)
        tgt_num_tokens = data.tgt_seq_lens.sum() - data.tgt_seq_lens.numel()
        nll_loss = loss.detach() / tgt_num_tokens / math.log(2)
        # compute the loss metric on device to avoid a cuda.synchronize
        metrics["loss"].update(nll_loss, weight=tgt_num_tokens)
        metrics["tgt_num_tokens"].update(tgt_num_tokens)
        metrics["src_num_tokens"].update(data.src_seq_lens.sum())
        metrics["tps"].update(
            tgt_num_tokens, elapsed_time_sec=state.timer.interval_time_seconds
        )
        if self.lr_scheduler is not None:
            metrics["lr"] = self.lr_scheduler.get_last_lr()[0]

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for k, v in state_dict.items():
            self.__getattr__(k).load_state_dict(v)

    def state_dict_for_inference(self) -> Dict[str, Any]:
        """Returns a minimal state_dict without optimizer or LR that can be used for inference."""
        app_state = {
            **self.tracked_modules(),
            **self.tracked_misc_statefuls(),
        }
        return app_state

    def on_eval_start(self, state: State) -> None:
        assert state.eval_state
        # TODO: upgrade once torchtnt has builtin support for metric
        if not hasattr(state.eval_state, "metrics"):
            state.eval_state.metrics = self.init_metrics("eval")  # type: ignore

    def predict_step(self, state: State, data: Seq2SeqBatch) -> List[Seq2SeqStr]:
        data = self.move_data_to_device(state, data)
        return self.generate_batch(data)  # type: ignore

    @functools.lru_cache()
    def default_strategy(self) -> SearchStrategy:
        return BeamSearchStrategy(
            beam_size=5, max_len=512, vocab_info=self.tokenizer.vocab_info
        )

    @torch.inference_mode()
    def generate_batch(self, data: Seq2SeqBatch) -> List[Seq2SeqStr]:
        token_decoder = self.tokenizer.create_decoder()

        source = token_decoder(data.source)
        target = token_decoder(data.target)

        # TODO: move to data loading
        padding_mask = data.source.ne(self.tokenizer.vocab_info.pad_idx)
        source_lens = torch.count_nonzero(padding_mask, dim=-1)

        strategy = self.default_strategy()
        with self.maybe_autocast_precision:
            # This only work with the "EncoderDecoder" model,
            # but users are allowed to override this method,
            # and therefore we don't enforce self.module to have this type.
            predicted_tokens = strategy.generate(
                self.module, data.source, source_lens, top=1  # type: ignore
            )

        predicted = token_decoder(predicted_tokens.squeeze(1))

        return [
            Seq2SeqStr(*x) for x in itertools.zip_longest(source, target, predicted)
        ]

    def active_metrics(self, state: State) -> Metrics:
        # TODO: upgrade once torchtnt has builtin support for metric
        if state.active_phase == tnt.state.ActivePhase.TRAIN:
            return state.train_state.metrics  # type: ignore
        else:
            return state.eval_state.metrics  # type: ignore


class TranslationTask(Seq2Seq):
    """Translation task"""

    data_type = Seq2SeqBatch
    module: "EncoderDecoderModel"

    def init_metrics(self, mode: str) -> Metrics:
        metrics = super().init_metrics(mode)

        if mode == "eval":
            # TODO: propagate the target language here so we can pass it to sacrebleu for best tokenization.
            metrics["bleu"] = fairseq2.metrics.Bleu()
            metrics["chrf"] = fairseq2.metrics.Bleu(sacrebleu.CHRF())  # type: ignore
            metrics["chrf++"] = fairseq2.metrics.Bleu(sacrebleu.CHRF(word_order=2))  # type: ignore

        return metrics

    def eval_step(self, state: State, data: Seq2SeqBatch) -> Any:
        super().eval_step(state, data)

        if not self.eval_gen:
            return

        metrics = self.active_metrics(state)
        bleu = metrics["bleu"]
        chrf = metrics["chrf"]
        chrfpp = metrics["chrf++"]
        translations = self.generate_batch(data)
        for translation in translations:
            for counter in (bleu, chrf, chrfpp):
                counter.update(translation.predicted, references=[translation.target])

        # TODO: compute tokenized bleu ?
        return translations
