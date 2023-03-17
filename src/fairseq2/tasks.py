import functools
import itertools
import math
from typing import Any, Dict, List, Optional

import sacrebleu  # type: ignore
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler
import torcheval.metrics
import torchtnt.framework
import torchtnt.framework.callbacks
import torchtnt.utils
from torch import Tensor
from torchtnt.framework.state import State
from torchtnt.framework.unit import EvalUnit, PredictUnit, TrainUnit

import fairseq2.callbacks
import fairseq2.nn
import fairseq2.optim.lr_scheduler
from fairseq2.callbacks import Metrics
from fairseq2.dataloader import Seq2SeqBatch, Seq2SeqStr
from fairseq2.generate import BeamSearchStrategy, SearchStrategy, Tokenizer
from fairseq2.models.transformer import TransformerModel
from fairseq2.optim.lr_scheduler import LRScheduler


class Seq2Seq(
    TrainUnit[Seq2SeqBatch], EvalUnit[Seq2SeqBatch], PredictUnit[Seq2SeqBatch]
):
    """Default seq2seq task"""

    # Note: this is very close to the tnt.AutoUnit, maybe we should inherit from them.
    def __init__(
        self,
        model: TransformerModel,
        tokenizer: Tokenizer,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[LRScheduler] = None,
        best_metric: str = "loss",
    ):
        super().__init__()
        # initialize module & optimizer
        self.model = model
        self.tokenizer = tokenizer
        self.best_metric = best_metric

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.clip_grad_norm = 0.00
        self.log_frequency_steps = 100
        self.lr_frequency_steps = 1
        self.seed = 1
        self.eval_gen = True

    def init_metrics(self, mode: str) -> Metrics:
        metrics = Metrics(
            loss=torcheval.metrics.Mean(),
            num_tokens=torcheval.metrics.Sum(),
            tps=torcheval.metrics.Throughput(),
            etps=fairseq2.callbacks.EffectiveThroughput(),
        )
        if mode == "eval":
            metrics["loss_min"] = torcheval.metrics.Min()
            metrics["best"] = False
        return metrics

    def replicated_keys(self) -> List[str]:
        return ["tokenizer"]

    def train_step(self, state: State, data: Seq2SeqBatch) -> None:
        assert state.train_state
        seed = self.seed + state.train_state.progress.num_steps_completed
        torchtnt.utils.seed(seed)

        # TODO: upgrade once torchtnt has builtin support for metric
        metrics = state.train_state.metrics  # type: ignore
        loss = self.loss(metrics, data)
        # TODO: allow accumulating gradients over several batch
        loss.backward()
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)  # type: ignore
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Increment step since we already did the forward/backward for this one.
        steps = state.train_state.progress.num_steps_completed + 1
        if self.lr_scheduler is not None and steps % self.lr_frequency_steps == 0:
            self.lr_scheduler.step()

        # This is the elapsed time since TNT started the "train_step"
        num_tokens = data.num_tokens
        metrics["tps"].update(
            num_tokens, elapsed_time_sec=state.timer.interval_time_seconds
        )
        metrics["etps"].update(num_tokens)

    def on_train_start(self, state: State) -> None:
        assert state.train_state
        # TODO: upgrade once torchtnt has builtin support for metric
        state.train_state.metrics = self.init_metrics("train")  # type: ignore

    def on_train_end(self, state: State) -> None:
        pass

    def loss(self, metrics: Metrics, data: Seq2SeqBatch) -> Tensor:
        net_output = self.model(data.source, data.target[:, :-1])
        if not isinstance(net_output, Tensor):
            net_output = net_output.logits

        lprobs = F.log_softmax(net_output, dim=-1).transpose(2, 1)
        # TODO: nll loss requires longs ? Why ?
        loss = F.nll_loss(
            lprobs, data.target[:, 1:], reduction="sum", ignore_index=self.tokenizer.PAD
        )
        num_tokens = torch.tensor(data.num_tokens)
        nll_loss = loss.detach().cpu() / num_tokens / math.log(2)
        metrics["loss"].update(nll_loss, weight=num_tokens)
        metrics["num_tokens"].update(num_tokens)
        if self.lr_scheduler is not None:
            metrics["lr"] = self.lr_scheduler.get_last_lr()[0]

        return loss

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for k, v in state_dict.items():
            self.__getattr__(k).load_state_dict(v)

    def state_dict_for_inference(self) -> Dict[str, Any]:
        """Returns a minimal state_dict without optimizer or LR that can be used for inference."""
        app_state = {
            **self.tracked_modules(),
            **self.tracked_misc_statefuls(),
        }
        assert "tokenizer" in app_state
        return app_state

    def on_eval_start(self, state: State) -> None:
        assert state.eval_state
        # TODO: upgrade once torchtnt has builtin support for metric
        if not hasattr(state.eval_state, "metrics"):
            state.eval_state.metrics = self.init_metrics("eval")  # type: ignore

    def on_eval_end(self, state: State) -> None:
        # TODO: upgrade once torchtnt has builtin support for metric
        metrics = state.eval_state.metrics  # type: ignore
        eval_loss = metrics["loss"]
        metrics["loss_min"].update(metrics[self.best_metric].compute())
        best_loss = float(metrics["loss_min"].compute())
        metrics["best"] = best_loss == eval_loss

    def eval_step(self, state: State, data: Seq2SeqBatch) -> Any:
        assert state.eval_state
        # TODO: upgrade once torchtnt has builtin support for metric
        metrics = state.eval_state.metrics  # type: ignore
        self.loss(metrics, data).detach().cpu()

    def predict_step(self, state: State, data: Seq2SeqBatch) -> List[Seq2SeqStr]:
        return self.generate_batch(data)  # type: ignore

    @functools.lru_cache()
    def default_strategy(self) -> SearchStrategy:
        return BeamSearchStrategy(beam_size=5, max_len=512, vocab_info=self.tokenizer)

    @torch.inference_mode()
    def generate_batch(self, data: Seq2SeqBatch) -> List[Seq2SeqStr]:
        source = self.tokenizer.decode_batch(data.source)
        target = self.tokenizer.decode_batch(data.target)
        strategy = self.default_strategy()
        predicted_tokens = strategy.generate(self.model, data.source, top=1)
        predicted = self.tokenizer.decode_batch(predicted_tokens.squeeze(1))
        return [
            Seq2SeqStr(*x) for x in itertools.zip_longest(source, target, predicted)
        ]


class TranslationTask(Seq2Seq):
    """Translation task"""

    def init_metrics(self, mode: str) -> Metrics:
        metrics = super().init_metrics(mode)
        self.best_metric = "bleu" if self.eval_gen else "loss"

        if mode == "eval":
            # TODO: propagate the target language here so we can pass it to sacrebleu for best tokenization.
            metrics["bleu"] = fairseq2.callbacks.Bleu()
            metrics["chrf"] = fairseq2.callbacks.Bleu(sacrebleu.CHRF())  # type: ignore
            metrics["chrf++"] = fairseq2.callbacks.Bleu(sacrebleu.CHRF(word_order=2))  # type: ignore

        return metrics

    def eval_step(self, state: State, data: Seq2SeqBatch) -> Any:
        super().eval_step(state, data)

        assert state.eval_state
        # TODO: upgrade once torchtnt has builtin support for metric
        metrics = state.eval_state.metrics  # type: ignore

        if self.eval_gen:
            bleu = metrics["bleu"]
            chrf = metrics["chrf"]
            chrfpp = metrics["chrf++"]
            translations = self.generate_batch(data)
            for translation in translations:
                for counter in (bleu, chrf, chrfpp):
                    counter.update(
                        translation.predicted, references=[translation.target]
                    )
            return translations

        # TODO: compute tokenized bleu ?
