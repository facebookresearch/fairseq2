# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence, final

import torch
import wandb
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from torch import Tensor
from typing_extensions import override
from wandb import Run as WandbRun

from fairseq2.data.tokenizers import Tokenizer
from fairseq2.data_type import DataType
from fairseq2.device import CPU, Device
from fairseq2.error import InternalError, NotSupportedError
from fairseq2.gang import Gang, Gangs
from fairseq2.logging import log
from fairseq2.models.clm import CausalLM
from fairseq2.task import Task, TaskStopException
from fairseq2.typing import ContextManager
from fairseq2.utils.progress import NOOP_PROGRESS_REPORTER, ProgressReporter


@final
class LMEvalHarness(Task):
    def __init__(
        self,
        *,
        model: CausalLM,
        tokenizer: Tokenizer,
        output_dir: Path,
        tasks: Sequence[str | dict[str, object]],
        gangs: Gangs,
        batch_size: int,
        amp: bool,
        amp_dtype: DataType,
        num_fewshot: int | None,
        cache_requests: bool,
        log_samples: bool,
        step_nr: int | None,
        wandb_run: WandbRun | None,
        progress_reporter: ProgressReporter,
    ) -> None:
        eval_lm = _EvalLM(
            model, tokenizer, gangs, batch_size, amp, amp_dtype, progress_reporter
        )

        save_path = output_dir.joinpath("lm_eval_harness.json")

        self._eval_lm = eval_lm
        self._save_path = save_path
        self._tasks = tasks
        self._gangs = gangs
        self._num_fewshot = num_fewshot
        self._cache_requests = cache_requests
        self._log_samples = log_samples
        self._step_nr = step_nr
        self._wandb_run = wandb_run

    @override
    def run(self) -> None:
        from lm_eval import simple_evaluate
        from lm_eval.loggers import EvaluationTracker, WandbLogger
        from lm_eval.utils import make_table

        if self._wandb_run is not None:
            if self._wandb_run is not wandb.run:
                raise NotSupportedError(
                    "LM Evaluation Harness supports only global Weights & Biases run."
                )

            wandb_logger = WandbLogger(init_args={"step": self._step_nr})
        else:
            wandb_logger = None

        tracker = EvaluationTracker(output_path=self._save_path)

        try:
            results = simple_evaluate(
                self._eval_lm,
                model_args={},
                tasks=self._tasks,
                num_fewshot=self._num_fewshot,
                device=self._gangs.root.device,
                cache_requests=self._cache_requests,
                random_seed=None,
                numpy_random_seed=None,
                torch_random_seed=None,
                fewshot_random_seed=None,
            )
        except KeyError as ex:
            task_name = str(ex)

            raise LMEvalTaskNotKnownError(task_name) from None

        if self._gangs.root.rank != 0:
            return

        if results is None:
            raise InternalError("`results` is `None` on rank 0.")

        tracker.general_config_tracker.log_experiment_args(
            model_source="fairseq2",
            model_args="",
            system_instruction="",
            chat_template="",
            fewshot_as_multiturn=False,
        )

        samples = results.pop("samples", None)

        if not self._log_samples:
            samples = None

        tracker.save_results_aggregated(results, samples)

        if samples is not None:
            for task_name, _ in results["configs"].items():
                tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )

        if wandb_logger is not None:
            wandb_logger.post_init(results)

            wandb_logger.log_eval_result()

            if samples is not None:
                wandb_logger.log_eval_samples(samples)

        table = make_table(results)

        log.info("Evaluation Results:\n\n{}", table)

        if "groups" in results:
            table = make_table(results, "groups")

            log.info("Group Evaluation Results:\n\n{}", table)

    @override
    def close(self) -> None:
        pass

    @override
    def request_stop(self) -> None:
        self._eval_lm.request_stop()

    @property
    @override
    def step_nr(self) -> int:
        return self._eval_lm.step_nr


class LMEvalTaskNotKnownError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a known LM evaluation task.")

        self.name = name


@final
class _EvalLM(TemplateLM):
    def __init__(
        self,
        model: CausalLM,
        tokenizer: Tokenizer,
        gangs: Gangs,
        batch_size: int,
        amp: bool,
        amp_dtype: DataType,
        progress_reporter: ProgressReporter,
    ) -> None:
        text_encoder = tokenizer.create_raw_encoder(device=CPU)

        root_gang = gangs.root

        accelerator = _EvalLMAccelerator(root_gang)

        self._model = model
        self._tokenizer = tokenizer
        self._text_encoder = text_encoder
        self._root_gang = root_gang
        self._accelerator = accelerator
        self._batch_size = batch_size
        self._amp = amp
        self._amp_dtype = amp_dtype
        self._progress_reporter = progress_reporter
        self._step_nr = 0
        self._stop_requested = False

    @override
    def eot_token_id(self) -> int:
        return self._tokenizer.vocab_info.eos_idx or 0

    @override
    def tok_encode(self, string: str, **kwargs: Any) -> list[int]:
        t = self._text_encoder(string)

        return t.tolist()

    @override
    def _loglikelihood_tokens(
        self, requests: tuple[tuple[str, str], list[int], list[int]], disable_tqdm: bool
    ) -> list[tuple[float, bool]]:
        from lm_eval.models.utils import Collator

        result = []

        progress_reporter = (
            NOOP_PROGRESS_REPORTER if disable_tqdm else self._progress_reporter
        )

        def collate(
            request: tuple[tuple[str, str], list[int], list[int]],
        ) -> tuple[int, tuple[int, ...]]:
            _, prompt_indices, target_indices = request

            indices = prompt_indices + target_indices

            return -len(indices), tuple(indices)

        collator = Collator(requests, sort_fn=collate)

        num_requests = len(collator)

        with progress_reporter:
            task = progress_reporter.create_task("loglikelihood", total=num_requests)

            with task:
                batches = collator.get_batched(n=self._batch_size)

                for batch in batches:
                    if self._stop_requested:
                        raise TaskStopException()

                    for example in batch:
                        _, prompt_indices, target_indices = example

                        # TODO: remove!
                        result.append((-1.0, False))

                    with self._maybe_autocast():
                        # TODO: Implement!

                        task.step(len(batch))

                    self._step_nr += 1

        return result

    @override
    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        result = []

        progress_reporter = (
            NOOP_PROGRESS_REPORTER if disable_tqdm else self._progress_reporter
        )

        with progress_reporter:
            task = progress_reporter.create_task(
                "loglikelihood_rolling", total=None, start=False
            )

            with task:
                for request in requests:
                    with self._maybe_autocast():
                        # TODO: Implement!
                        result.append(-1.0)

        return result

    @override
    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        result = []

        progress_reporter = (
            NOOP_PROGRESS_REPORTER if disable_tqdm else self._progress_reporter
        )

        with progress_reporter:
            task = progress_reporter.create_task(
                "generate_until", total=None, start=False
            )

            with task:
                for request in requests:
                    with self._maybe_autocast():
                        # TODO: Implement
                        result.append("lol")

        return result

    def _maybe_autocast(self) -> ContextManager[None]:
        if not self._amp or self._amp_dtype == torch.float32:
            return nullcontext()

        device_type = self._gangs.root.device.type

        return torch.autocast(device_type=device_type, dtype=self._amp_dtype)

    @property
    def device(self) -> Device:
        return self._root_gang.device

    @property
    def rank(self) -> int:
        return self._root_gang.rank

    @property
    def world_size(self) -> int:
        return self._root_gang.size

    @property
    def accelerator(self) -> _EvalLMAccelerator:
        return self._accelerator

    def request_stop(self) -> None:
        self._stop_requested = True

    @property
    def step_nr(self) -> int:
        return self._step_nr


@final
class _EvalLMAccelerator:
    def __init__(self, gang: Gang) -> None:
        self._gang = gang

    def gather(self, tensor: Tensor) -> Tensor:
        tensors = [tensor.new_zeros((1,)) for _ in range(self._gang.size)]

        self._gang.all_gather_to_list(tensors, tensor)

        return torch.cat(tensors)

    def wait_for_everyone(self) -> None:
        self._gang.barrier()
