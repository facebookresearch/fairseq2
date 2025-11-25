# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing_extensions import override
from wandb import Run as WandbRun

from fairseq2.evals.lm_eval_harness import LMEvalHarness, LMEvalTaskNotKnownError
from fairseq2.models.clm import CausalLM
from fairseq2.recipe import Recipe, RecipeContext
from fairseq2.task import Task
from fairseq2.utils.validation import ValidationError

from ..common import check_model_vocabulary
from .config import LMEvalConfig


class LMEvalRecipe(Recipe):
    @override
    def create_task(self, context: RecipeContext) -> Task:
        config = context.get_config_as(LMEvalConfig)

        check_model_vocabulary(context)

        model = context.get_model_as(CausalLM)

        tokenizer = context.get_tokenizer()

        wandb_run = context.resolver.maybe_resolve(WandbRun)

        try:
            return LMEvalHarness(
                model=model,
                tokenizer=tokenizer,
                output_dir=context.output_dir,
                gangs=context.gangs,
                tasks=config.evaluator.tasks,
                batch_size=config.evaluator.batch_size,
                amp=config.evaluator.amp,
                amp_dtype=config.evaluator.amp_dtype,
                num_fewshot=config.evaluator.num_fewshot,
                cache_requests=config.evaluator.cache_requests,
                log_samples=config.evaluator.log_samples,
                step_nr=config.evaluator.step_nr,
                wandb_run=wandb_run,
                progress_reporter=context.progress_reporter,
            )
        except LMEvalTaskNotKnownError as ex:
            raise ValidationError(
                f"{ex.name} in `tasks` is not a known LM Evaluation Harness task.", field="evaluator"  # fmt: skip
            ) from None

    @property
    @override
    def config_kls(self) -> type[object]:
        return LMEvalConfig
