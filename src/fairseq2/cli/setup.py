# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.cli import Cli
from fairseq2.cli.commands.assets import ListAssetsHandler, ShowAssetHandler
from fairseq2.cli.commands.chatbot import ChatbotHandler
from fairseq2.cli.commands.llama import (
    ConvertLLaMACheckpointHandler,
    WriteLLaMAHFConfigHandler,
)
from fairseq2.cli.commands.recipe import RecipeCommandHandler
from fairseq2.extensions import run_extensions
from fairseq2.recipes.lm import (
    LMInstructionFinetuneConfig,
    LMNllEvalConfig,
    LMPreferenceFinetuneConfig,
    LMTextGenerateConfig,
    load_lm_instruction_finetuner,
    load_lm_nll_evaluator,
    load_lm_preference_finetuner,
    load_lm_text_generator,
)
from fairseq2.recipes.mt import (
    MTEvalConfig,
    MTTrainConfig,
    MTTranslateConfig,
    load_mt_evaluator,
    load_mt_trainer,
    load_mt_translator,
)
from fairseq2.recipes.wav2vec2 import (
    Wav2Vec2EvalConfig,
    Wav2Vec2TrainConfig,
    load_wav2vec2_evaluator,
    load_wav2vec2_trainer,
)
from fairseq2.recipes.wav2vec2.asr import (
    Wav2Vec2AsrEvalConfig,
    Wav2Vec2AsrTrainConfig,
    load_wav2vec2_asr_evaluator,
    load_wav2vec2_asr_trainer,
)


def setup_cli(cli: Cli) -> None:
    _setup_asset_cli(cli)
    _setup_chatbot_cli(cli)
    _setup_llama_cli(cli)
    _setup_lm_cli(cli)
    _setup_mt_cli(cli)
    _setup_wav2vec2_cli(cli)
    _setup_wav2vec2_asr_cli(cli)

    run_extensions("fairseq2.cli", cli)


def _setup_asset_cli(cli: Cli) -> None:
    group = cli.add_group(
        "assets", help="list and show assets (e.g. models, tokenizers, datasets)"
    )

    group.add_command(
        "list",
        ListAssetsHandler(),
        help="list assets",
    )

    group.add_command(
        "show",
        ShowAssetHandler(),
        help="show asset",
    )


def _setup_chatbot_cli(cli: Cli) -> None:
    group = cli.add_group("chatbot", help="chatbot demo")

    group.add_command(
        name="run",
        handler=ChatbotHandler(),
        help="run a terminal-based chatbot demo",
    )


def _setup_llama_cli(cli: Cli) -> None:
    group = cli.add_group("llama", help="LLaMA recipes")

    group.add_command(
        name="convert_checkpoint",
        handler=ConvertLLaMACheckpointHandler(),
        help="convert fairseq2 LLaMA checkpoints to reference checkpoints",
    )

    group.add_command(
        name="write_hf_config",
        handler=WriteLLaMAHFConfigHandler(),
        help="write fairseq2 LLaMA config in Huggingface config format",
    )


def _setup_lm_cli(cli: Cli) -> None:
    group = cli.add_group("lm", help="language model recipes")

    # Instruction Finetune
    instruction_finetune_handler = RecipeCommandHandler(
        loader=load_lm_instruction_finetuner,
        config_kls=LMInstructionFinetuneConfig,
        default_preset="llama3_1_instruct",
    )

    group.add_command(
        name="instruction_finetune",
        handler=instruction_finetune_handler,
        help="instruction-finetune a language model",
    )

    # NLL Evaluation
    nll_eval_handler = RecipeCommandHandler(
        loader=load_lm_nll_evaluator,
        config_kls=LMNllEvalConfig,
        default_preset="llama3_1_base_eval",
    )

    group.add_command(
        name="nll_eval",
        handler=nll_eval_handler,
        help="Evaluate the model and compute NLL loss over a given dataset",
    )

    # Preference Finetune
    preference_finetune_handler = RecipeCommandHandler(
        loader=load_lm_preference_finetuner,
        config_kls=LMPreferenceFinetuneConfig,
        default_preset="llama3_1_instruct",
    )

    group.add_command(
        name="preference_finetune",
        handler=preference_finetune_handler,
        help="preference-finetune a language model (e.g. DPO, SimPO).",
    )

    # Text Generate
    text_generate_handler = RecipeCommandHandler(
        loader=load_lm_text_generator,
        config_kls=LMTextGenerateConfig,
        default_preset="llama3_1_8b_instruct",
    )

    group.add_command(
        name="generate",
        handler=text_generate_handler,
        help="generate text",
    )


def _setup_mt_cli(cli: Cli) -> None:
    extra_sweep_keys = {"source_lang", "target_lang"}

    group = cli.add_group("mt", help="machine translation recipes")

    # Eval
    eval_handler = RecipeCommandHandler(
        loader=load_mt_evaluator,
        config_kls=MTEvalConfig,
        default_preset="nllb_dense_600m",
        extra_sweep_keys=extra_sweep_keys,
    )

    group.add_command(
        name="eval",
        handler=eval_handler,
        help="evaluate a machine translation model",
    )

    # Train
    train_handler = RecipeCommandHandler(
        loader=load_mt_trainer,
        config_kls=MTTrainConfig,
        default_preset="nllb_dense_600m",
        extra_sweep_keys=extra_sweep_keys,
    )

    group.add_command(
        name="train",
        handler=train_handler,
        help="train a machine translation model",
    )

    # Translate
    translate_handler = RecipeCommandHandler(
        loader=load_mt_translator,
        config_kls=MTTranslateConfig,
        default_preset="nllb_dense_600m",
        extra_sweep_keys=extra_sweep_keys,
    )

    group.add_command(
        name="translate",
        handler=translate_handler,
        help="translate text",
    )


def _setup_wav2vec2_cli(cli: Cli) -> None:
    extra_sweep_keys = {"max_audio_len", "min_audio_len", "normalize_audio"}

    group = cli.add_group("wav2vec2", help="wav2vec 2.0 pretraining recipes")

    # Eval
    eval_handler = RecipeCommandHandler(
        loader=load_wav2vec2_evaluator,
        config_kls=Wav2Vec2EvalConfig,
        default_preset="base_ls960h",
        extra_sweep_keys=extra_sweep_keys,
    )

    group.add_command(
        name="eval",
        handler=eval_handler,
        help="evaluate a wav2vec 2.0 model",
    )

    # Train
    train_handler = RecipeCommandHandler(
        loader=load_wav2vec2_trainer,
        config_kls=Wav2Vec2TrainConfig,
        default_preset="base_960h",
        extra_sweep_keys=extra_sweep_keys,
    )

    group.add_command(
        name="train",
        handler=train_handler,
        help="train a wav2vec 2.0 model",
    )


def _setup_wav2vec2_asr_cli(cli: Cli) -> None:
    extra_sweep_keys = {
        "freeze_encoder_for_n_steps",
        "max_audio_len",
        "min_audio_len",
        "normalize_audio",
    }

    group = cli.add_group("wav2vec2_asr", help="wav2vec 2.0 ASR recipes")

    # Eval
    eval_handler = RecipeCommandHandler(
        loader=load_wav2vec2_asr_evaluator,
        config_kls=Wav2Vec2AsrEvalConfig,
        default_preset="base_10h",
        extra_sweep_keys=extra_sweep_keys,
    )

    group.add_command(
        name="eval",
        handler=eval_handler,
        help="evaluate a wav2vec 2.0 ASR model",
    )

    # Train
    train_handler = RecipeCommandHandler(
        loader=load_wav2vec2_asr_trainer,
        config_kls=Wav2Vec2AsrTrainConfig,
        default_preset="base_10h",
        extra_sweep_keys=extra_sweep_keys,
    )

    group.add_command(
        name="train",
        handler=train_handler,
        help="train a wav2vec 2.0 ASR model",
    )
