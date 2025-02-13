# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.cli import Cli
from fairseq2.cli.commands.assets import ListAssetsHandler, ShowAssetHandler
from fairseq2.cli.commands.chatbot import RunChatbotHandler
from fairseq2.cli.commands.llama import (
    ConvertLLaMACheckpointHandler,
    WriteHFLLaMAConfigHandler,
)
from fairseq2.cli.commands.recipe import RecipeCommandHandler
from fairseq2.context import RuntimeContext
from fairseq2.extensions import run_extensions
from fairseq2.recipes.asr import AsrEvalConfig, load_asr_evaluator
from fairseq2.recipes.lm import (
    InstructionFinetuneConfig,
    LMLossEvalConfig,
    POFinetuneConfig,
    TextGenerateConfig,
    load_instruction_finetuner,
    load_lm_loss_evaluator,
    load_po_finetuner,
    load_text_generator,
)
from fairseq2.recipes.mt import (
    MTEvalConfig,
    MTTrainConfig,
    TextTranslateConfig,
    load_mt_evaluator,
    load_mt_trainer,
    load_text_translator,
)
from fairseq2.recipes.wav2vec2 import (
    Wav2Vec2EvalConfig,
    Wav2Vec2TrainConfig,
    load_wav2vec2_evaluator,
    load_wav2vec2_trainer,
)
from fairseq2.recipes.wav2vec2.asr import (
    Wav2Vec2AsrTrainConfig,
    load_wav2vec2_asr_trainer,
)
from fairseq2.setup._cli._error_types import _register_user_error_types


def setup_cli(context: RuntimeContext) -> Cli:
    from fairseq2 import __version__

    cli = Cli(
        name="fairseq2",
        origin_module="fairseq2",
        version=__version__,
        description="command line interface of fairseq2",
    )

    _register_asr_cli(cli)
    _register_asset_cli(cli)
    _register_chatbot_cli(cli)
    _register_llama_cli(cli)
    _register_lm_cli(cli)
    _register_mt_cli(cli)
    _register_wav2vec2_asr_cli(cli)
    _register_wav2vec2_cli(cli)

    _register_user_error_types(cli)

    run_extensions("fairseq2.cli", context, cli)

    return cli


def _register_asr_cli(cli: Cli) -> None:
    extra_sweep_keys = {"max_audio_len", "min_audio_len", "normalize_audio"}

    group = cli.add_group("asr", help="ASR recipes")

    # Eval
    eval_handler = RecipeCommandHandler(
        loader=load_asr_evaluator,
        config_kls=AsrEvalConfig,
        default_preset="wav2vec2",
        extra_sweep_keys=extra_sweep_keys,
    )

    group.add_command(
        name="eval",
        handler=eval_handler,
        help="evaluate an ASR model",
    )


def _register_asset_cli(cli: Cli) -> None:
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


def _register_chatbot_cli(cli: Cli) -> None:
    group = cli.add_group("chatbot", help="chatbot demo")

    group.add_command(
        name="run",
        handler=RunChatbotHandler(),
        help="run a terminal-based chatbot demo",
    )


def _register_llama_cli(cli: Cli) -> None:
    group = cli.add_group("llama", help="LLaMA recipes")

    group.add_command(
        name="convert_checkpoint",
        handler=ConvertLLaMACheckpointHandler(),
        help="convert fairseq2 LLaMA checkpoints to reference checkpoints",
    )

    group.add_command(
        name="write_hf_config",
        handler=WriteHFLLaMAConfigHandler(),
        help="write fairseq2 LLaMA configurations in Hugging Face format",
    )


def _register_lm_cli(cli: Cli) -> None:
    group = cli.add_group("lm", help="language model recipes")

    # Instruction Finetune
    instruction_finetune_handler = RecipeCommandHandler(
        loader=load_instruction_finetuner,
        config_kls=InstructionFinetuneConfig,
        default_preset="llama3_1_instruct",
    )

    group.add_command(
        name="instruction_finetune",
        handler=instruction_finetune_handler,
        help="instruction-finetune a language model",
    )

    # Loss Evaluation
    loss_eval_handler = RecipeCommandHandler(
        loader=load_lm_loss_evaluator,
        config_kls=LMLossEvalConfig,
        default_preset="llama3_1_base_eval",
    )

    group.add_command(
        name="nll_eval",
        handler=loss_eval_handler,
        help="Evaluate the model and compute NLL loss over a given dataset",
    )

    # PO Finetune
    po_finetune_handler = RecipeCommandHandler(
        loader=load_po_finetuner,
        config_kls=POFinetuneConfig,
        default_preset="llama3_1_instruct",
    )

    group.add_command(
        name="preference_finetune",
        handler=po_finetune_handler,
        help="preference-finetune a language model (e.g. DPO, SimPO).",
    )

    # Text Generate
    text_generate_handler = RecipeCommandHandler(
        loader=load_text_generator,
        config_kls=TextGenerateConfig,
        default_preset="llama3_1_instruct",
    )

    group.add_command(
        name="generate",
        handler=text_generate_handler,
        help="generate text",
    )


def _register_mt_cli(cli: Cli) -> None:
    extra_sweep_keys = {"source_lang", "target_lang"}

    group = cli.add_group("mt", help="machine translation recipes")

    # Eval
    eval_handler = RecipeCommandHandler(
        loader=load_mt_evaluator,
        config_kls=MTEvalConfig,
        default_preset="nllb_dense",
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
        default_preset="nllb_dense",
        extra_sweep_keys=extra_sweep_keys,
    )

    group.add_command(
        name="train",
        handler=train_handler,
        help="train a machine translation model",
    )

    # Translate
    text_translate_handler = RecipeCommandHandler(
        loader=load_text_translator,
        config_kls=TextTranslateConfig,
        default_preset="nllb_dense",
        extra_sweep_keys=extra_sweep_keys,
    )

    group.add_command(
        name="translate",
        handler=text_translate_handler,
        help="translate text",
    )


def _register_wav2vec2_cli(cli: Cli) -> None:
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


def _register_wav2vec2_asr_cli(cli: Cli) -> None:
    extra_sweep_keys = {
        "freeze_encoder_for_n_steps",
        "max_audio_len",
        "min_audio_len",
        "normalize_audio",
    }

    group = cli.add_group("wav2vec2_asr", help="wav2vec 2.0 ASR recipes")

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
