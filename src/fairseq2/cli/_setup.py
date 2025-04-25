# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.chatbots import UnknownChatbotError
from fairseq2.cli.commands.assets import ListAssetsHandler, ShowAssetHandler
from fairseq2.cli.commands.chatbot import RunChatbotHandler
from fairseq2.cli.commands.llama import (
    ConvertLLaMACheckpointHandler,
    WriteHFLLaMAConfigHandler,
)
from fairseq2.cli.commands.recipe import RecipeCommandHandler
from fairseq2.context import RuntimeContext
from fairseq2.data.text.tokenizers import (
    UnknownTextTokenizerError,
    UnknownTextTokenizerFamilyError,
)
from fairseq2.datasets import (
    InvalidDatasetTypeError,
    UnknownDatasetError,
    UnknownDatasetFamilyError,
    UnknownSplitError,
)
from fairseq2.extensions import run_extensions
from fairseq2.generation import (
    UnknownBeamSearchAlgorithmError,
    UnknownSamplerError,
    UnknownSeq2SeqGeneratorError,
    UnknownSequenceGeneratorError,
)
from fairseq2.metrics import UnknownMetricDescriptorError
from fairseq2.metrics.recorders import UnknownMetricRecorderError
from fairseq2.metrics.text import UnknownBleuTokenizerError
from fairseq2.models import (
    InvalidModelTypeError,
    ShardedModelLoadError,
    UnknownModelArchitectureError,
    UnknownModelError,
    UnknownModelFamilyError,
)
from fairseq2.optim import UnknownOptimizerError
from fairseq2.optim.lr_scheduler import (
    UnknownLRSchedulerError,
    UnspecifiedNumberOfStepsError,
)
from fairseq2.profilers import UnknownProfilerError
from fairseq2.recipes import InconsistentGradientNormError, MinimumLossScaleReachedError
from fairseq2.recipes.asr import AsrEvalConfig, load_asr_evaluator
from fairseq2.recipes.common import (
    ActivationCheckpointingNotSupportedError,
    DatasetPathNotFoundError,
    FsdpNotSupportedError,
    HybridShardingNotSupportedError,
    InvalidModelPathError,
    ModelCompilationNotSupportedError,
    ModelParallelismNotSupportedError,
    ModelPathNotFoundError,
)
from fairseq2.recipes.lm import (
    InstructionFinetuneConfig,
    LMLossEvalConfig,
    LMTrainConfig,
    POFinetuneConfig,
    TextGenerateConfig,
    load_instruction_finetuner,
    load_lm_loss_evaluator,
    load_lm_trainer,
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
from fairseq2.utils.validation import ValidationError

# isort: split

from fairseq2.cli._cli import Cli


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

    signature = "extension_function(context: RuntimeContext, cli: Cli) -> None"

    run_extensions("fairseq2.cli", signature, context, cli)

    return cli


def _register_asr_cli(cli: Cli) -> None:
    group = cli.add_group("asr", help="ASR recipes")

    # Eval
    eval_handler = RecipeCommandHandler(
        loader=load_asr_evaluator,
        config_kls=AsrEvalConfig,
        default_preset="wav2vec2",
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

    train_handler = RecipeCommandHandler(
        loader=load_lm_trainer,
        config_kls=LMTrainConfig,
        default_preset="llama3_8b",
    )

    group.add_command(
        name="train",
        handler=train_handler,
        help="trains a language model",
    )

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
    group = cli.add_group("mt", help="machine translation recipes")

    # Eval
    eval_handler = RecipeCommandHandler(
        loader=load_mt_evaluator,
        config_kls=MTEvalConfig,
        default_preset="nllb_dense",
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
    )

    group.add_command(
        name="translate",
        handler=text_translate_handler,
        help="translate text",
    )


def _register_wav2vec2_cli(cli: Cli) -> None:
    group = cli.add_group("wav2vec2", help="wav2vec 2.0 pretraining recipes")

    # Eval
    eval_handler = RecipeCommandHandler(
        loader=load_wav2vec2_evaluator,
        config_kls=Wav2Vec2EvalConfig,
        default_preset="base_ls960h",
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
    )

    group.add_command(
        name="train",
        handler=train_handler,
        help="train a wav2vec 2.0 model",
    )


def _register_wav2vec2_asr_cli(cli: Cli) -> None:
    group = cli.add_group("wav2vec2_asr", help="wav2vec 2.0 ASR recipes")

    # Train
    train_handler = RecipeCommandHandler(
        loader=load_wav2vec2_asr_trainer,
        config_kls=Wav2Vec2AsrTrainConfig,
        default_preset="base_10h",
    )

    group.add_command(
        name="train",
        handler=train_handler,
        help="train a wav2vec 2.0 ASR model",
    )


def _register_user_error_types(cli: Cli) -> None:
    cli.register_user_error_type(ActivationCheckpointingNotSupportedError)
    cli.register_user_error_type(DatasetPathNotFoundError)
    cli.register_user_error_type(FsdpNotSupportedError)
    cli.register_user_error_type(HybridShardingNotSupportedError)
    cli.register_user_error_type(InconsistentGradientNormError)
    cli.register_user_error_type(InvalidDatasetTypeError)
    cli.register_user_error_type(InvalidModelPathError)
    cli.register_user_error_type(InvalidModelTypeError)
    cli.register_user_error_type(MinimumLossScaleReachedError)
    cli.register_user_error_type(ModelCompilationNotSupportedError)
    cli.register_user_error_type(ModelParallelismNotSupportedError)
    cli.register_user_error_type(ModelPathNotFoundError)
    cli.register_user_error_type(ShardedModelLoadError)
    cli.register_user_error_type(UnknownBeamSearchAlgorithmError)
    cli.register_user_error_type(UnknownBleuTokenizerError)
    cli.register_user_error_type(UnknownChatbotError)
    cli.register_user_error_type(UnknownDatasetError)
    cli.register_user_error_type(UnknownDatasetFamilyError)
    cli.register_user_error_type(UnknownLRSchedulerError)
    cli.register_user_error_type(UnknownMetricDescriptorError)
    cli.register_user_error_type(UnknownMetricRecorderError)
    cli.register_user_error_type(UnknownModelArchitectureError)
    cli.register_user_error_type(UnknownModelError)
    cli.register_user_error_type(UnknownModelFamilyError)
    cli.register_user_error_type(UnknownOptimizerError)
    cli.register_user_error_type(UnknownProfilerError)
    cli.register_user_error_type(UnknownSamplerError)
    cli.register_user_error_type(UnknownSeq2SeqGeneratorError)
    cli.register_user_error_type(UnknownSequenceGeneratorError)
    cli.register_user_error_type(UnknownSplitError)
    cli.register_user_error_type(UnknownTextTokenizerError)
    cli.register_user_error_type(UnknownTextTokenizerFamilyError)
    cli.register_user_error_type(UnspecifiedNumberOfStepsError)
    cli.register_user_error_type(ValidationError)
