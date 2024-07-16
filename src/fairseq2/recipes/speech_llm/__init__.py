# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipes.cli import Cli, RecipeCommandHandler
from fairseq2.recipes.lm.chatbot import ChatbotCommandHandler

from fairseq2.recipes.speech_llm.speech_text_eval import load_speech_text_evaluator, speech_text_eval_presets
from fairseq2.recipes.speech_llm.mmlu_eval import mmlu_eval_presets, load_mmlu_evaluator
from fairseq2.recipes.speech_llm.speech_text_align import speech_text_presets, load_speech_text_trainer

def _setup_speech_llm_cli(cli: Cli) -> None:
    group = cli.add_group("speech_llm", help="Speech LLM modality-fusion recipes")
    ########################### Stage 1: Speech to text Alignment Learning ##########################
    representation_align_pretrain_handler = RecipeCommandHandler(
        loader=load_speech_text_trainer,
        preset_configs=speech_text_presets,
        default_preset="llama3_8b_speech_text_align",
    )

    group.add_command(
        name="speech_text_align",
        handler=representation_align_pretrain_handler,
        help="align speech to text repr with pre-computed alignment",
    )

    cif_representation_align_pretrain_handler = RecipeCommandHandler(
        loader=load_speech_text_trainer,
        preset_configs=speech_text_presets,
        default_preset="llama3_8b_speech_text_align_cif",
    )

    group.add_command(
        name="speech_text_cif_align",
        handler=cif_representation_align_pretrain_handler,
        help="align speech to text repr with CIF",
    )




    ########################### PPL, Similarity, ACC Eval ##########################
    eval_handler = RecipeCommandHandler(
        loader=load_speech_text_evaluator,
        preset_configs=speech_text_eval_presets,
        default_preset="librispeech_similarity",
    )

    group.add_command(
        name="eval",
        handler=eval_handler,
        help="evaluate the trained Speech-Text alignment Model",
    )

    ########################## MMLU evaluation (Speech and Text version) ##########################
    mmlu_eval_handler = RecipeCommandHandler(
        loader=load_mmlu_evaluator,
        preset_configs=mmlu_eval_presets,
        default_preset="speech_mmlu",
    )

    group.add_command(
        name="eval_mmlu",
        handler=mmlu_eval_handler,
        help="evaluate the trained Speech-Text alignment Model",
    )
