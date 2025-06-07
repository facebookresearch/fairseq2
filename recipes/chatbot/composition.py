# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.chatbots.chatbot import Chatbot
from fairseq2.dependency import DependencyContainer, DependencyResolver
from fairseq2.models.llama import LLAMA_FAMILY
from fairseq2.models.mistral import MISTRAL_FAMILY
from fairseq2.recipe.composition import register_common_recipe_objects
from fairseq2.recipe.eval_model import load_generator_model, prepare_eval_model
from fairseq2.recipe.model import Model

from .chatbots import create_chatbot, create_llama_chatbot, create_mistral_chatbot


def register_program(container: DependencyContainer) -> None:
    register_common_recipe_objects(container)

    # Model
    def load_model(resolver: DependencyResolver) -> Model:
        model = load_generator_model(resolver)

        return prepare_eval_model(resolver, model)

    container.register(Model, load_model)

    # Chatbot
    container.register(Chatbot, create_chatbot)

    # Chatbots
    container.register(Chatbot, create_llama_chatbot, key=LLAMA_FAMILY)

    container.register(Chatbot, create_mistral_chatbot, key=MISTRAL_FAMILY)
