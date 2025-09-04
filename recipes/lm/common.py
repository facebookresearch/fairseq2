# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.logging import log
from fairseq2.models.transformer_lm import TransformerLM
from fairseq2.nn import Embedding
from fairseq2.recipe import RecipeContext
from fairseq2.recipe.error import RecipeError
from fairseq2.recipe.model import RecipeModel


def check_vocab_info(context: RecipeContext) -> None:
    embed = maybe_get_embed(context.model)
    if embed is None:
        raise RecipeError("Model must be a text-only Transformer language model.")

    vocab_info = context.default_tokenizer.vocab_info

    if embed.num_embeddings > vocab_info.size:
        log.warning("Vocabulary size of the tokenizer ({}) is less than the number of embeddings in the model ({}). If this is not intentional (e.g. padding for efficient GPU utilization), check your job configuration.", vocab_info.size, embed.num_embeddings)  # fmt: skip

    if embed.num_embeddings < vocab_info.size:
        raise RecipeError(
            f"Number of embeddings in the model ({embed.num_embeddings}) is less than the vocabulary size of the tokenizer ({vocab_info.size})."
        )

    if embed.pad_idx != vocab_info.pad_idx:
        raise RecipeError(
            f"Padding index in the embedding table ({embed.pad_idx}) does not match the pad token index in the tokenizer ({vocab_info.pad_idx})."
        )


def maybe_get_embed(model: RecipeModel) -> Embedding | None:
    if not isinstance(model.base_module, TransformerLM):
        return None

    try:
        embed = model.base_module.decoder_frontend.embed
    except AttributeError:
        return None

    if not isinstance(embed, Embedding):
        return None

    return embed
