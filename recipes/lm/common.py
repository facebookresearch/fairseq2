# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.logging import log
from fairseq2.models.clm import CausalLM
from fairseq2.nn import Embedding
from fairseq2.recipe import RecipeContext
from fairseq2.recipe.error import RecipeError


def check_model_vocabulary(context: RecipeContext) -> None:
    model = context.get_model_as(CausalLM)

    embed = _maybe_get_embed(model)
    if embed is None:
        raise RecipeError("Model must be a text-only causal language model.")

    tokenizer = context.get_tokenizer()

    vocab_info = tokenizer.vocab_info

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


def _maybe_get_embed(model: CausalLM) -> Embedding | None:
    try:
        embed = model.decoder_frontend.embed  # type: ignore[union-attr]
    except AttributeError:
        return None

    if not isinstance(embed, Embedding):
        return None

    return embed
