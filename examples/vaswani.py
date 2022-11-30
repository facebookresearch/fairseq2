# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch

from fairseq2.nn import (
    Embedding,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TiedProjection,
)
from fairseq2.nn.transformer import (
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


@dataclass
class ModelConfig:
    """Holds the hyperparameters for the transformer model described in
    "Attention Is All You Need" (Vaswani et al., 2017)

    In this particular example, the model has only four hyperparameters as
    described in the original paper.
    """

    model_dim: int
    ffn_inner_dim: int
    num_attn_heads: int
    dropout_p: float


num_layers = 6
"""The size of the encoder and decoder stacks is fixed to 6."""


def load_embeddings(cfg: ModelConfig) -> Embedding:
    # Just a placeholder for demonstration purposes. A real implementation
    # would load the embedding table from a provided dictionary.
    return Embedding(10, embedding_dim=cfg.model_dim, scaled=True)


def build_model(cfg: ModelConfig, device: Any, dtype: Any) -> Transformer:
    """Builds a Transformer model as described in the original paper.

    In fairseq2 models are constructed by composing modules as building blocks.
    This follows the dependency inversion principle, which means instead of a
    model being responsible for instantiating its submodules, it expects them to
    be provided by the user. This avoids having to subclass or copy/edit entire
    model architectures, and gives a chance to modify the behavior of a model at
    a much granular level.
    """
    embed = load_embeddings(cfg)

    positional_embed = SinusoidalPositionalEmbedding(
        max_seq_len=4096, embedding_dim=cfg.model_dim
    )

    encoder = build_encoder(cfg, embed, positional_embed, device, dtype)

    decoder = build_decoder(cfg, embed, positional_embed, device, dtype)

    # Share the weight matrix between the embedding layers and the pre-softmax
    # score projection as described in the original paper.
    score_proj = TiedProjection(embed.weight)

    return Transformer(encoder, decoder, score_proj)


def build_encoder(
    cfg: ModelConfig,
    embed: Embedding,
    positional_embed: PositionalEmbedding,
    device: Any,
    dtype: Any,
) -> TransformerEncoder:
    layers = []

    for i in range(num_layers):
        layers.append(build_encoder_layer(cfg, device, dtype))

    return StandardTransformerEncoder(
        embed,
        positional_embed,
        layers,
        embed_dropout_p=cfg.dropout_p,
        device=device,
        dtype=dtype,
    )


def build_encoder_layer(
    cfg: ModelConfig, device: Any, dtype: Any
) -> TransformerEncoderLayer:
    # Teaser: the next example will mix MoE and distributed encoder layers for
    # demonstration purposes (e.g. ShardedFeedForwardNetwork)

    self_attn = StandardMultiheadAttention(
        model_dim=cfg.model_dim,
        num_heads=cfg.num_attn_heads,
        device=device,
        dtype=dtype,
    )

    ffn = StandardFeedForwardNetwork(
        model_dim=cfg.model_dim,
        inner_dim=cfg.ffn_inner_dim,
        device=device,
        dtype=dtype,
    )

    return StandardTransformerEncoderLayer(
        self_attn,
        ffn,
        dropout_p=cfg.dropout_p,
        device=device,
        dtype=dtype,
    )


def build_decoder(
    cfg: ModelConfig,
    embed: Embedding,
    positional_embed: PositionalEmbedding,
    device: Any,
    dtype: Any,
) -> TransformerDecoder:
    layers = []

    for i in range(num_layers):
        layers.append(build_decoder_layer(cfg, device, dtype))

    return StandardTransformerDecoder(
        embed,
        positional_embed,
        layers,
        embed_dropout_p=cfg.dropout_p,
        device=device,
        dtype=dtype,
    )


def build_decoder_layer(
    cfg: ModelConfig, device: Any, dtype: Any
) -> TransformerDecoderLayer:
    # Teaser: the next example will mix MoE and distributed decoder layers for
    # demonstration purposes (e.g. ShardedFeedForwardNetwork)

    self_attn = StandardMultiheadAttention(
        model_dim=cfg.model_dim,
        num_heads=cfg.num_attn_heads,
        device=device,
        dtype=dtype,
    )

    enc_dec_attn = StandardMultiheadAttention(
        model_dim=cfg.model_dim,
        num_heads=cfg.num_attn_heads,
        device=device,
        dtype=dtype,
    )

    ffn = StandardFeedForwardNetwork(
        model_dim=cfg.model_dim,
        inner_dim=cfg.ffn_inner_dim,
        device=device,
        dtype=dtype,
    )

    return StandardTransformerDecoderLayer(
        self_attn,
        enc_dec_attn,
        ffn,
        dropout_p=cfg.dropout_p,
        device=device,
        dtype=dtype,
    )


def get_config_for_big_variant() -> ModelConfig:
    return ModelConfig(
        model_dim=512, ffn_inner_dim=4096, num_attn_heads=16, dropout_p=0.3
    )


if __name__ == "__main__":
    # In the future this call will be integrated with Hydra once we revise the
    # configuration system.
    cfg = get_config_for_big_variant()

    # Just for demonstration purposes we initialize the model on the meta
    # device. This is now possible since all fairseq2 modules follow the
    # device/reset_parameters convention of PyTorch. As a module author this
    # gives us a chance to modify a subset of parameters if necessary
    # (e.g. convert some projections to fp16/bf16) before materializing the
    # model.
    m = build_model(cfg, "meta", torch.float32)

    print(m)
