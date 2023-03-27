# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final, Optional

import torch
from overrides import override

from fairseq2.data.text import VocabularyInfo
from fairseq2.models.transformer import (
    TransformerBuilder,
    TransformerConfig,
    TransformerModel,
)
from fairseq2.nn.transformer import (
    MultiheadAttention,
    StandardMultiheadAttention,
    TransformerNormOrder,
)

_CONFIGS: Final = {
    "1b": lambda: TransformerConfig(
        model_dim=1024,
        num_enc_layers=24,
        num_dec_layers=24,
        num_enc_attn_heads=16,
        num_dec_attn_heads=16,
        ffn_inner_dim=1024 * 8,
        dropout_p=0.1,
        norm_order=TransformerNormOrder.PRE,
        legacy_pos_embed=True,
    ),
    "3b": lambda: TransformerConfig(
        model_dim=2048,
        num_enc_layers=24,
        num_dec_layers=24,
        num_enc_attn_heads=16,
        num_dec_attn_heads=16,
        ffn_inner_dim=1024 * 8,
        dropout_p=0.1,
        norm_order=TransformerNormOrder.PRE,
        legacy_pos_embed=True,
    ),
    "600m": lambda: TransformerConfig(
        model_dim=1024,
        num_enc_layers=12,
        num_dec_layers=12,
        num_enc_attn_heads=16,
        num_dec_attn_heads=16,
        ffn_inner_dim=1024 * 4,
        dropout_p=0.1,
        norm_order=TransformerNormOrder.PRE,
        legacy_pos_embed=True,
    ),
}


def get_nllb_config(arch_name: str) -> TransformerConfig:
    """Return the configuration of the specified NLLB architecture.

    :param arch_name:
        The name of the architecture.
    """
    try:
        return _CONFIGS[arch_name]()
    except KeyError:
        raise ValueError(f"{arch_name} is not a known NLLB architecture.")


def create_nllb_model(
    cfg: TransformerConfig,
    vocab_info: VocabularyInfo,
    device: Optional[torch.device] = None,
) -> TransformerModel:
    """Create an NLLB model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2207.04672`.

    :param cfg:
        The configuration to use.
    :param tokenizer:
        The vocabulary information to use.
    :param device:
        The device on which to initialize the model.
    """
    return NllbBuilder(cfg, vocab_info, device).build_model()


class NllbBuilder(TransformerBuilder):
    """Builds modules of an NLLB model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2207.04672`.

    To tweak the model architecture, you can derive from this class and override
    the corresponding methods.
    """

    @override
    def build_attention(self, num_heads: int) -> MultiheadAttention:
        # NLLB applies dropout to attention weights as well.
        return StandardMultiheadAttention(
            num_heads,
            self.cfg.model_dim,
            attn_dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.cfg.dtype,
        )
