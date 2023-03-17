# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final, FrozenSet

from fairseq2.models.transformer import TransformerConfig
from fairseq2.nn.transformer import TransformerNormOrder

_VARIANTS: Final = frozenset(
    [
        "dense_1b",
        "dense_3b",
        "dense_distill_1b",
        "dense_distill_600m",
    ]
)

_CONFIGS: Final = {
    "dense_1b": lambda: TransformerConfig(
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
    "dense_3b": lambda: TransformerConfig(
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
    "dense_distill_1b": lambda: TransformerConfig(
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
    "dense_distill_600m": lambda: TransformerConfig(
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


def supported_nllb_variants() -> FrozenSet[str]:
    """Return the names of the supported NLLB model variants."""
    return _VARIANTS


def get_nllb_config(variant: str) -> TransformerConfig:
    """Return the configuration of the specified NLLB model variant.

    :param variant:
        The model variant.
    """
    try:
        return _CONFIGS[variant]()
    except KeyError:
        raise ValueError(f"{variant} is not a known NLLB model variant name.")
