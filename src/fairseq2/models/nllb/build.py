# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from overrides import override

from fairseq2.data.text import Tokenizer
from fairseq2.models.transformer import (
    TransformerBuilder,
    TransformerConfig,
    TransformerModel,
)
from fairseq2.nn.transformer import MultiheadAttention, StandardMultiheadAttention


def create_nllb_model(
    cfg: TransformerConfig,
    tokenizer: Tokenizer,
    device: Optional[torch.device] = None,
) -> TransformerModel:
    """Create a model that follows the NLLB architecture as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2207.04672`.

    :param cfg:
        The configuration to use.
    :param tokenizer:
        The tokenizer that holds the vocabulary information to use.
    """
    return NllbBuilder(cfg, tokenizer.vocab_info, device).build_model()


class NllbBuilder(TransformerBuilder):
    """Builds models that follow the NLLB architecture as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2207.04672`.

    To tweak the model architecture, you can derive from this class and override
    the corresponding methods.
    """

    @override
    def build_attention(self, num_heads: int) -> MultiheadAttention:
        # NLLB applies dropout to attention weights as well.
        return StandardMultiheadAttention(
            num_heads=num_heads,
            model_dim=self.cfg.model_dim,
            attn_dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.cfg.dtype,
        )
