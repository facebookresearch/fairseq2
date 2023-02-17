# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, List

import pytest
import torch

from fairseq2.generate.search import BeamSearchStrategy
from fairseq2.generate.tokenizer import TokenMeta
from fairseq2.models.transformer import (
    Transformer,
    TransformerConfig,
    build_transformer,
)
from fairseq2.nn.transformer import StoreAttentionWeights

VOCAB_SIZE = 111


@functools.lru_cache()
def build_model() -> Transformer:
    cfg = TransformerConfig(
        src_num_tokens=VOCAB_SIZE,
        tgt_num_tokens=VOCAB_SIZE,
        src_padding_token_idx=None,
        tgt_padding_token_idx=None,
        max_src_len=64,
        max_tgt_len=64,
        model_dim=16,
        num_enc_layers=2,
        num_dec_layers=2,
        ffn_inner_dim=32,
    )

    return build_transformer(cfg)


@pytest.mark.parametrize("prefix_tokens", [None, 99, [99, 17], [[99, 17], [99, 18]]])
def test_generate(prefix_tokens: Any) -> None:
    m = build_model()

    src_len, tgt_len = (4, 6)
    token_meta = TokenMeta(vocab_size=VOCAB_SIZE, BOS=0, EOS=1, UNK=2, PAD=3)

    bs = BeamSearchStrategy(token_meta=token_meta, max_len=tgt_len, beam_size=2)
    src_tokens = torch.tensor([[1, 2, 3, 3], [7, 8, 9, 3]], dtype=torch.int64)

    attn_weights: List[torch.Tensor] = []
    hook = StoreAttentionWeights(attn_weights)
    m.decoder.layers[0].enc_dec_attn.register_attn_weight_hook(hook)  # type: ignore[index,union-attr]

    if prefix_tokens is not None:
        prefix_tokens = torch.tensor(prefix_tokens)

    tgt_tokens = bs.generate(
        m, src_tokens=src_tokens, prefix_tokens=prefix_tokens, top=1
    )

    # We should generate one step per len
    assert len(attn_weights) == tgt_len
    for i in range(tgt_len):
        assert attn_weights[i].shape == (32, 1, src_len)

    if prefix_tokens is None:
        assert torch.all(tgt_tokens[:, 0] == token_meta.BOS)
    elif prefix_tokens.ndim == 0:
        assert torch.all(tgt_tokens[:, 0] == prefix_tokens)
    else:
        assert torch.all(tgt_tokens[:, : prefix_tokens.size(-1)] == prefix_tokens)
