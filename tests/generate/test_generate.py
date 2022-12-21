import functools
from typing import Any, List

import pytest
import torch

from fairseq2.generate.search import BeamSearchStrategy
from fairseq2.generate.tokenizer import TokenMeta
from fairseq2.nn import transformer

VOCAB_SIZE = 111


@functools.lru_cache()
def build_model() -> transformer.Transformer:
    builder = transformer.TransformerBuilder(
        num_enc_layers=2,
        num_dec_layers=2,
        num_tokens=VOCAB_SIZE,
        model_dim=16,
        ffn_inner_dim=32,
        max_seq_len=64,
    )
    return builder.build()


@pytest.mark.parametrize("prefix_tokens", [None, 99, [99, 17], [[99, 17], [99, 18]]])
def test_generate(prefix_tokens: Any) -> None:
    m = build_model()

    src_len, tgt_len = (4, 6)
    token_meta = TokenMeta(vocab_size=VOCAB_SIZE, BOS=0, EOS=1, UNK=2, PAD=3)

    bs = BeamSearchStrategy(token_meta=token_meta, max_len=tgt_len, beam_size=2)
    src_tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], dtype=torch.int64)

    attn_weights: List[torch.Tensor] = []
    hook = transformer.StoreAttentionWeights(attn_weights)
    m.decoder.layers[0].enc_dec_attn.register_attn_weight_hook(hook)  # type: ignore

    if prefix_tokens is not None:
        prefix_tokens = torch.tensor(prefix_tokens)

    tgt_tokens = bs.generate(
        m, src_tokens=src_tokens, prefix_tokens=prefix_tokens, top=1
    )

    # We should generate one step per len
    assert len(attn_weights) == tgt_len
    for i in range(tgt_len):
        assert attn_weights[i].shape == (32, i + 1, src_len)

    if prefix_tokens is None:
        assert torch.all(tgt_tokens[:, 0] == token_meta.BOS)
    elif prefix_tokens.ndim == 0:
        assert torch.all(tgt_tokens[:, 0] == prefix_tokens)
    else:
        assert torch.all(tgt_tokens[:, : prefix_tokens.size(-1)] == prefix_tokens)
