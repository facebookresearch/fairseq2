from typing import Any, List

import torch
from torch import Tensor

import fairseq2.nn
from fairseq2.nn import transformer
from tests import tensor_matchers as tm


@torch.inference_mode()  # type: ignore[misc]
def test_mha_is_consistent_wrt_batch_first() -> None:
    bs, l_src, dim, heads = 2, 7, 16, 4
    device = torch.device("cpu")

    attn_batch_first = transformer.StandardMultiheadAttention(
        model_dim=dim,
        num_heads=heads,
        device=device,
        dtype=torch.float32,
        batch_first=True,
    )
    attn_batch_first.eval()

    attn_batch_second = transformer.StandardMultiheadAttention(
        model_dim=dim,
        num_heads=heads,
        device=device,
        dtype=torch.float32,
        batch_first=False,
        q_proj=attn_batch_first.q_proj,
        k_proj=attn_batch_first.k_proj,
        v_proj=attn_batch_first.v_proj,
        out_proj=attn_batch_first.out_proj,
    )
    attn_batch_second.eval()

    attn_weights: List[Tensor] = []
    hook = transformer.StoreAttentionWeights(attn_weights)
    # attn_batch_first.register_attn_weight_hook(hook)
    attn_batch_first._attn_weight_hooks[0] = hook
    # attn_batch_second.register_attn_weight_hook(hook)
    attn_batch_second._attn_weight_hooks[0] = hook

    x = torch.zeros((bs, l_src, dim))
    torch.nn.init.uniform_(x, -1, 1)

    y = attn_batch_first(x, x, x)
    y_attn = attn_weights.pop()

    x_t = x.transpose(0, 1).clone()
    y_t = attn_batch_second(x_t, x_t, x_t)
    y_t_attn = attn_weights.pop()

    assert y.shape == (bs, l_src, dim)
    assert y_t.shape == (l_src, bs, dim)
    tm.assert_tensor_equals(y, y_t.transpose(1, 0), close=True)
    assert y_attn.shape == (bs * heads, l_src, l_src)
    assert y_t_attn.shape == (bs * heads, l_src, l_src)
    tm.assert_tensor_equals(y_attn, y_t_attn, close=True)
    assert not attn_weights


@torch.inference_mode()  # type: ignore[misc]
def test_enc_dec_mha_is_consistent_wrt_batch_first() -> None:
    device = torch.device("cpu")

    bs, l_src, l_tgt, dim, heads = 2, 7, 5, 16, 4
    attn_batch_first = transformer.StandardMultiheadAttention(
        model_dim=dim,
        num_heads=heads,
        device=device,
        dtype=torch.float32,
        batch_first=True,
    )
    attn_batch_first.eval()

    attn_batch_second = transformer.StandardMultiheadAttention(
        model_dim=dim,
        num_heads=heads,
        device=device,
        dtype=torch.float32,
        batch_first=False,
        q_proj=attn_batch_first.q_proj,
        k_proj=attn_batch_first.k_proj,
        v_proj=attn_batch_first.v_proj,
        out_proj=attn_batch_first.out_proj,
    )
    attn_batch_second.eval()

    attn_weights: List[Tensor] = []
    hook = transformer.StoreAttentionWeights(attn_weights)
    # attn_batch_first.register_attn_weight_hook(hook)
    attn_batch_first._attn_weight_hooks[0] = hook
    # attn_batch_second.register_attn_weight_hook(hook)
    attn_batch_second._attn_weight_hooks[0] = hook

    x_src = torch.zeros((bs, l_src, dim))
    torch.nn.init.uniform_(x_src, -1, 1)
    x_tgt = torch.zeros((bs, l_tgt, dim))
    torch.nn.init.uniform_(x_tgt, -1, 1)

    y = attn_batch_first(x_tgt, x_src, x_src)
    y_attn = attn_weights.pop()

    x_src_t = x_src.transpose(0, 1).clone()
    x_tgt_t = x_tgt.transpose(0, 1).clone()
    y_t = attn_batch_second(x_tgt_t, x_src_t, x_src_t)
    y_t_attn = attn_weights.pop()

    assert y.shape == (bs, l_tgt, dim)
    assert y_t.shape == (l_tgt, bs, dim)
    tm.assert_tensor_equals(y, y_t.transpose(1, 0), close=True)
    assert y_attn.shape == (bs * heads, l_tgt, l_src)
    tm.assert_tensor_equals(y_attn, y_t_attn, close=True)
    assert not attn_weights


@torch.inference_mode()  # type: ignore[misc]
def test_enc_dec_mha_is_consistent_wrt_inc_state(monkeypatch: Any) -> None:
    device = torch.device("cpu")

    bs, l_src, l_tgt, dim, heads = 2, 7, 5, 16, 4
    attn_batch_first = transformer.StandardMultiheadAttention(
        model_dim=dim,
        num_heads=heads,
        device=device,
        dtype=torch.float32,
        batch_first=True,
    )
    attn_batch_first.eval()
    attn_weights: List[Tensor] = []
    hook = transformer.StoreAttentionWeights(attn_weights)
    # attn_batch_first.register_attn_weight_hook(hook)
    attn_batch_first._attn_weight_hooks[0] = hook

    x_src = torch.zeros((bs, l_src, dim))
    torch.nn.init.uniform_(x_src, -1, 1)
    x_tgt = torch.zeros((bs, l_tgt, dim))
    torch.nn.init.uniform_(x_tgt, -1, 1)

    y = attn_batch_first(x_tgt, x_src, x_src)
    y_attn = attn_weights.pop()

    inc_state = fairseq2.nn.IncrementalStateBag()
    # Do a first pass, this should put the source key, values into inc_state
    y_inc = attn_batch_first(
        x_tgt[:, :1, :], x_src, x_src, incremental_state_bag=inc_state
    )
    y_inc_attn = attn_weights.pop()
    assert y_inc.shape == (bs, 1, dim)
    assert y_inc_attn.shape == (bs * heads, 1, l_src)

    # Do the rest of compute, check the cache is used
    with monkeypatch.context() as m:
        proj = attn_batch_first._forward_proj

        def check_proj_of_x_src_is_cached(
            weight: fairseq2.nn.Projection, x: Tensor
        ) -> Tensor:
            tm.assert_tensor_storage_differs(x, x_src)
            return proj(weight, x)

        m.setattr(attn_batch_first, "_forward_proj", check_proj_of_x_src_is_cached)
        for t in range(1, l_tgt):
            y_inc = attn_batch_first(
                x_tgt[:, : t + 1, :], x_src, x_src, incremental_state_bag=inc_state
            )
            y_inc_attn = attn_weights.pop()
            assert y_inc.shape == (bs, t + 1, dim)
            assert y_inc_attn.shape == (bs * heads, t + 1, l_src)

    assert y.shape == (bs, l_tgt, dim)
    assert y_inc.shape == y.shape
    assert y_attn.shape == (bs * heads, l_tgt, l_src)
    assert y_inc_attn.shape == y_attn.shape
    tm.assert_tensor_equals(y, y_inc)
    tm.assert_tensor_equals(y_attn, y_inc_attn)
    assert not attn_weights
