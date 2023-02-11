from typing import List

import torch
from torch import Tensor

from fairseq2.nn import IncrementalStateBag
from fairseq2.nn.transformer import StandardMultiheadAttention, StoreAttentionWeights
from tests.common import assert_close, device


class TestMultiheadAttention:
    @torch.inference_mode()
    def test_mha_is_consistent_wrt_batch_first(self) -> None:
        bs, l_src, dim, heads = 2, 7, 16, 4

        attn_batch_first = StandardMultiheadAttention(
            model_dim=dim,
            num_heads=heads,
            device=device,
            dtype=torch.float32,
            batch_first=True,
        )
        attn_batch_first.eval()

        attn_batch_second = StandardMultiheadAttention(
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
        hook = StoreAttentionWeights(attn_weights)
        attn_batch_first.register_attn_weight_hook(hook)
        attn_batch_second.register_attn_weight_hook(hook)

        x = torch.zeros((bs, l_src, dim), device=device)
        torch.nn.init.uniform_(x, -1, 1)

        y = attn_batch_first(x, x, x)
        y_attn = attn_weights.pop()

        x_t = x.transpose(0, 1).clone()
        y_t = attn_batch_second(x_t, x_t, x_t)
        y_t_attn = attn_weights.pop()

        assert y.shape == (bs, l_src, dim)
        assert y_t.shape == (l_src, bs, dim)
        assert_close(y, y_t.transpose(1, 0))
        assert y_attn.shape == (bs * heads, l_src, l_src)
        assert y_t_attn.shape == (bs * heads, l_src, l_src)
        assert_close(y_attn, y_t_attn)
        assert not attn_weights

    @torch.inference_mode()
    def test_enc_dec_mha_is_consistent_wrt_batch_first(self) -> None:
        bs, l_src, l_tgt, dim, heads = 2, 7, 5, 16, 4

        attn_batch_first = StandardMultiheadAttention(
            model_dim=dim,
            num_heads=heads,
            device=device,
            dtype=torch.float32,
            batch_first=True,
        )
        attn_batch_first.eval()

        attn_batch_second = StandardMultiheadAttention(
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
        hook = StoreAttentionWeights(attn_weights)
        # attn_batch_first.register_attn_weight_hook(hook)
        attn_batch_first._attn_weight_hooks[0] = hook
        # attn_batch_second.register_attn_weight_hook(hook)
        attn_batch_second._attn_weight_hooks[0] = hook

        x_src = torch.zeros((bs, l_src, dim), device=device)
        torch.nn.init.uniform_(x_src, -1, 1)
        x_tgt = torch.zeros((bs, l_tgt, dim), device=device)
        torch.nn.init.uniform_(x_tgt, -1, 1)

        y = attn_batch_first(x_tgt, x_src, x_src)
        y_attn = attn_weights.pop()

        x_src_t = x_src.transpose(0, 1).clone()
        x_tgt_t = x_tgt.transpose(0, 1).clone()
        y_t = attn_batch_second(x_tgt_t, x_src_t, x_src_t)
        y_t_attn = attn_weights.pop()

        assert y.shape == (bs, l_tgt, dim)
        assert y_t.shape == (l_tgt, bs, dim)
        assert_close(y, y_t.transpose(1, 0))
        assert y_attn.shape == (bs * heads, l_tgt, l_src)
        assert_close(y_attn, y_t_attn)
        assert not attn_weights

    @torch.inference_mode()
    def test_enc_dec_mha_is_consistent_wrt_inc_state(self) -> None:
        bs, l_src, l_tgt, dim, heads = 2, 7, 5, 16, 4

        attn_batch_first = StandardMultiheadAttention(
            model_dim=dim,
            num_heads=heads,
            device=device,
            dtype=torch.float32,
            batch_first=True,
        )
        attn_batch_first.eval()
        attn_weights: List[Tensor] = []
        hook = StoreAttentionWeights(attn_weights)
        # attn_batch_first.register_attn_weight_hook(hook)
        attn_batch_first._attn_weight_hooks[0] = hook

        x_src = torch.zeros((bs, l_src, dim), device=device)
        torch.nn.init.uniform_(x_src, -1, 1)
        x_tgt = torch.zeros((bs, l_tgt, dim), device=device)
        torch.nn.init.uniform_(x_tgt, -1, 1)

        # y = attn_batch_first(x_tgt, x_src, x_src)
        # y_attn = attn_weights.pop()

        state_bag = IncrementalStateBag()
        # Do a first pass, this should put the source key, values into state_bag
        y_inc = attn_batch_first(x_tgt[:, :1, :], x_src, x_src, state_bag=state_bag)
        y_inc_attn = attn_weights.pop()
        assert y_inc.shape == (bs, 1, dim)
        assert y_inc_attn.shape == (bs * heads, 1, l_src)

        # Do the rest of compute, check the cache is used


#        with monkeypatch.context() as m:
#            proj = attn_batch_first._forward_proj
#
#            def check_proj_of_x_src_is_cached(
#                weight: fairseq2.nn.Projection, x: Tensor
#            ) -> Tensor:
#                self.assertNotEqual(x.storage(), x_src.storage())
#                return proj(weight, x)
#
#            m.setattr(attn_batch_first, "_forward_proj", check_proj_of_x_src_is_cached)
#            for t in range(1, l_tgt):
#                y_inc = attn_batch_first(
#                    x_tgt[:, : t + 1, :], x_src, x_src, incremental_state_bag=state_bag
#                )
#                y_inc_attn = attn_weights.pop()
#                assert y_inc.shape == (bs, t + 1, dim)
#                assert y_inc_attn.shape == (bs * heads, t + 1, l_src)
#
#        assert y.shape == (bs, l_tgt, dim)
#        assert y_inc.shape == y.shape
#        assert y_attn.shape == (bs * heads, l_tgt, l_src)
#        assert y_inc_attn.shape == y_attn.shape
#        self.assertAllClose(y, y_inc)
#        self.assertAllClose(y_attn, y_inc_attn)
#        assert not attn_weights
