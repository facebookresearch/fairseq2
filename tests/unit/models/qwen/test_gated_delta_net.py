# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn.functional as F

from fairseq2.models.qwen.gated_delta_net import (
    GatedDeltaNet,
    GatedDeltaNetState,
    RMSNormGated,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from fairseq2.nn import IncrementalStateBag
from tests.common import assert_close, device


class TestGatedDeltaNet:
    def test_forward_produces_correct_shape(self) -> None:
        """GatedDeltaNet forward output shape matches input shape (B, S, D)."""
        gdn = GatedDeltaNet(
            hidden_size=64,
            num_k_heads=2,
            num_v_heads=4,
            head_k_dim=16,
            head_v_dim=16,
            conv_kernel_size=4,
        )
        gdn = gdn.to(device)

        seqs = torch.randn(2, 8, 64, device=device)
        with torch.no_grad():
            out = gdn(seqs)

        assert out.shape == (2, 8, 64)

    def test_incremental_decode_matches_full_forward(self) -> None:
        """Step-by-step decode with IncrementalStateBag matches full forward."""
        gdn = GatedDeltaNet(
            hidden_size=64,
            num_k_heads=2,
            num_v_heads=4,
            head_k_dim=16,
            head_v_dim=16,
        )
        gdn = gdn.to(device).eval()

        seq_len = 8
        seqs = torch.randn(1, seq_len, 64, device=device)

        with torch.no_grad():
            full_out = gdn(seqs)

        state_bag = IncrementalStateBag(max_num_steps=seq_len)

        with torch.no_grad():
            prefill_out = gdn(seqs, state_bag=state_bag)

        assert_close(prefill_out, full_out, atol=1e-5)

    def test_chunked_vs_recurrent_consistency(self) -> None:
        """torch_chunk_gated_delta_rule and torch_recurrent_gated_delta_rule
        produce the same output for the same input."""
        B, S, H, K, V = 1, 16, 4, 16, 16
        q = torch.randn(B, S, H, K, device=device)
        k = torch.randn(B, S, H, K, device=device)
        v = torch.randn(B, S, H, V, device=device)
        g = -torch.rand(B, S, H, device=device).abs()
        beta = torch.rand(B, S, H, device=device)

        chunk_out, chunk_state = torch_chunk_gated_delta_rule(
            q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        recurrent_out, recurrent_state = torch_recurrent_gated_delta_rule(
            q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True
        )

        assert_close(chunk_out, recurrent_out, atol=1e-4)
        assert_close(chunk_state, recurrent_state, atol=1e-4)

    def test_gated_delta_net_state_reorder(self) -> None:
        """GatedDeltaNetState.reorder correctly reorders batch dimension."""
        conv = torch.randn(3, 8, 3, device=device)
        rec = torch.randn(3, 4, 16, 16, device=device)
        state = GatedDeltaNetState(conv, rec)

        new_order = torch.tensor([2, 0, 1], device=device)
        state.reorder(new_order)

        assert_close(state.conv_state[0], conv[2])
        assert_close(state.conv_state[1], conv[0])
        assert_close(state.recurrent_state[0], rec[2])

    def test_rmsnorm_gated_output(self) -> None:
        """RMSNormGated produces norm(x) * silu(gate)."""
        dim = 16
        norm = RMSNormGated(dim).to(device)

        x = torch.randn(4, dim, device=device)
        gate = torch.randn(4, dim, device=device)

        out = norm(x, gate)

        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_normed = x_f32 * torch.rsqrt(variance + 1e-6)
        expected = (norm.inner_norm.weight * x_normed) * F.silu(gate.float())

        assert_close(out, expected.to(out.dtype), atol=1e-5)

    def test_step_by_step_decode_matches_prefill(self) -> None:
        """After prefilling, incremental decode of one token matches full forward."""
        gdn = GatedDeltaNet(
            hidden_size=64,
            num_k_heads=2,
            num_v_heads=4,
            head_k_dim=16,
            head_v_dim=16,
        )
        gdn = gdn.to(device).eval()

        prefill_len = 8
        full_seq = torch.randn(1, prefill_len + 1, 64, device=device)

        with torch.no_grad():
            full_out = gdn(full_seq)
        ground_truth = full_out[:, -1:, :]

        state_bag = IncrementalStateBag(max_num_steps=prefill_len + 1)

        with torch.no_grad():
            gdn(full_seq[:, :prefill_len, :], state_bag=state_bag)

        state_bag.increment_step_nr(prefill_len)

        with torch.no_grad():
            incr_out = gdn(full_seq[:, prefill_len:, :], state_bag=state_bag)

        assert_close(incr_out, ground_truth, atol=1e-4)
