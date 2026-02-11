#!/usr/bin/env python3
"""Deep dive: compare Q, K, V, and mask tensors before SDPA."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.models.transformer import AttentionBiasCache, CausalAttentionBias
from fairseq2.nn import BatchLayout, IncrementalStateBag


def main() -> None:
    device = torch.device("cpu")

    print("Loading models...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3n-E2B-it",
        torch_dtype=torch.float32,
        device_map=device,
        local_files_only=True,
    )
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)

    config = get_gemma3n_e2b_config()
    fs2_model = create_gemma3n_model(config, device=device, dtype=torch.float32)
    fs2_model.eval()

    hf_state_dict = hf_model.state_dict()
    fs2_state_dict = convert_gemma3n_state_dict(hf_state_dict, config)
    fs2_model.load_state_dict(fs2_state_dict, strict=False)
    print("✓ Models loaded\n")

    test_text = "Hello world"  # Use longer text to test sliding window
    hf_inputs = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = hf_inputs["input_ids"]
    seq_len = input_ids.shape[1]

    print(f"Test text: '{test_text}'")
    print(f"Sequence length: {seq_len}\n")

    with torch.no_grad():
        # Setup: get normalized input to attention
        hf_lm = hf_model.model.language_model
        hf_embeds = hf_lm.embed_tokens(input_ids)

        # HF 4D stacking
        hidden_states_0 = hf_embeds
        target_magnitude = torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
        epsilon_tensor = torch.tensor(1e-5, device=device, dtype=torch.float32)
        temp_hidden_states = [hidden_states_0]
        for i in range(1, 4):
            altup_proj = hf_lm.altup_projections[i - 1](hidden_states_0)
            new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
            altup_proj = altup_proj * target_magnitude / new_magnitude
            temp_hidden_states.append(altup_proj)
        hf_hidden_4d = torch.stack(temp_hidden_states, dim=0)

        hf_layer = hf_lm.layers[0]
        hf_predictions = hf_layer.altup.predict(hf_hidden_4d)
        hf_active = hf_predictions[0]
        hf_active_normed = hf_layer.input_layernorm(hf_active)
        hf_laurel = hf_layer.laurel(hf_active_normed)

        # FS2 setup
        seq_lens = [input_ids.shape[1]]
        batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
        state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
        _, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
        fs2_hidden_4d = fs2_model.decoder._stack_altup(
            fs2_model.decoder_frontend.embed(input_ids) * fs2_model.decoder_frontend.scale
        )

        fs2_layer = fs2_model.decoder.layers[0]
        fs2_predictions = fs2_layer.altup(fs2_hidden_4d)
        fs2_active = fs2_predictions[0]
        fs2_active_normed = fs2_layer.input_layernorm(fs2_active)
        fs2_laurel = fs2_layer.laurel(fs2_active_normed)

        # HF: Project Q, K, V (before transpose)
        print("="*80)
        print("HF: Q, K, V before transpose (shape: B, S, H, D)")
        print("="*80)

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding, apply_rotary_pos_emb
        rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
        cos, sin = rope(hf_laurel, position_ids, "sliding_attention")

        input_shape = hf_laurel.shape[:-1]
        hidden_shape = (*input_shape, -1, config.head_dim)

        hf_q = hf_layer.self_attn.q_proj(hf_laurel).view(hidden_shape)
        hf_q = hf_layer.self_attn.q_norm(hf_q)
        hf_q = apply_rotary_pos_emb(hf_q, cos, sin, unsqueeze_dim=2)
        print(f"Q (B,S,H,D): shape={hf_q.shape}, mean={hf_q.mean():.6f}, std={hf_q.std():.6f}")

        hf_k = hf_layer.self_attn.k_proj(hf_laurel).view(hidden_shape)
        hf_k = hf_layer.self_attn.k_norm(hf_k)
        hf_k = apply_rotary_pos_emb(hf_k, cos, sin, unsqueeze_dim=2)
        print(f"K (B,S,H,D): shape={hf_k.shape}, mean={hf_k.mean():.6f}, std={hf_k.std():.6f}")

        hf_v = hf_layer.self_attn.v_proj(hf_laurel).view(hidden_shape)
        hf_v = hf_layer.self_attn.v_norm(hf_v)
        print(f"V (B,S,H,D): shape={hf_v.shape}, mean={hf_v.mean():.6f}, std={hf_v.std():.6f}")

        # FS2: Get Q, K, V via _project_q and _project_kv
        print("\n" + "="*80)
        print("FS2: Q, K, V (shape: B, S, H, D)")
        print("="*80)

        fs2_q = fs2_layer.self_attn._project_q(fs2_laurel, batch_layout, state_bag)
        print(f"Q (B,S,H,D): shape={fs2_q.shape}, mean={fs2_q.mean():.6f}, std={fs2_q.std():.6f}")

        fs2_k, fs2_v = fs2_layer.self_attn._project_kv(fs2_laurel, batch_layout, fs2_laurel, state_bag)
        print(f"K (B,S,H,D): shape={fs2_k.shape}, mean={fs2_k.mean():.6f}, std={fs2_k.std():.6f}")
        print(f"V (B,S,H,D): shape={fs2_v.shape}, mean={fs2_v.mean():.6f}, std={fs2_v.std():.6f}")

        # Compare Q, K, V before GQA repeat
        print("\n" + "="*80)
        print("Comparison before GQA repeat")
        print("="*80)
        q_diff = (hf_q - fs2_q).abs()
        k_diff = (hf_k - fs2_k).abs()
        v_diff = (hf_v - fs2_v).abs()
        print(f"Q diff: max={q_diff.max():.6e}, mean={q_diff.mean():.6e}")
        print(f"K diff: max={k_diff.max():.6e}, mean={k_diff.mean():.6e}")
        print(f"V diff: max={v_diff.max():.6e}, mean={v_diff.mean():.6e}")

        # HF: Repeat for GQA and transpose
        from transformers.models.gemma3n.modeling_gemma3n import repeat_kv
        hf_q = hf_q.transpose(1, 2)  # (B, H, S, D)
        hf_k = hf_k.transpose(1, 2)
        hf_v = hf_v.transpose(1, 2)

        num_groups = config.num_attn_heads // config.num_key_value_heads
        hf_k = repeat_kv(hf_k, num_groups)
        hf_v = repeat_kv(hf_v, num_groups)

        print("\n" + "="*80)
        print("HF: After GQA repeat and transpose (B, H, S, D)")
        print("="*80)
        print(f"Q: shape={hf_q.shape}, mean={hf_q.mean():.6f}")
        print(f"K: shape={hf_k.shape}, mean={hf_k.mean():.6f}")
        print(f"V: shape={hf_v.shape}, mean={hf_v.mean():.6f}")

        # FS2: Repeat for GQA (before transpose - will be transposed in SDPA)
        from fairseq2.nn.utils.mask import repeat_interleave
        fs2_k_repeated = repeat_interleave(fs2_k, dim=-2, repeat=fs2_layer.self_attn.num_query_groups)
        fs2_v_repeated = repeat_interleave(fs2_v, dim=-2, repeat=fs2_layer.self_attn.num_query_groups)

        # Transpose to match HF format for comparison
        fs2_q_t = fs2_q.transpose(-2, -3)
        fs2_k_t = fs2_k_repeated.transpose(-2, -3)
        fs2_v_t = fs2_v_repeated.transpose(-2, -3)

        print("\n" + "="*80)
        print("FS2: After GQA repeat and transpose (B, H, S, D)")
        print("="*80)
        print(f"Q: shape={fs2_q_t.shape}, mean={fs2_q_t.mean():.6f}")
        print(f"K: shape={fs2_k_t.shape}, mean={fs2_k_t.mean():.6f}")
        print(f"V: shape={fs2_v_t.shape}, mean={fs2_v_t.mean():.6f}")

        # Compare after GQA
        print("\n" + "="*80)
        print("Comparison after GQA repeat and transpose")
        print("="*80)
        q_diff = (hf_q - fs2_q_t).abs()
        k_diff = (hf_k - fs2_k_t).abs()
        v_diff = (hf_v - fs2_v_t).abs()
        print(f"Q diff: max={q_diff.max():.6e}, mean={q_diff.mean():.6e}")
        print(f"K diff: max={k_diff.max():.6e}, mean={k_diff.mean():.6e}")
        print(f"V diff: max={v_diff.max():.6e}, mean={v_diff.mean():.6e}")

        # Compare masks
        print("\n" + "="*80)
        print("Attention Mask Comparison")
        print("="*80)

        # HF mask (sliding window causal)
        sliding_window = 512
        hf_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=torch.float32),
            diagonal=1
        )
        if sliding_window < seq_len:
            window_mask = torch.tril(
                torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=torch.float32),
                diagonal=-sliding_window
            )
            hf_mask = hf_mask + window_mask

        print(f"HF mask shape: {hf_mask.shape}")
        print(f"HF mask (first 5x5):\n{hf_mask[:5, :5]}")

        # FS2 mask
        bias = CausalAttentionBias(attn_window_len=512)
        fs2_mask_raw = bias.create_bias_tensor(seq_len, seq_len, device, torch.float32)
        print(f"\nFS2 bias shape: {fs2_mask_raw.shape}")
        print(f"FS2 bias (first 5x5):\n{fs2_mask_raw[:5, :5]}")

        # Convert FS2 bias (0/1) to mask (0/-inf)
        fs2_mask = torch.where(
            fs2_mask_raw == 0,
            torch.tensor(float('-inf'), device=device, dtype=torch.float32),
            torch.tensor(0.0, device=device, dtype=torch.float32)
        )
        print(f"\nFS2 mask converted (first 5x5):\n{fs2_mask[:5, :5]}")

        mask_diff = (hf_mask - fs2_mask).abs()
        mask_match = torch.allclose(hf_mask, fs2_mask, atol=1e-6)
        print(f"\nMask diff: max={mask_diff.max():.6e}, matches={mask_match}")


if __name__ == "__main__":
    main()
