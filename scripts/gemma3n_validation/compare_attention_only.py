#!/usr/bin/env python3
"""Isolate and compare just the attention computation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.models.transformer import AttentionBiasCache
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

    test_text = "Hi"
    hf_inputs = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = hf_inputs["input_ids"]

    with torch.no_grad():
        # Get the same input to attention for both models
        hf_lm = hf_model.model.language_model
        hf_embeds = hf_lm.embed_tokens(input_ids)

        # HF: 4D stacking
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

        # HF: Get layer 0
        hf_layer = hf_lm.layers[0]
        hf_predictions = hf_layer.altup.predict(hf_hidden_4d)
        hf_active = hf_predictions[0]
        hf_active_normed = hf_layer.input_layernorm(hf_active)
        hf_laurel = hf_layer.laurel(hf_active_normed)

        # FS2: Same setup
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

        # Verify inputs to attention match
        print("="*80)
        print("Input to Attention (after input_norm + LAuReL)")
        print("="*80)
        input_diff = (hf_laurel - fs2_laurel).abs()
        print(f"Input diff: max={input_diff.max():.6e}, mean={input_diff.mean():.6e}")

        if input_diff.max() > 1e-4:
            print("⚠️  Inputs differ significantly, attention comparison may not be meaningful\n")
        else:
            print("✓ Inputs match well enough for comparison\n")

        # HF attention computation - manually run through the steps
        print("="*80)
        print("HF Attention Detailed Steps")
        print("="*80)

        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding
        rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
        cos, sin = rope(hf_laurel, position_ids, "sliding_attention")

        # Project Q, K, V
        input_shape = hf_laurel.shape[:-1]
        hidden_shape = (*input_shape, -1, config.head_dim)

        hf_q = hf_layer.self_attn.q_proj(hf_laurel).view(hidden_shape)
        hf_q = hf_layer.self_attn.q_norm(hf_q)
        print(f"Q after norm: mean={hf_q.mean():.6f}, std={hf_q.std():.6f}")

        # Apply RoPE to Q
        from transformers.models.gemma3n.modeling_gemma3n import apply_rotary_pos_emb
        hf_q = apply_rotary_pos_emb(hf_q, cos, sin, unsqueeze_dim=2)
        hf_q = hf_q.transpose(1, 2)  # (B, H, S, D)
        print(f"Q after RoPE: mean={hf_q.mean():.6f}, std={hf_q.std():.6f}")

        hf_k = hf_layer.self_attn.k_proj(hf_laurel).view(hidden_shape)
        hf_k = hf_layer.self_attn.k_norm(hf_k)
        print(f"K after norm: mean={hf_k.mean():.6f}, std={hf_k.std():.6f}")
        hf_k = apply_rotary_pos_emb(hf_k, cos, sin, unsqueeze_dim=2)
        hf_k = hf_k.transpose(1, 2)
        print(f"K after RoPE: mean={hf_k.mean():.6f}, std={hf_k.std():.6f}")

        hf_v = hf_layer.self_attn.v_proj(hf_laurel).view(hidden_shape)
        hf_v = hf_layer.self_attn.v_norm(hf_v)
        print(f"V after norm: mean={hf_v.mean():.6f}, std={hf_v.std():.6f}")
        hf_v = hf_v.transpose(1, 2)

        # Repeat K, V for GQA
        from transformers.models.gemma3n.modeling_gemma3n import repeat_kv
        hf_k = repeat_kv(hf_k, config.num_attn_heads // config.num_key_value_heads)
        hf_v = repeat_kv(hf_v, config.num_attn_heads // config.num_key_value_heads)

        # Compute attention weights
        scaling = config.head_dim ** -0.5
        hf_attn_weights = torch.matmul(hf_q, hf_k.transpose(2, 3)) * scaling
        print(f"\nAttention weights (pre-mask): mean={hf_attn_weights.mean():.6f}, std={hf_attn_weights.std():.6f}")

        # Apply sliding window mask
        # (Note: HF might have attention_mask=None, using is_causal or sliding_window in SDPA)
        # For manual computation, we need to create the mask
        seq_len = hf_q.size(2)
        sliding_window = 512
        from torch.nn.functional import softmax

        # Create causal mask with sliding window
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )
        # Apply sliding window: mask out positions beyond window
        if sliding_window < seq_len:
            window_mask = torch.tril(
                torch.full((seq_len, seq_len), float('-inf'), device=device),
                diagonal=-sliding_window
            )
            causal_mask = causal_mask + window_mask

        hf_attn_weights = hf_attn_weights + causal_mask
        print(f"Attention weights (post-mask): mean={hf_attn_weights.mean():.6f}, std={hf_attn_weights.std():.6f}")

        # Softmax (with upcast to fp32)
        hf_attn_weights = softmax(hf_attn_weights, dim=-1, dtype=torch.float32).to(hf_q.dtype)
        print(f"Attention probs: mean={hf_attn_weights.mean():.6f}, std={hf_attn_weights.std():.6f}")

        # Apply to values
        hf_attn_manual = torch.matmul(hf_attn_weights, hf_v)
        hf_attn_manual = hf_attn_manual.transpose(1, 2).contiguous()
        hf_attn_manual = hf_attn_manual.reshape(*input_shape, -1)
        hf_attn_manual = hf_layer.self_attn.o_proj(hf_attn_manual)

        print(f"HF manual attn output: mean={hf_attn_manual.mean():.6f}, std={hf_attn_manual.std():.6f}")

        # FS2 attention
        print("\n" + "="*80)
        print("FS2 Attention via StandardMultiheadAttention")
        print("="*80)

        attn_bias_cache = AttentionBiasCache()
        fs2_attn = fs2_layer.self_attn(
            fs2_laurel,
            batch_layout,
            keys=fs2_laurel,
            keys_layout=batch_layout,
            values=fs2_laurel,
            bias_cache=attn_bias_cache,
            state_bag=state_bag,
        )

        print(f"FS2 attn output: mean={fs2_attn.mean():.6f}, std={fs2_attn.std():.6f}")

        # Compare
        print("\n" + "="*80)
        print("Comparison")
        print("="*80)
        diff = (hf_attn_manual - fs2_attn).abs()
        print(f"Attention output diff: max={diff.max():.6e}, mean={diff.mean():.6e}")

        if diff.max() < 1e-5:
            print("✓ Attention outputs match!")
        elif diff.max() < 1e-3:
            print("⚠️  Small difference in attention outputs")
        else:
            print("❌ Significant difference in attention outputs")


if __name__ == "__main__":
    main()
