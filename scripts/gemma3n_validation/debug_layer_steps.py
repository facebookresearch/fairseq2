#!/usr/bin/env python3
"""Step-by-step comparison of layer 0 forward pass."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag


def main() -> None:
    device = torch.device("cpu")  # Use CPU for easier debugging

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
        # Setup: embeddings + 4D stacking + PLE
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

        # HF PLE
        hf_ple_raw = hf_lm.embed_tokens_per_layer(input_ids)
        hf_ple_discrete = hf_ple_raw.reshape(*input_ids.shape, 30, 256)
        hf_ple_cont = hf_lm.per_layer_model_projection(hf_embeds) * (2048**-0.5)
        hf_ple_cont = hf_ple_cont.reshape(*hf_embeds.shape[:-1], 30, 256)
        hf_ple_cont = hf_lm.per_layer_projection_norm(hf_ple_cont)
        hf_ple = (hf_ple_discrete + hf_ple_cont) / (2**0.5)
        hf_ple_layer0 = hf_ple[:, :, 0, :]

        # FS2 setup
        seq_lens = [input_ids.shape[1]]
        batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
        state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
        _, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
        fs2_ple_layer0 = state_bag.per_layer_inputs[:, :, 0, :]
        fs2_hidden_4d = fs2_model.decoder._stack_altup(fs2_model.decoder_frontend.embed(input_ids) * fs2_model.decoder_frontend.scale)

        print("Input 4D shapes match:", hf_hidden_4d.shape == fs2_hidden_4d.shape)
        print(f"Input 4D diff: {(hf_hidden_4d - fs2_hidden_4d).abs().max():.6e}\n")

        # Step 1: AltUp Predict
        print("="*80)
        print("Step 1: AltUp Predict")
        print("="*80)

        hf_layer = hf_lm.layers[0]
        fs2_layer = fs2_model.decoder.layers[0]

        hf_predictions = hf_layer.altup.predict(hf_hidden_4d)
        fs2_predictions = fs2_layer.altup(fs2_hidden_4d)

        print(f"HF predictions:  shape={hf_predictions.shape}, mean={hf_predictions.mean():.6f}")
        print(f"FS2 predictions: shape={fs2_predictions.shape}, mean={fs2_predictions.mean():.6f}")
        print(f"Diff: {(hf_predictions - fs2_predictions).abs().max():.6e}\n")

        # Step 2: Extract active version
        hf_active = hf_predictions[0]
        fs2_active = fs2_predictions[0]
        print(f"Active version diff: {(hf_active - fs2_active).abs().max():.6e}\n")

        # Step 3: Input normalization
        print("="*80)
        print("Step 3: Input Normalization")
        print("="*80)

        hf_active_normed = hf_layer.input_layernorm(hf_active)
        fs2_active_normed = fs2_layer.input_layernorm(fs2_active)

        print(f"HF normed:  mean={hf_active_normed.mean():.6f}, std={hf_active_normed.std():.6f}")
        print(f"FS2 normed: mean={fs2_active_normed.mean():.6f}, std={fs2_active_normed.std():.6f}")
        print(f"Diff: {(hf_active_normed - fs2_active_normed).abs().max():.6e}\n")

        # Step 4: LAuReL
        print("="*80)
        print("Step 4: LAuReL")
        print("="*80)

        hf_laurel = hf_layer.laurel(hf_active_normed)
        fs2_laurel = fs2_layer.laurel(fs2_active_normed)

        print(f"HF LAuReL:  mean={hf_laurel.mean():.6f}, std={hf_laurel.std():.6f}")
        print(f"FS2 LAuReL: mean={fs2_laurel.mean():.6f}, std={fs2_laurel.std():.6f}")
        print(f"Diff: {(hf_laurel - fs2_laurel).abs().max():.6e}\n")

        # Check: Input to attention (should be LAuReL output, not just normed)
        print("="*80)
        print("Input to Attention Check (LAuReL output)")
        print("="*80)
        print(f"HF LAuReL input to attn: mean={hf_laurel.mean():.6f}, std={hf_laurel.std():.6f}")
        print(f"FS2 LAuReL input to attn: mean={fs2_laurel.mean():.6f}, std={fs2_laurel.std():.6f}")
        attn_input_diff = (hf_laurel - fs2_laurel).abs()
        print(f"Attention input diff: max={attn_input_diff.max():.6e}, mean={attn_input_diff.mean():.6e}")
        if attn_input_diff.max() > 1e-4:
            print("⚠️  Inputs to attention differ significantly!\n")
        else:
            print("✓ Inputs to attention match\n")

        # Step 5: Attention
        print("="*80)
        print("Step 5: Attention")
        print("="*80)

        # Get position embeddings for local layer
        layer_type = hf_lm.config.layer_types[0]
        print(f"Layer 0 attention type: {layer_type}")

        # HF attention - manual computation to match compare_attention_only.py
        from transformers.models.gemma3n.modeling_gemma3n import (
            Gemma3nRotaryEmbedding, apply_rotary_pos_emb, repeat_kv
        )
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
        cos, sin = rope(hf_laurel, position_ids, layer_type)

        input_shape = hf_laurel.shape[:-1]
        hidden_shape = (*input_shape, -1, config.head_dim)

        # HF manual SDPA
        hf_q = hf_layer.self_attn.q_proj(hf_laurel).view(hidden_shape)
        hf_q = hf_layer.self_attn.q_norm(hf_q)
        hf_q = apply_rotary_pos_emb(hf_q, cos, sin, unsqueeze_dim=2).transpose(1, 2)

        hf_k = hf_layer.self_attn.k_proj(hf_laurel).view(hidden_shape)
        hf_k = hf_layer.self_attn.k_norm(hf_k)
        hf_k = apply_rotary_pos_emb(hf_k, cos, sin, unsqueeze_dim=2).transpose(1, 2)

        hf_v = hf_layer.self_attn.v_proj(hf_laurel).view(hidden_shape)
        hf_v = hf_layer.self_attn.v_norm(hf_v).transpose(1, 2)

        hf_k = repeat_kv(hf_k, 4)
        hf_v = repeat_kv(hf_v, 4)

        scaling = config.head_dim ** -0.5
        hf_attn_weights = torch.matmul(hf_q, hf_k.transpose(2, 3)) * scaling

        # Apply causal mask
        seq_len = hf_q.size(2)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
        hf_attn_weights = hf_attn_weights + causal_mask

        hf_attn_weights = torch.nn.functional.softmax(hf_attn_weights, dim=-1, dtype=torch.float32).to(hf_q.dtype)
        hf_attn = torch.matmul(hf_attn_weights, hf_v).transpose(1, 2).contiguous()
        hf_attn = hf_attn.reshape(*input_shape, -1)
        hf_attn = hf_layer.self_attn.o_proj(hf_attn)

        # FS2 attention
        from fairseq2.models.transformer import AttentionBiasCache
        attn_bias_cache = AttentionBiasCache()

        fs2_attn = fs2_layer.self_attn(
            fs2_laurel,  # Use LAuReL output
            batch_layout,
            keys=fs2_laurel,  # Use LAuReL output
            keys_layout=batch_layout,
            values=fs2_laurel,  # Use LAuReL output
            bias_cache=attn_bias_cache,
            state_bag=state_bag,  # Use same state_bag as frontend
        )

        print(f"HF attention:  mean={hf_attn.mean():.6f}, std={hf_attn.std():.6f}")
        print(f"FS2 attention: mean={fs2_attn.mean():.6f}, std={fs2_attn.std():.6f}")
        print(f"Diff: {(hf_attn - fs2_attn).abs().max():.6e}\n")

        # Step 6: Post-attention norm
        print("="*80)
        print("Step 6: Post-Attention Normalization")
        print("="*80)

        hf_attn_norm = hf_layer.post_attention_layernorm(hf_attn)
        fs2_attn_norm = fs2_layer.post_attention_layernorm(fs2_attn)

        print(f"HF post-attn norm:  mean={hf_attn_norm.mean():.6f}, std={hf_attn_norm.std():.6f}")
        print(f"FS2 post-attn norm: mean={fs2_attn_norm.mean():.6f}, std={fs2_attn_norm.std():.6f}")
        print(f"Diff: {(hf_attn_norm - fs2_attn_norm).abs().max():.6e}\n")


if __name__ == "__main__":
    main()
