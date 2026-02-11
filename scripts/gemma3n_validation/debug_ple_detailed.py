#!/usr/bin/env python3
"""Compare PLE computation step-by-step between HF and FS2."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag

device = torch.device("cpu")

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=torch.float32, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=torch.float32).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)
input_ids = tokenizer("Hi", return_tensors="pt")["input_ids"].to(device)

with torch.no_grad():
    hf_lm = hf_model.model.language_model

    print("="*80)
    print("HF PLE Computation")
    print("="*80)

    # Step 1: Discrete embeddings (raw lookup)
    hf_discrete = hf_lm.embed_tokens_per_layer(input_ids)
    print(f"\n1. Discrete lookup: shape={hf_discrete.shape}")
    print(f"   mean={hf_discrete.mean():.6f}, std={hf_discrete.std():.6f}")

    # Step 2: Main embeddings
    hf_embeds = hf_lm.embed_tokens(input_ids)
    print(f"\n2. Main embeddings: shape={hf_embeds.shape}")
    print(f"   mean={hf_embeds.mean():.6f}, std={hf_embeds.std():.6f}")

    # Step 3: Continuous projection
    hf_continuous = hf_lm.per_layer_model_projection(hf_embeds)
    print(f"\n3. Continuous projection: shape={hf_continuous.shape}")
    print(f"   mean={hf_continuous.mean():.6f}, std={hf_continuous.std():.6f}")

    # Step 4: Scale continuous
    hf_scale = hf_lm.per_layer_projection_scale
    print(f"\n4. Projection scale: {hf_scale.item():.6f}")
    hf_continuous_scaled = hf_continuous * hf_scale
    print(f"   Scaled continuous: mean={hf_continuous_scaled.mean():.6f}, std={hf_continuous_scaled.std():.6f}")

    # Step 5: Add discrete + continuous (BEFORE reshape and norm)
    hf_combined_raw = hf_discrete + hf_continuous_scaled
    print(f"\n5. Combined (discrete + continuous_scaled):")
    print(f"   mean={hf_combined_raw.mean():.6f}, std={hf_combined_raw.std():.6f}")

    # Step 6: Reshape
    hf_combined_4d = hf_combined_raw.reshape(*input_ids.shape, config.num_layers, -1)
    print(f"\n6. Reshaped: shape={hf_combined_4d.shape}")

    # Step 7: Extract layer 0
    hf_ple0_unnormed = hf_combined_4d[:, :, 0, :]
    print(f"\n7. Layer 0 (unnormalized):")
    print(f"   mean={hf_ple0_unnormed.mean():.6f}, std={hf_ple0_unnormed.std():.6f}")

    # Step 8: Normalize
    hf_ple0_normed = hf_lm.per_layer_projection_norm(hf_ple0_unnormed)
    print(f"\n8. Normalized:")
    print(f"   mean={hf_ple0_normed.mean():.6f}, std={hf_ple0_normed.std():.6f}")

    # Step 9: Final scale
    hf_input_scale = hf_lm.per_layer_input_scale
    print(f"\n9. Input scale: {hf_input_scale.item():.6f}")
    hf_ple0_final = hf_ple0_normed * hf_input_scale
    print(f"   Final PLE: mean={hf_ple0_final.mean():.6f}, std={hf_ple0_final.std():.6f}")

    print("\n" + "="*80)
    print("FS2 PLE Computation")
    print("="*80)

    # FS2 computes PLE in frontend
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    _, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
    fs2_ple = state_bag.per_layer_inputs

    print(f"\nFS2 PLE shape: {fs2_ple.shape}")
    fs2_ple0 = fs2_ple[:, :, 0, :]
    print(f"FS2 Layer 0 PLE: mean={fs2_ple0.mean():.6f}, std={fs2_ple0.std():.6f}")

    print("\n" + "="*80)
    print("Manual FS2 PLE Computation (to debug)")
    print("="*80)

    # Step 1: Discrete (with scaling!)
    fs2_discrete = fs2_model.decoder_frontend.embed_tokens_per_layer(input_ids)
    print(f"\n1. Discrete lookup (raw): shape={fs2_discrete.shape}")
    print(f"   mean={fs2_discrete.mean():.6f}, std={fs2_discrete.std():.6f}")

    fs2_embed_scale = fs2_model.decoder_frontend.per_layer_embed_scale
    print(f"\n   Embed scale: {fs2_embed_scale.item():.6f}")
    fs2_discrete_scaled = fs2_discrete * fs2_embed_scale
    print(f"   Scaled discrete: mean={fs2_discrete_scaled.mean():.6f}, std={fs2_discrete_scaled.std():.6f}")

    fs2_discrete_4d = fs2_discrete_scaled.reshape(*input_ids.shape, config.num_layers, -1)
    print(f"   Reshaped: shape={fs2_discrete_4d.shape}")

    # Step 2: Main embeddings
    fs2_embeds = fs2_model.decoder_frontend.embed(input_ids) * fs2_model.decoder_frontend.scale
    print(f"\n2. Main embeddings: shape={fs2_embeds.shape}")
    print(f"   mean={fs2_embeds.mean():.6f}, std={fs2_embeds.std():.6f}")

    # Step 3: Continuous projection
    fs2_continuous = fs2_model.decoder_frontend.per_layer_model_projection(fs2_embeds)
    print(f"\n3. Continuous projection: shape={fs2_continuous.shape}")
    print(f"   mean={fs2_continuous.mean():.6f}, std={fs2_continuous.std():.6f}")

    # Step 4: Scale continuous
    fs2_proj_scale = fs2_model.decoder_frontend.per_layer_projection_scale
    print(f"\n4. Projection scale: {fs2_proj_scale.item():.6f}")
    fs2_continuous_scaled = fs2_continuous * fs2_proj_scale
    print(f"   Scaled continuous: mean={fs2_continuous_scaled.mean():.6f}, std={fs2_continuous_scaled.std():.6f}")

    # Step 5: Reshape continuous
    fs2_continuous_4d = fs2_continuous_scaled.reshape(*fs2_embeds.shape[:-1], config.num_layers, -1)
    print(f"\n5. Reshaped continuous: shape={fs2_continuous_4d.shape}")

    # Step 6: Normalize continuous (BEFORE adding!)
    fs2_continuous_normed = fs2_model.decoder_frontend.per_layer_projection_norm(fs2_continuous_4d)
    print(f"\n6. Normalized continuous:")
    print(f"   mean={fs2_continuous_normed.mean():.6f}, std={fs2_continuous_normed.std():.6f}")

    # Step 7: Add discrete + continuous
    fs2_combined = fs2_discrete_4d + fs2_continuous_normed
    print(f"\n7. Combined (discrete_scaled + continuous_normed):")
    print(f"   mean={fs2_combined.mean():.6f}, std={fs2_combined.std():.6f}")

    # Step 8: Final scale
    fs2_input_scale = fs2_model.decoder_frontend.per_layer_input_scale
    print(f"\n8. Input scale: {fs2_input_scale.item():.6f}")
    fs2_ple0_manual = fs2_combined[:, :, 0, :] * fs2_input_scale
    print(f"   Layer 0 final: mean={fs2_ple0_manual.mean():.6f}, std={fs2_ple0_manual.std():.6f}")

    print("\n" + "="*80)
    print("Comparison")
    print("="*80)

    diff = (hf_ple0_final - fs2_ple0).abs()
    print(f"\nHF vs FS2 (from state_bag): max={diff.max():.6e}, mean={diff.mean():.6e}")

    diff_manual = (hf_ple0_final - fs2_ple0_manual).abs()
    print(f"HF vs FS2 (manual):         max={diff_manual.max():.6e}, mean={diff_manual.mean():.6e}")

    # Key differences
    print("\n" + "="*80)
    print("Key Differences")
    print("="*80)
    print("1. Discrete embeddings:")
    print(f"   HF:  NO scaling (raw lookup)")
    print(f"   FS2: Scaled by sqrt(ple_hidden_dim) = {fs2_embed_scale.item():.6f}")
    print("\n2. Order of operations:")
    print(f"   HF:  discrete + (continuous * scale) → reshape → normalize → final_scale")
    print(f"   FS2: (discrete * embed_scale) + normalize(continuous * scale) → final_scale")
