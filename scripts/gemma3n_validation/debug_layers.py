#!/usr/bin/env python3
"""Debug layer-by-layer forward pass."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
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

    # Load checkpoint
    hf_state_dict = hf_model.state_dict()
    fs2_state_dict = convert_gemma3n_state_dict(hf_state_dict, config)
    fs2_model.load_state_dict(fs2_state_dict, strict=False)
    print("✓ Models loaded\n")

    # Simple input
    test_text = "Hi"
    hf_inputs = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = hf_inputs["input_ids"]
    print(f"Input: '{test_text}'")
    print(f"Token IDs: {input_ids[0].tolist()}\n")

    with torch.no_grad():
        # HF layer-by-layer
        print("="*80)
        print("HuggingFace Layer-by-Layer")
        print("="*80)

        hf_lm = hf_model.model.language_model

        # Embed
        x = hf_lm.embed_tokens(input_ids)
        print(f"\nAfter embedding: mean={x.mean():.6f}, std={x.std():.6f}, shape={x.shape}")

        # Stack to 4D (HF does this before layers)
        hidden_states_0 = x
        target_magnitude = torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
        epsilon_tensor = torch.tensor(1e-5, device=x.device, dtype=x.dtype)

        temp_hidden_states = [hidden_states_0]
        for i in range(1, hf_lm.config.altup_num_inputs):
            altup_proj = hf_lm.altup_projections[i - 1](hidden_states_0)
            current_hidden_state = altup_proj.to(dtype=hidden_states_0.dtype, device=target_magnitude.device)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)

        x_4d = torch.stack(temp_hidden_states, dim=0)  # [4, B, S, M]
        print(f"After 4D stack: mean={x_4d.mean():.6f}, std={x_4d.std():.6f}, shape={x_4d.shape}")

        # Get PLE inputs
        per_layer_inputs = hf_lm.embed_tokens_per_layer(input_ids)
        per_layer_inputs = per_layer_inputs.reshape(*input_ids.shape, hf_lm.config.num_hidden_layers, hf_lm.config.hidden_size_per_layer_input)
        per_layer_proj = hf_lm.per_layer_model_projection(x)
        per_layer_proj = per_layer_proj * (hf_lm.config.hidden_size**-0.5)
        per_layer_proj = per_layer_proj.reshape(*x.shape[:-1], hf_lm.config.num_hidden_layers, hf_lm.config.hidden_size_per_layer_input)
        per_layer_proj = hf_lm.per_layer_projection_norm(per_layer_proj)
        per_layer_inputs = (per_layer_inputs + per_layer_proj) / (2**0.5)
        print(f"PLE inputs: mean={per_layer_inputs.mean():.6f}, std={per_layer_inputs.std():.6f}, shape={per_layer_inputs.shape}")

        # First 3 layers
        for i in range(min(3, len(hf_lm.layers))):
            layer_ple = per_layer_inputs[:, :, i, :]
            x_4d = hf_lm.layers[i](x_4d, per_layer_input=layer_ple)[0]
            print(f"After layer {i}: mean={x_4d.mean():.6f}, std={x_4d.std():.6f}, shape={x_4d.shape}")

        # FS2 layer-by-layer
        print("\n" + "="*80)
        print("fairseq2 Layer-by-Layer")
        print("="*80)

        seq_lens = [input_ids.shape[1]]
        batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
        state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))

        # Frontend
        x, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
        print(f"\nAfter frontend: mean={x.mean():.6f}, std={x.std():.6f}")
        print(f"  Shape: {x.shape}")

        # Check if PLE stored
        if hasattr(state_bag, 'per_layer_inputs'):
            ple = state_bag.per_layer_inputs
            print(f"  PLE shape: {ple.shape}, mean={ple.mean():.6f}")
        else:
            print("  ⚠️ PLE NOT in state_bag!")

        # Check decoder's 4D stacking
        print(f"\nDecoder processing:")
        print(f"  Input to decoder: shape={x.shape}")

        # Stack to 4D
        x_4d = fs2_model.decoder._stack_altup(x)
        print(f"  After 4D stack: shape={x_4d.shape}, mean={x_4d.mean():.6f}")

        # Process through first few layers manually
        for i in range(min(3, len(fs2_model.decoder.layers))):
            layer = fs2_model.decoder.layers[i]

            # Get PLE for this layer
            per_layer_input = None
            if hasattr(state_bag, 'per_layer_inputs'):
                per_layer_input = state_bag.per_layer_inputs[:, :, i, :]
                print(f"  Layer {i} PLE: shape={per_layer_input.shape}, mean={per_layer_input.mean():.6f}")

            # Forward through layer
            x_4d = layer(x_4d, batch_layout, None, per_layer_input=per_layer_input, state_bag=state_bag)
            print(f"  After layer {i}: shape={x_4d.shape}, mean={x_4d.mean():.6f}")

        # Unstack from 4D
        x_3d = fs2_model.decoder._unstack_altup(x_4d)
        print(f"  After 4D unstack: shape={x_3d.shape}, mean={x_3d.mean():.6f}")

        print("\n" + "="*80)


if __name__ == "__main__":
    main()
