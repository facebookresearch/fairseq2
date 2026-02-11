#!/usr/bin/env python3
"""Compare 4D stacking and PLE between HF and FS2."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag


def main() -> None:
    device = torch.device("cpu")

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

    # Test input
    test_text = "Hi"
    hf_inputs = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = hf_inputs["input_ids"]
    print(f"Input: '{test_text}'")
    print(f"Token IDs: {input_ids[0].tolist()}\n")

    with torch.no_grad():
        print("="*80)
        print("1. Embeddings")
        print("="*80)

        hf_lm = hf_model.model.language_model
        hf_embeds = hf_lm.embed_tokens(input_ids)
        fs2_embeds = fs2_model.decoder_frontend.embed(input_ids) * fs2_model.decoder_frontend.scale

        print(f"HF embeddings:  mean={hf_embeds.mean():.6f}, std={hf_embeds.std():.6f}")
        print(f"FS2 embeddings: mean={fs2_embeds.mean():.6f}, std={fs2_embeds.std():.6f}")
        print(f"Match: {torch.allclose(hf_embeds, fs2_embeds, atol=1e-5)}")
        print(f"Max diff: {(hf_embeds - fs2_embeds).abs().max():.6e}")

        print("\n" + "="*80)
        print("2. 4D Stacking")
        print("="*80)

        # HF 4D stacking
        hidden_states_0 = hf_embeds
        target_magnitude = torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
        epsilon_tensor = torch.tensor(1e-5, device=device, dtype=torch.float32)

        temp_hidden_states = [hidden_states_0]
        for i in range(1, hf_lm.config.altup_num_inputs):
            altup_proj = hf_lm.altup_projections[i - 1](hidden_states_0)
            current_hidden_state = altup_proj.to(dtype=hidden_states_0.dtype, device=target_magnitude.device)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)

        hf_4d = torch.stack(temp_hidden_states, dim=0)

        # FS2 4D stacking
        fs2_4d = fs2_model.decoder._stack_altup(fs2_embeds)

        print(f"HF 4D:  shape={hf_4d.shape}, mean={hf_4d.mean():.6f}, std={hf_4d.std():.6f}")
        print(f"FS2 4D: shape={fs2_4d.shape}, mean={fs2_4d.mean():.6f}, std={fs2_4d.std():.6f}")
        print(f"Match: {torch.allclose(hf_4d, fs2_4d, atol=1e-5)}")
        print(f"Max diff: {(hf_4d - fs2_4d).abs().max():.6e}")

        print("\n" + "="*80)
        print("3. PLE Computation")
        print("="*80)

        # HF PLE
        hf_ple_discrete = hf_lm.embed_tokens_per_layer(input_ids)
        hf_ple_discrete = hf_ple_discrete.reshape(
            *input_ids.shape,
            hf_lm.config.num_hidden_layers,
            hf_lm.config.hidden_size_per_layer_input
        )

        hf_ple_continuous = hf_lm.per_layer_model_projection(hf_embeds)
        hf_ple_continuous = hf_ple_continuous * (hf_lm.config.hidden_size**-0.5)
        hf_ple_continuous = hf_ple_continuous.reshape(
            *hf_embeds.shape[:-1],
            hf_lm.config.num_hidden_layers,
            hf_lm.config.hidden_size_per_layer_input
        )
        hf_ple_continuous = hf_lm.per_layer_projection_norm(hf_ple_continuous)

        hf_ple = (hf_ple_discrete + hf_ple_continuous) / (2**0.5)

        # FS2 PLE
        batch_layout = BatchLayout(input_ids.shape, [input_ids.shape[1]], device=device)
        state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
        _, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
        fs2_ple = state_bag.per_layer_inputs

        print(f"HF PLE:  shape={hf_ple.shape}, mean={hf_ple.mean():.6f}, std={hf_ple.std():.6f}")
        print(f"FS2 PLE: shape={fs2_ple.shape}, mean={fs2_ple.mean():.6f}, std={fs2_ple.std():.6f}")
        print(f"Match: {torch.allclose(hf_ple, fs2_ple, atol=1e-5)}")
        print(f"Max diff: {(hf_ple - fs2_ple).abs().max():.6e}")

        # Check layer 0 PLE
        print(f"\nLayer 0 PLE:")
        print(f"  HF:  {hf_ple[0, 0, 0, :5].tolist()}")
        print(f"  FS2: {fs2_ple[0, 0, 0, :5].tolist()}")

        print("\n" + "="*80)


if __name__ == "__main__":
    main()
