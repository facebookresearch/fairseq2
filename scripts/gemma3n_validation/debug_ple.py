#!/usr/bin/env python3
"""Debug PLE computation step by step."""

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
        hf_lm = hf_model.model.language_model
        hf_embeds = hf_lm.embed_tokens(input_ids)
        fs2_frontend = fs2_model.decoder_frontend

        print("="*80)
        print("PLE Step 1: Discrete Embeddings")
        print("="*80)

        # HF discrete
        hf_discrete_raw = hf_lm.embed_tokens_per_layer(input_ids)
        hf_discrete = hf_discrete_raw.reshape(
            *input_ids.shape,
            hf_lm.config.num_hidden_layers,
            hf_lm.config.hidden_size_per_layer_input
        )

        # FS2 discrete (using the proper method that includes scaling)
        fs2_discrete = fs2_frontend._get_per_layer_inputs(input_ids)
        fs2_discrete_raw = fs2_discrete.reshape(*input_ids.shape, -1)

        print(f"HF discrete raw:  shape={hf_discrete_raw.shape}, mean={hf_discrete_raw.mean():.6f}")
        print(f"FS2 discrete raw: shape={fs2_discrete_raw.shape}, mean={fs2_discrete_raw.mean():.6f}")
        print(f"Match: {torch.allclose(hf_discrete_raw, fs2_discrete_raw, atol=1e-5)}")
        print(f"\nHF discrete:  shape={hf_discrete.shape}, mean={hf_discrete.mean():.6f}")
        print(f"FS2 discrete: shape={fs2_discrete.shape}, mean={fs2_discrete.mean():.6f}")
        print(f"Match: {torch.allclose(hf_discrete, fs2_discrete, atol=1e-5)}")

        print("\n" + "="*80)
        print("PLE Step 2: Continuous Projection")
        print("="*80)

        # HF continuous
        hf_cont_proj_raw = hf_lm.per_layer_model_projection(hf_embeds)
        print(f"HF continuous proj (raw): mean={hf_cont_proj_raw.mean():.6f}, std={hf_cont_proj_raw.std():.6f}")

        hf_cont_scaled = hf_cont_proj_raw * (hf_lm.config.hidden_size**-0.5)
        print(f"HF continuous scaled:     mean={hf_cont_scaled.mean():.6f}, std={hf_cont_scaled.std():.6f}")
        print(f"  Scale factor: {hf_lm.config.hidden_size**-0.5:.6f}")

        hf_cont_reshaped = hf_cont_scaled.reshape(
            *hf_embeds.shape[:-1],
            hf_lm.config.num_hidden_layers,
            hf_lm.config.hidden_size_per_layer_input
        )
        print(f"HF continuous reshaped:   mean={hf_cont_reshaped.mean():.6f}, std={hf_cont_reshaped.std():.6f}")

        hf_continuous = hf_lm.per_layer_projection_norm(hf_cont_reshaped)
        print(f"HF continuous normed:     mean={hf_continuous.mean():.6f}, std={hf_continuous.std():.6f}")

        # FS2 continuous (using proper method)
        fs2_embeds = fs2_frontend.embed(input_ids) * fs2_frontend.scale
        fs2_continuous = fs2_frontend._project_per_layer_inputs(fs2_embeds)
        print(f"\nFS2 continuous normed:     mean={fs2_continuous.mean():.6f}, std={fs2_continuous.std():.6f}")
        print(f"Match: {torch.allclose(hf_continuous, fs2_continuous, atol=1e-5)}")

        print("\n" + "="*80)
        print("PLE Step 3: Combination")
        print("="*80)

        # HF combine
        hf_ple = (hf_discrete + hf_continuous) / (2**0.5)
        print(f"HF PLE:  mean={hf_ple.mean():.6f}, std={hf_ple.std():.6f}")
        print(f"  Combine scale: {1/(2**0.5):.6f}")

        # FS2 combine (using proper method)
        fs2_ple = fs2_frontend._combine_per_layer_inputs(fs2_discrete, fs2_continuous)
        print(f"FS2 PLE: mean={fs2_ple.mean():.6f}, std={fs2_ple.std():.6f}")
        print(f"  Combine scale: {fs2_frontend.per_layer_input_scale.item():.6f}")

        print(f"\nMatch: {torch.allclose(hf_ple, fs2_ple, atol=1e-5)}")
        print(f"Max diff: {(hf_ple - fs2_ple).abs().max():.6e}")


if __name__ == "__main__":
    main()
