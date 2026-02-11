#!/usr/bin/env python3
"""Compare first layer output between HF and FS2."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(f"Input: '{test_text}'")
    print(f"Token IDs: {input_ids[0].tolist()}\n")

    with torch.no_grad():
        # HF: Full forward through model to layer 0
        hf_outputs = hf_model(input_ids, use_cache=False, output_hidden_states=True)
        hf_layer0_out = hf_outputs.hidden_states[1]  # hidden_states[0] is embeddings, [1] is after layer 0

        print("HuggingFace layer 0 output:")
        print(f"  Shape: {hf_layer0_out.shape}")
        print(f"  Mean: {hf_layer0_out.mean():.6f}")
        print(f"  Std: {hf_layer0_out.std():.6f}")

        # FS2: Process through frontend and first layer
        seq_lens = [input_ids.shape[1]]
        batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
        state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))

        # Frontend
        seqs, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)

        # Stack to 4D
        hidden_states_4d = fs2_model.decoder._stack_altup(seqs)

        # Get PLE for layer 0
        per_layer_inputs = state_bag.per_layer_inputs
        layer0_ple = per_layer_inputs[:, :, 0, :]

        # Process through layer 0
        from fairseq2.models.transformer import AttentionBiasCache
        attn_bias_cache = AttentionBiasCache()

        layer0_out_4d = fs2_model.decoder.layers[0](
            hidden_states_4d,
            batch_layout,
            attn_bias_cache,
            per_layer_input=layer0_ple,
            state_bag=state_bag,
        )

        # Unstack to 3D for comparison
        layer0_out_3d = fs2_model.decoder._unstack_altup(layer0_out_4d)

        print("\nfairseq2 layer 0 output:")
        print(f"  4D shape: {layer0_out_4d.shape}")
        print(f"  4D mean: {layer0_out_4d.mean():.6f}")
        print(f"  3D shape: {layer0_out_3d.shape}")
        print(f"  3D mean: {layer0_out_3d.mean():.6f}")
        print(f"  3D std: {layer0_out_3d.std():.6f}")

        # Compare 4D outputs directly
        print("\n" + "="*80)
        print("Comparison (4D outputs)")
        print("="*80)

        print(f"\nHF 4D output:  {hf_layer0_out.shape}, mean={hf_layer0_out.mean():.6f}, std={hf_layer0_out.std():.6f}")
        print(f"FS2 4D output: {layer0_out_4d.shape}, mean={layer0_out_4d.mean():.6f}, std={layer0_out_4d.std():.6f}")

        diff = (hf_layer0_out - layer0_out_4d).abs()
        print(f"\nDifference:")
        print(f"  Max: {diff.max():.6e}")
        print(f"  Mean: {diff.mean():.6e}")
        print(f"  Match (atol=1e-3): {torch.allclose(hf_layer0_out, layer0_out_4d, atol=1e-3)}")

        # Check each version separately
        print(f"\nPer-version comparison:")
        for i in range(4):
            diff_i = (hf_layer0_out[i] - layer0_out_4d[i]).abs()
            print(f"  Version {i}: max_diff={diff_i.max():.6e}, mean_diff={diff_i.mean():.6e}")


if __name__ == "__main__":
    main()
