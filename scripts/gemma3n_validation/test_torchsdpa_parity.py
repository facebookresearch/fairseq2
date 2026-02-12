#!/usr/bin/env python3
# Test if TorchSDPA with scale=1.0 achieves parity (faster than NaiveSDPA)

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout


def main() -> None:
    print("Testing TorchSDPA vs NaiveSDPA for Gemma3n parity")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load HuggingFace reference
    print("[1/3] Loading HuggingFace model...")
    hf_model_id = "google/gemma-3n-E2B-it"
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=torch.float32,
        device_map=device,
    )
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    print("✓ HF model loaded\n")

    # Test input
    print("[2/3] Preparing test input...")
    test_text = "The quick brown fox jumps over the lazy dog"
    hf_inputs = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = hf_inputs["input_ids"]
    print(f"Text: '{test_text}'")
    print(f"Tokens: {input_ids.shape}\n")

    # Get HF reference output
    print("[3/3] Running HuggingFace reference...")
    with torch.no_grad():
        hf_output = hf_model(input_ids, past_key_values=None)
        hf_logits = hf_output.logits
        hf_next_token = hf_logits[0, -1].argmax().item()
    print(f"✓ HF next token: {hf_next_token}\n")

    # Create fairseq2 model config
    config = get_gemma3n_e2b_config()

    # Convert checkpoint once
    print("Converting checkpoint...")
    hf_state_dict = hf_model.state_dict()
    fs2_state_dict = convert_gemma3n_state_dict(hf_state_dict, config)
    print("✓ Checkpoint converted\n")

    # Test both SDPA backends
    sdpa_types = ["TorchSDPA", "NaiveSDPA"]
    results = {}

    for sdpa_type in sdpa_types:
        print(f"\n{'='*80}")
        print(f"Testing with {sdpa_type}")
        print(f"{'='*80}\n")

        # Temporarily patch factory to use specific SDPA
        import fairseq2.models.gemma3n.factory as factory_module
        from fairseq2.models.transformer import CausalAttentionBias

        if sdpa_type == "TorchSDPA":
            from fairseq2.models.transformer.sdpa.torch import TorchSDPA
            original_create = factory_module.create_gemma3n_decoder_layer

            def patched_create(layer_idx, config, *, device=None, dtype=None):
                layer = original_create(layer_idx, config, device=device, dtype=dtype)
                # Replace NaiveSDPA with TorchSDPA in self_attn
                is_global = factory_module.is_global_layer(layer_idx, config.num_layers)
                if is_global:
                    attention_bias = CausalAttentionBias()
                else:
                    attention_bias = CausalAttentionBias(attn_window_len=config.sliding_window)

                layer.self_attn.sdpa = TorchSDPA(
                    bias=attention_bias,
                    dropout_p=0.0,
                    scale=1.0,
                )
                return layer

            factory_module.create_gemma3n_decoder_layer = patched_create

        # Create model
        fs2_model = create_gemma3n_model(config, device=device, dtype=torch.float32)
        fs2_model.eval()
        fs2_model.load_state_dict(fs2_state_dict, strict=False)

        # Restore original if patched
        if sdpa_type == "TorchSDPA":
            factory_module.create_gemma3n_decoder_layer = original_create

        # Run inference
        with torch.no_grad():
            seq_lens = [input_ids.shape[1]]
            batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
            fs2_logits = fs2_model(input_ids, batch_layout)
            fs2_next_token = fs2_logits[0, -1].argmax().item()

        # Compare
        max_abs_diff = (hf_logits - fs2_logits).abs().max().item()
        max_rel_diff = ((hf_logits - fs2_logits).abs() / (hf_logits.abs() + 1e-8)).max().item()
        token_match = hf_next_token == fs2_next_token

        results[sdpa_type] = {
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "token_match": token_match,
            "next_token": fs2_next_token,
        }

        print(f"Next token: {fs2_next_token}")
        print(f"Token match: {'✓' if token_match else '✗'}")
        print(f"Max abs diff: {max_abs_diff:.6e}")
        print(f"Max rel diff: {max_rel_diff:.6e}")

        del fs2_model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    print(f"HuggingFace next token: {hf_next_token}\n")

    for sdpa_type in sdpa_types:
        r = results[sdpa_type]
        status = "✓ PASS" if r["token_match"] else "✗ FAIL"
        print(f"{sdpa_type:12s} {status}")
        print(f"  Next token:   {r['next_token']}")
        print(f"  Max abs diff: {r['max_abs_diff']:.6e}")
        print(f"  Max rel diff: {r['max_rel_diff']:.6e}")
        print()

    # Recommendation
    print(f"{'='*80}")
    if results["TorchSDPA"]["token_match"]:
        print("✓ TorchSDPA achieves parity - RECOMMENDED (faster)")
    else:
        print("✗ TorchSDPA does NOT achieve parity - stick with NaiveSDPA")
        print(f"  Diff: TorchSDPA={results['TorchSDPA']['max_abs_diff']:.6e} vs "
              f"NaiveSDPA={results['NaiveSDPA']['max_abs_diff']:.6e}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
