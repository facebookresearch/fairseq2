#!/usr/bin/env python3
"""Debug forward pass step by step."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout


def main() -> None:
    print("="*80)
    print("FORWARD PASS DEBUG")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load HF model
    print("\n[1/4] Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3n-E2B-it",
        torch_dtype=torch.float32,
        device_map=device,
        local_files_only=True,
    )
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-3n-E2B-it",
        local_files_only=True,
    )

    # Create FS2 model
    print("\n[2/4] Creating fairseq2 model...")
    config = get_gemma3n_e2b_config()
    fs2_model = create_gemma3n_model(config, device=device, dtype=torch.float32)
    fs2_model.eval()

    # Load checkpoint
    print("\n[3/4] Loading checkpoint...")
    hf_state_dict = hf_model.state_dict()
    fs2_state_dict = convert_gemma3n_state_dict(hf_state_dict, config)
    fs2_model.load_state_dict(fs2_state_dict, strict=False)
    print("✓ Checkpoint loaded")

    # Test input
    print("\n[4/4] Testing forward pass...")
    test_text = "Hello"
    hf_inputs = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = hf_inputs["input_ids"]

    print(f"\nInput:")
    print(f"  Text: '{test_text}'")
    print(f"  Token IDs: {input_ids[0].tolist()}")
    print(f"  Shape: {input_ids.shape}")

    with torch.no_grad():
        # HF forward
        print("\n[HF Forward]")
        hf_outputs = hf_model(input_ids, use_cache=False)
        hf_logits = hf_outputs.logits
        hf_pred = hf_logits.argmax(dim=-1)

        print(f"  Logits shape: {hf_logits.shape}")
        print(f"  First token logits - min: {hf_logits[0, 0].min():.4f}, max: {hf_logits[0, 0].max():.4f}, mean: {hf_logits[0, 0].mean():.4f}")
        print(f"  Predictions: {hf_pred[0].tolist()}")
        print(f"  Unique predictions: {hf_pred.unique().tolist()}")

        # FS2 forward
        print("\n[FS2 Forward]")
        seq_lens = [input_ids.shape[1]]
        batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)

        # Add debug hooks
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    print(f"  {name}: output shape={output.shape}, mean={output.mean():.6f}, std={output.std():.6f}")
            return hook

        # Register hooks
        fs2_model.decoder_frontend.register_forward_hook(hook_fn("Frontend"))
        fs2_model.decoder.layers[0].register_forward_hook(hook_fn("Layer 0"))
        fs2_model.final_proj.register_forward_hook(hook_fn("Final Proj"))

        fs2_logits = fs2_model(input_ids, batch_layout)
        fs2_pred = fs2_logits.argmax(dim=-1)

        print(f"\n  Final logits shape: {fs2_logits.shape}")
        print(f"  First token logits - min: {fs2_logits[0, 0].min():.4f}, max: {fs2_logits[0, 0].max():.4f}, mean: {fs2_logits[0, 0].mean():.4f}")
        print(f"  Predictions: {fs2_pred[0].tolist()}")
        print(f"  Unique predictions: {fs2_pred.unique().tolist()}")

    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"\nPredictions match: {torch.equal(hf_pred, fs2_pred)}")
    print(f"Logits MSE: {((hf_logits - fs2_logits) ** 2).mean().item():.6e}")
    print(f"Logits max diff: {(hf_logits - fs2_logits).abs().max().item():.6e}")
    print("="*80)


if __name__ == "__main__":
    main()
