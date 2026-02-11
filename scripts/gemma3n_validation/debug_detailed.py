#!/usr/bin/env python3
"""Detailed forward pass debugging with intermediate outputs."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load models
    print("[1/3] Loading models...")
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

    config = get_gemma3n_e2b_config()
    fs2_model = create_gemma3n_model(config, device=device, dtype=torch.float32)
    fs2_model.eval()

    # Load checkpoint
    print("[2/3] Loading checkpoint...")
    hf_state_dict = hf_model.state_dict()
    fs2_state_dict = convert_gemma3n_state_dict(hf_state_dict, config)
    fs2_model.load_state_dict(fs2_state_dict, strict=False)
    print("✓ Checkpoint loaded\n")

    # Simple input
    print("[3/3] Testing forward pass...")
    test_text = "Hello"
    hf_inputs = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = hf_inputs["input_ids"]

    print(f"Input: '{test_text}'")
    print(f"Token IDs: {input_ids[0].tolist()}")
    print(f"Shape: {input_ids.shape}\n")

    with torch.no_grad():
        # HF forward with intermediate checks
        print("="*80)
        print("HuggingFace Forward Pass")
        print("="*80)

        hf_lm = hf_model.model.language_model

        # Embeddings
        hf_embeds = hf_lm.embed_tokens(input_ids)
        print(f"\n1. Embeddings: shape={hf_embeds.shape}")
        print(f"   mean={hf_embeds.mean():.6f}, std={hf_embeds.std():.6f}")

        # After first layer
        hf_layer0_out = hf_lm.layers[0](hf_embeds)[0]
        print(f"\n2. After Layer 0: shape={hf_layer0_out.shape}")
        print(f"   mean={hf_layer0_out.mean():.6f}, std={hf_layer0_out.std():.6f}")

        # Full forward
        hf_outputs = hf_model(input_ids, use_cache=False)
        hf_logits = hf_outputs.logits
        print(f"\n3. Final Logits: shape={hf_logits.shape}")
        print(f"   mean={hf_logits.mean():.6f}, std={hf_logits.std():.6f}")
        print(f"   argmax: {hf_logits.argmax(dim=-1)[0].tolist()}")

        # FS2 forward with intermediate checks
        print("\n" + "="*80)
        print("fairseq2 Forward Pass")
        print("="*80)

        seq_lens = [input_ids.shape[1]]
        batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)

        # Create state_bag explicitly to track PLE
        from fairseq2.nn import IncrementalStateBag
        state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))

        # Frontend
        fs2_embeds, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
        print(f"\n1. Embeddings: shape={fs2_embeds.shape}")
        print(f"   mean={fs2_embeds.mean():.6f}, std={fs2_embeds.std():.6f}")

        # Check PLE
        if hasattr(state_bag, 'per_layer_inputs'):
            ple = state_bag.per_layer_inputs
            print(f"\n   PLE stored: shape={ple.shape}")
            print(f"   PLE mean={ple.mean():.6f}, std={ple.std():.6f}")
        else:
            print("\n   ⚠️  PLE NOT stored in state_bag!")

        # Decoder
        fs2_decoder_out = fs2_model.decoder(fs2_embeds, batch_layout, state_bag=state_bag)
        print(f"\n2. After Decoder: shape={fs2_decoder_out.shape}")
        print(f"   mean={fs2_decoder_out.mean():.6f}, std={fs2_decoder_out.std():.6f}")

        # Full forward
        fs2_logits = fs2_model(input_ids, batch_layout)
        print(f"\n3. Final Logits: shape={fs2_logits.shape}")
        print(f"   mean={fs2_logits.mean():.6f}, std={fs2_logits.std():.6f}")
        print(f"   argmax: {fs2_logits.argmax(dim=-1)[0].tolist()}")

        # Comparison
        print("\n" + "="*80)
        print("Comparison")
        print("="*80)

        embed_diff = (hf_embeds - fs2_embeds).abs()
        print(f"\nEmbedding diff: max={embed_diff.max():.6e}, mean={embed_diff.mean():.6e}")

        logit_diff = (hf_logits - fs2_logits).abs()
        print(f"Logits diff: max={logit_diff.max():.6e}, mean={logit_diff.mean():.6e}")

        print(f"\nPredictions match: {torch.equal(hf_logits.argmax(dim=-1), fs2_logits.argmax(dim=-1))}")


if __name__ == "__main__":
    main()
