#!/usr/bin/env python3
"""Verify Gemma3n tokenizer differences between base and instruct."""

from __future__ import annotations

from transformers import AutoTokenizer


def compare_tokenizers() -> None:
    print("=" * 80)
    print("GEMMA3N TOKENIZER COMPARISON")
    print("=" * 80)

    variants = {
        "E2B Base": "google/gemma-3n-E2B",
        "E2B Instruct": "google/gemma-3n-E2B-it",
        "E4B Base": "google/gemma-3n-E4B",
        "E4B Instruct": "google/gemma-3n-E4B-it",
    }

    tokenizers = {}

    for name, model_id in variants.items():
        print(f"\n[{name}] Loading from {model_id}...")
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizers[name] = tok

        print(f"  Vocab size: {tok.vocab_size}")
        print(f"  BOS token: {tok.bos_token} (ID: {tok.bos_token_id})")
        print(f"  EOS token: {tok.eos_token} (ID: {tok.eos_token_id})")
        print(f"  PAD token: {tok.pad_token} (ID: {tok.pad_token_id})")
        print(f"  UNK token: {tok.unk_token} (ID: {tok.unk_token_id})")

        # Check for special tokens
        special_tokens = tok.all_special_tokens
        print(f"  All special tokens: {special_tokens}")

        # Check for instruct-specific tokens
        instruct_tokens = [
            "<start_of_turn>",
            "<end_of_turn>",
            "user",
            "model",
            "system",
        ]
        found_instruct = [t for t in instruct_tokens if t in tok.get_vocab()]
        if found_instruct:
            print(f"  Instruct-specific tokens found: {found_instruct}")

    # Compare base vs instruct
    print("\n" + "=" * 80)
    print("DIFFERENCES")
    print("=" * 80)

    for size in ["E2B", "E4B"]:
        base = tokenizers[f"{size} Base"]
        instruct = tokenizers[f"{size} Instruct"]

        print(f"\n{size}:")
        if base.vocab_size != instruct.vocab_size:
            print(f"  ❌ Vocab size differs: {base.vocab_size} vs {instruct.vocab_size}")
        else:
            print(f"  ✅ Vocab size same: {base.vocab_size}")

        if base.eos_token != instruct.eos_token:
            print(
                f"  ❌ EOS token differs: '{base.eos_token}' vs '{instruct.eos_token}'"
            )
        else:
            print(f"  ✅ EOS token same: '{base.eos_token}'")

        # Test encoding
        test_text = "Hello world"
        base_encoded = base.encode(test_text)
        instruct_encoded = instruct.encode(test_text)

        if base_encoded != instruct_encoded:
            print(f"  ❌ Encoding differs:")
            print(f"     Base:     {base_encoded}")
            print(f"     Instruct: {instruct_encoded}")
        else:
            print(f"  ✅ Encoding same: {base_encoded}")


if __name__ == "__main__":
    compare_tokenizers()
