#!/usr/bin/env python3
"""Test Gemma3n tokenizer loading."""

from __future__ import annotations

from fairseq2.data.tokenizers import load_tokenizer


def test_e2b_tokenizer() -> None:
    print("=" * 80)
    print("TEST: E2B Tokenizer Loading")
    print("=" * 80)

    print("\n[fairseq2] Loading tokenizer...")
    tokenizer = load_tokenizer("gemma3n_e2b")

    print(f"  Tokenizer type: {type(tokenizer)}")
    print(f"  Vocab size: {tokenizer.vocab_info.size}")

    # Test text
    test_text = "The capital of France is"

    print(f"\n[fairseq2] Encoding: '{test_text}'")
    encoder = tokenizer.create_encoder()
    encoded = encoder(test_text)
    print(f"  Encoded type: {type(encoded)}")
    print(f"  Token IDs: {encoded}")

    # Decode
    decoder = tokenizer.create_decoder()
    decoded = decoder(encoded)
    print(f"  Decoded: '{decoded}'")

    print("\n✅ E2B tokenizer loading successful!")


def test_e4b_tokenizer() -> None:
    print("\n" + "=" * 80)
    print("TEST: E4B Tokenizer Loading")
    print("=" * 80)

    print("\n[fairseq2] Loading tokenizer...")
    tokenizer = load_tokenizer("gemma3n_e4b")

    print(f"  Tokenizer type: {type(tokenizer)}")
    print(f"  Vocab size: {tokenizer.vocab_info.size}")

    # Test text
    test_text = "The capital of France is"

    print(f"\n[fairseq2] Encoding: '{test_text}'")
    encoder = tokenizer.create_encoder()
    encoded = encoder(test_text)
    print(f"  Token IDs: {encoded}")

    # Decode
    decoder = tokenizer.create_decoder()
    decoded = decoder(encoded)
    print(f"  Decoded: '{decoded}'")

    print("\n✅ E4B tokenizer loading successful!")


def test_instruct_tokenizers() -> None:
    print("\n" + "=" * 80)
    print("TEST: Instruct Tokenizer Loading")
    print("=" * 80)

    for variant in ["gemma3n_e2b_instruct", "gemma3n_e4b_instruct"]:
        print(f"\n[fairseq2] Loading {variant}...")
        tokenizer = load_tokenizer(variant)
        print(f"  ✓ {variant}: {tokenizer.vocab_info.size} vocab")


def main() -> None:
    print("GEMMA3N TOKENIZER LOADING VERIFICATION")
    print()

    # Test E2B
    test_e2b_tokenizer()

    # Test E4B
    test_e4b_tokenizer()

    # Test instruct variants
    test_instruct_tokenizers()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
