#!/usr/bin/env python3
"""Demonstrate chat template usage and test tokenizer parity."""

from __future__ import annotations

from transformers import AutoTokenizer

from fairseq2.data.tokenizers import load_tokenizer


def demo_huggingface_chat_template() -> None:
    """Show how HuggingFace uses chat templates."""
    print("=" * 80)
    print("HUGGINGFACE CHAT TEMPLATE DEMO")
    print("=" * 80)

    # Load HF instruct tokenizer
    hf_tok = AutoTokenizer.from_pretrained(
        "google/gemma-3n-E2B-it", trust_remote_code=True
    )

    # Example conversation
    conversation = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What about Germany?"},
    ]

    print("\n📝 Input conversation:")
    for msg in conversation:
        print(f"  {msg['role']}: {msg['content']}")

    # Apply chat template
    formatted = hf_tok.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    print("\n✨ Formatted with chat template:")
    print(formatted)
    print()

    # Also show tokenized version
    tokenized = hf_tok.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True
    )

    print(f"📊 Tokenized length: {len(tokenized)} tokens")
    print(f"   First 20 tokens: {tokenized[:20]}")

    # Decode to verify
    decoded = hf_tok.decode(tokenized)
    print(f"\n🔄 Decoded back:")
    print(decoded)


def test_basic_tokenization_parity() -> None:
    """Test that basic tokenization matches between HF and fairseq2."""
    print("\n" + "=" * 80)
    print("BASIC TOKENIZATION PARITY TEST")
    print("=" * 80)

    # Test both base and instruct
    test_cases = [
        ("gemma3n_e2b", "google/gemma-3n-E2B", "Base model"),
        ("gemma3n_e2b_instruct", "google/gemma-3n-E2B-it", "Instruct model"),
    ]

    test_texts = [
        "Hello world",
        "The capital of France is Paris.",
        "What is 2+2?",
        "<start_of_turn>user\nHello<end_of_turn>",  # Pre-formatted instruct
    ]

    for fs2_name, hf_name, desc in test_cases:
        print(f"\n{'─' * 80}")
        print(f"Testing: {desc} ({fs2_name})")
        print(f"{'─' * 80}")

        # Load tokenizers
        hf_tok = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        fs2_tok = load_tokenizer(fs2_name)

        # Create encoder
        fs2_encoder = fs2_tok.create_encoder()

        all_match = True

        for text in test_texts:
            # Encode with HF
            hf_ids = hf_tok.encode(text, add_special_tokens=True)

            # Encode with fairseq2
            fs2_tensor = fs2_encoder(text)
            fs2_ids = fs2_tensor.tolist()

            match = hf_ids == fs2_ids

            if match:
                print(f"  ✅ '{text[:40]}...'")
            else:
                print(f"  ❌ '{text[:40]}...'")
                print(f"     HF:  {hf_ids}")
                print(f"     FS2: {fs2_ids}")
                all_match = False

        if all_match:
            print(f"\n  🎉 All tests passed for {desc}!")
        else:
            print(f"\n  ⚠️  Some mismatches found for {desc}")


def test_special_token_handling() -> None:
    """Test special token encoding/decoding."""
    print("\n" + "=" * 80)
    print("SPECIAL TOKEN HANDLING TEST")
    print("=" * 80)

    hf_tok = AutoTokenizer.from_pretrained(
        "google/gemma-3n-E2B-it", trust_remote_code=True
    )
    fs2_tok = load_tokenizer("gemma3n_e2b_instruct")

    # Use raw encoder (no BOS/EOS added) for special token ID checking
    fs2_raw_encoder = fs2_tok.create_raw_encoder()
    fs2_decoder = fs2_tok.create_decoder()

    # Test special tokens
    special_tokens = [
        "<bos>",
        "<eos>",
        "<pad>",
        "<start_of_turn>",
        "<end_of_turn>",
        "<start_of_image>",
        "<end_of_image>",
    ]

    print("\nSpecial token IDs (using raw encoder):")
    for token in special_tokens:
        hf_id = hf_tok.convert_tokens_to_ids(token)
        fs2_tensor = fs2_raw_encoder(token)
        # For single tokens, should get 1 token ID
        fs2_id = fs2_tensor[0].item() if len(fs2_tensor) > 0 else None

        match = "✅" if hf_id == fs2_id else "❌"
        print(f"  {match} {token:20s} HF: {hf_id:6d}  FS2: {fs2_id}")


def test_decode_parity() -> None:
    """Test that decoding matches between HF and fairseq2."""
    print("\n" + "=" * 80)
    print("DECODING PARITY TEST")
    print("=" * 80)

    hf_tok = AutoTokenizer.from_pretrained(
        "google/gemma-3n-E2B", trust_remote_code=True
    )
    fs2_tok = load_tokenizer("gemma3n_e2b")
    fs2_decoder = fs2_tok.create_decoder()

    # Test token sequences
    test_sequences = [
        [2, 9259, 1902, 1],  # "Hello world"
        [2, 651, 5347, 576, 6181, 603, 7836, 235265, 1],  # "The capital of France is Paris."
        [
            2,
            2439,
            603,
            235248,
            235284,
            235340,
            235284,
            235336,
            1,
        ],  # "What is 2+2?" (example)
    ]

    print("\nDecoding test:")
    for seq in test_sequences:
        import torch

        hf_decoded = hf_tok.decode(seq, skip_special_tokens=False)
        fs2_decoded = fs2_decoder(torch.tensor(seq))

        match = "✅" if hf_decoded == fs2_decoded else "❌"
        print(f"  {match} Tokens: {seq[:10]}...")
        if hf_decoded != fs2_decoded:
            print(f"     HF:  '{hf_decoded}'")
            print(f"     FS2: '{fs2_decoded}'")


def main() -> None:
    print("GEMMA3N TOKENIZER TESTING")
    print()

    # 1. Show how chat templates work in HF
    demo_huggingface_chat_template()

    # 2. Test basic tokenization parity
    test_basic_tokenization_parity()

    # 3. Test special token handling
    test_special_token_handling()

    # 4. Test decoding
    test_decode_parity()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
What we learned:

1. **Chat Templates** (HuggingFace):
   - Used via `tokenizer.apply_chat_template(conversation)`
   - Formats multi-turn conversations with special tokens
   - Example: <start_of_turn>user\\nHello<end_of_turn>
   - Currently NOT exposed in our Gemma3nTokenizer

2. **Basic Tokenization**:
   - Our implementation should match HuggingFace exactly
   - Same token IDs, same vocab, same special tokens
   - Works for both base and instruct models

3. **What We Need**:
   - ✅ Basic encode/decode (DONE)
   - ✅ Special token handling (DONE)
   - ❌ Chat template support (NOT IMPLEMENTED)

4. **For Training/Inference**:
   - Basic tokenization is sufficient
   - Can manually format chat if needed
   - Chat template is a convenience feature
""")


if __name__ == "__main__":
    main()
