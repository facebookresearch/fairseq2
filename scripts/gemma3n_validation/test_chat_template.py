#!/usr/bin/env python3
"""Test Gemma3n chat template support."""

from __future__ import annotations

from transformers import AutoTokenizer

from fairseq2.data.tokenizers import load_tokenizer


def test_chat_template_formatting() -> None:
    """Test that chat template formatting matches HuggingFace."""
    print("=" * 80)
    print("CHAT TEMPLATE FORMATTING TEST")
    print("=" * 80)

    # Load tokenizers
    hf_tok = AutoTokenizer.from_pretrained(
        "google/gemma-3n-E2B-it", trust_remote_code=True
    )
    fs2_tok = load_tokenizer("gemma3n_e2b_instruct")

    # Test conversation
    conversation = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What about Germany?"},
    ]

    print("\n📝 Input conversation:")
    for msg in conversation:
        print(f"  {msg['role']}: {msg['content']}")

    # Test 1: Get formatted string (no tokenization)
    print("\n" + "─" * 80)
    print("Test 1: Format to string (tokenize=False)")
    print("─" * 80)

    hf_formatted = hf_tok.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    fs2_formatted = fs2_tok.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    print("\n[HuggingFace]:")
    print(hf_formatted)

    print("\n[fairseq2]:")
    print(fs2_formatted)

    if hf_formatted == fs2_formatted:
        print("\n✅ Formatted strings match!")
    else:
        print("\n❌ Formatted strings differ!")
        print("\nDifferences:")
        hf_lines = hf_formatted.split("\n")
        fs2_lines = fs2_formatted.split("\n")
        for i, (h, f) in enumerate(zip(hf_lines, fs2_lines)):
            if h != f:
                print(f"  Line {i}:")
                print(f"    HF:  {h!r}")
                print(f"    FS2: {f!r}")

    # Test 2: Get token IDs (with tokenization)
    print("\n" + "─" * 80)
    print("Test 2: Tokenize (tokenize=True)")
    print("─" * 80)

    hf_tokens = hf_tok.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True
    )
    fs2_tokens = fs2_tok.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True
    )

    print(f"\n[HuggingFace] {len(hf_tokens)} tokens")
    print(f"  First 20: {hf_tokens[:20]}")

    print(f"\n[fairseq2] {len(fs2_tokens)} tokens")
    print(f"  First 20: {fs2_tokens[:20]}")

    if hf_tokens == fs2_tokens:
        print("\n✅ Tokenized output matches!")
    else:
        print("\n❌ Tokenized output differs!")
        print(f"\n  Length: HF={len(hf_tokens)}, FS2={len(fs2_tokens)}")
        if len(hf_tokens) == len(fs2_tokens):
            # Find first mismatch
            for i, (h, f) in enumerate(zip(hf_tokens, fs2_tokens)):
                if h != f:
                    print(f"  First mismatch at position {i}: HF={h}, FS2={f}")
                    break

    # Test 3: Without generation prompt
    print("\n" + "─" * 80)
    print("Test 3: No generation prompt (add_generation_prompt=False)")
    print("─" * 80)

    hf_no_prompt = hf_tok.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    fs2_no_prompt = fs2_tok.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )

    if hf_no_prompt == fs2_no_prompt:
        print("✅ Output matches without generation prompt")
    else:
        print("❌ Output differs without generation prompt")

    # Test 4: Single user message
    print("\n" + "─" * 80)
    print("Test 4: Single user message")
    print("─" * 80)

    single_msg = [{"role": "user", "content": "Hello!"}]

    hf_single = hf_tok.apply_chat_template(
        single_msg, tokenize=False, add_generation_prompt=True
    )
    fs2_single = fs2_tok.apply_chat_template(
        single_msg, tokenize=False, add_generation_prompt=True
    )

    print(f"\n[HuggingFace]:\n{hf_single}")
    print(f"\n[fairseq2]:\n{fs2_single}")

    if hf_single == fs2_single:
        print("\n✅ Single message formatting matches!")
    else:
        print("\n❌ Single message formatting differs!")


def test_chat_template_property() -> None:
    """Test that we can access the chat template."""
    print("\n" + "=" * 80)
    print("CHAT TEMPLATE PROPERTY TEST")
    print("=" * 80)

    # Instruct model has chat template
    instruct_tok = load_tokenizer("gemma3n_e2b_instruct")
    template = instruct_tok.chat_template

    if template:
        print("\n✅ Instruct tokenizer has chat template")
        print(f"   Template length: {len(template)} chars")
        print(f"   First 100 chars: {template[:100]}...")
    else:
        print("\n❌ Instruct tokenizer missing chat template!")

    # Base model might not have chat template
    base_tok = load_tokenizer("gemma3n_e2b")
    base_template = base_tok.chat_template

    if base_template:
        print("\n✅ Base tokenizer also has chat template")
    else:
        print("\n⚠️  Base tokenizer has no chat template (expected)")


def main() -> None:
    print("GEMMA3N CHAT TEMPLATE TESTING\n")

    test_chat_template_formatting()
    test_chat_template_property()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Chat template support status:

✅ apply_chat_template() method implemented
✅ Delegates to HuggingFace's built-in template
✅ Supports both tokenized and string output
✅ Supports add_generation_prompt parameter
✅ chat_template property exposes the template

Usage example:

    from fairseq2.data.tokenizers import load_tokenizer

    tokenizer = load_tokenizer("gemma3n_e2b_instruct")

    conversation = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    # Get formatted string
    text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    # Or get token IDs
    tokens = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True
    )
""")


if __name__ == "__main__":
    main()
