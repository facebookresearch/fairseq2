#!/usr/bin/env python3
"""Discover all Gemma3n model variants from HuggingFace Hub.

Queries HF API to find all Gemma3n models and extracts their architectural configs.
Focus on text-only models for fairseq2 integration.
"""

from __future__ import annotations

from huggingface_hub import list_models, model_info
from transformers import AutoConfig


def main() -> None:
    print("=" * 80)
    print("GEMMA3N VARIANT DISCOVERY")
    print("=" * 80)
    print()

    # Search for all Gemma3n models
    print("Searching HuggingFace Hub for Gemma3n models...")
    models = list(list_models(search="gemma-3", author="google"))
    gemma3_models = [m for m in models if "gemma-3" in m.id.lower()]

    # Filter out quantized models (int4, int8, GGUF, etc.)
    quantized_keywords = ["int4", "int8", "gguf", "awq", "gptq", "bitsandbytes"]
    base_models = [
        m
        for m in gemma3_models
        if not any(kw in m.id.lower() for kw in quantized_keywords)
    ]

    print(f"Found {len(gemma3_models)} total models ({len(base_models)} non-quantized):")
    for m in base_models:
        print(f"  - {m.id}")
    print()

    # Extract configs for each model
    print("=" * 80)
    print("VARIANT CONFIGURATIONS")
    print("=" * 80)
    print()

    for model in base_models:
        print(f"\n{'─' * 80}")
        print(f"Model: {model.id}")
        print(f"{'─' * 80}")

        try:
            config = AutoConfig.from_pretrained(model.id, trust_remote_code=True)

            # Extract key architectural parameters
            params = {
                "model_type": getattr(config, "model_type", "N/A"),
                "hidden_size": getattr(config, "hidden_size", "N/A"),
                "num_hidden_layers": getattr(config, "num_hidden_layers", "N/A"),
                "num_attention_heads": getattr(config, "num_attention_heads", "N/A"),
                "num_key_value_heads": getattr(config, "num_key_value_heads", "N/A"),
                "head_dim": getattr(config, "head_dim", "N/A"),
                "intermediate_size": getattr(config, "intermediate_size", "N/A"),
                "altup_hidden_dim": getattr(config, "altup_hidden_dim", "N/A"),
                "sliding_window": getattr(config, "sliding_window", "N/A"),
                "rope_theta": getattr(config, "rope_theta", "N/A"),
                "rope_theta_global": getattr(
                    config, "rope_theta_global", "N/A"
                ),
                "num_kv_shared_layers": getattr(
                    config, "num_kv_shared_layers", "N/A"
                ),
                "laurel_rank": getattr(config, "laurel_rank", "N/A"),
                "altup_num_inputs": getattr(config, "altup_num_inputs", "N/A"),
                "vocab_size": getattr(config, "vocab_size", "N/A"),
                "vocab_size_per_layer_input": getattr(
                    config, "vocab_size_per_layer_input", "N/A"
                ),
                "hidden_size_per_layer_input": getattr(
                    config, "hidden_size_per_layer_input", "N/A"
                ),
                "max_position_embeddings": getattr(
                    config, "max_position_embeddings", "N/A"
                ),
                "final_logit_soft_cap": getattr(
                    config, "final_logit_soft_cap", "N/A"
                ),
                "activation_sparsity": getattr(
                    config, "activation_sparsity", "N/A"
                ),
            }

            # Print config
            for key, value in params.items():
                print(f"  {key:30s}: {value}")

            # Check if multimodal
            vision_config = getattr(config, "vision_config", None)
            audio_config = getattr(config, "audio_config", None)
            print(f"\n  Modalities:")
            print(f"    Text:   YES")
            print(f"    Vision: {'YES' if vision_config else 'NO'}")
            print(f"    Audio:  {'YES' if audio_config else 'NO'}")

        except Exception as e:
            print(f"  ERROR loading config: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal models found: {len(base_models)} non-quantized")

    # Categorize by modality
    text_only = []
    multimodal_vision = []
    multimodal_audio = []
    multimodal_full = []

    for m in base_models:
        try:
            config = AutoConfig.from_pretrained(m.id, trust_remote_code=True)
            vision = getattr(config, "vision_config", None)
            audio = getattr(config, "audio_config", None)

            if vision and audio:
                multimodal_full.append(m.id)
            elif vision:
                multimodal_vision.append(m.id)
            elif audio:
                multimodal_audio.append(m.id)
            else:
                text_only.append(m.id)
        except Exception:
            pass

    print("\n📝 Text-only models (integrate first):")
    for m_id in text_only:
        print(f"  - {m_id}")

    print("\n🖼️  Vision models (future integration):")
    for m_id in multimodal_vision:
        print(f"  - {m_id}")

    print("\n🔊 Audio models (future integration):")
    for m_id in multimodal_audio:
        print(f"  - {m_id}")

    print("\n🎬 Full multimodal models (future integration):")
    for m_id in multimodal_full:
        print(f"  - {m_id}")

    print("\nNext steps:")
    print("1. Create config presets for text-only variants")
    print("2. Verify which configs differ from E2B/E4B defaults")
    print("3. Check KV projection sharing defaults per variant")
    print("4. Create asset cards with checkpoint URLs")
    print("5. (Future) Add vision encoder support")
    print("6. (Future) Add audio encoder support")


if __name__ == "__main__":
    main()
