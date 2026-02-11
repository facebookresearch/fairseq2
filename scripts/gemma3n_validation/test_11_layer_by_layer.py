"""Test 11: Layer-by-layer divergence tracking to find where parity breaks."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.models.transformer import AttentionBiasCache

device = torch.device("cpu")
dtype = torch.float32
model_name = "google/gemma-3n-E2B-it"

print("Loading models...")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
text = "The quick brown"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
print(f"Input: {text!r}")
print(f"Tokens: {input_ids.shape[1]}\n")

seq_lens = [input_ids.shape[1]]
batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
state_bag = IncrementalStateBag(input_ids.shape[1])

print("=" * 80)
print("TEST 11: Layer-by-Layer Divergence Tracking")
print("=" * 80)

with torch.no_grad():
    # Get embeddings
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)
    fs2_embeds, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)

    # Stack to 4D
    target_magnitude = torch.mean(hf_embeds**2, dim=-1, keepdim=True) ** 0.5
    epsilon = torch.tensor(1e-5, device=device, dtype=dtype)
    temp_hidden_states = [hf_embeds]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hf_embeds)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden_states.append(altup_proj)
    hf_hidden_4d = torch.stack(temp_hidden_states, dim=0)
    fs2_hidden_4d = fs2_model.decoder._stack_altup(fs2_embeds)

    # Setup for layer iteration
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    text_config = hf_model.model.language_model.config
    position_embeddings = {}
    for layer_type in set(text_config.layer_types):
        position_embeddings[layer_type] = hf_lm.rotary_emb(hf_hidden_4d, position_ids, layer_type)

    attn_bias_cache = AttentionBiasCache()
    per_layer_inputs = state_bag.per_layer_inputs

    print(f"\n{'Layer':<8} {'Type':<10} {'Max Diff':<15} {'Mean Diff':<15} {'Status':<10}")
    print("-" * 80)

    hf_hidden = hf_hidden_4d
    fs2_hidden = fs2_hidden_4d

    for layer_idx in range(min(15, config.num_layers)):  # Test first 15 layers
        hf_layer = hf_lm.layers[layer_idx]
        fs2_layer = fs2_model.decoder.layers[layer_idx]

        layer_type = hf_layer.attention_type
        layer_ple = per_layer_inputs[:, :, layer_idx, :]

        # HF forward
        hf_hidden = hf_layer(
            hf_hidden,
            position_embeddings=position_embeddings[layer_type],
            per_layer_input=layer_ple,
        )

        # FS2 forward
        fs2_hidden = fs2_layer(
            fs2_hidden,
            batch_layout,
            attn_bias_cache,
            per_layer_input=layer_ple,
            state_bag=state_bag,
        )

        # Compare active dimension
        diff = (hf_hidden[0] - fs2_hidden[0]).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        status = "✅" if max_diff < 1e-4 else "❌"
        print(f"{layer_idx:<8} {layer_type:<10} {max_diff:<15.6e} {mean_diff:<15.6e} {status:<10}")

    print("-" * 80)
    print("\nLegend:")
    print("  ✅ = Acceptable divergence (< 1e-4)")
    print("  ❌ = Significant divergence (>= 1e-4)")
    print("\nIf divergence grows consistently, error accumulates across layers")
    print("If divergence jumps at specific layer, that layer has an issue")
