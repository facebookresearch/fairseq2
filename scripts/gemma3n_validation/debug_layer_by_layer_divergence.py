#!/usr/bin/env python3
"""Trace divergence layer-by-layer through all 30 layers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device
).eval()

# Disable HF activation sparsity
for layer in hf_model.model.language_model.layers:
    if hasattr(layer, 'mlp'):
        layer.mlp.activation_sparsity = 0.0

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
text = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("LAYER-BY-LAYER DIVERGENCE TRACKING")
print("="*80)

text_config = hf_model.model.language_model.config

with torch.no_grad():
    # HF setup
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)
    per_layer_inputs = hf_lm.get_per_layer_inputs(input_ids)
    per_layer_inputs = hf_lm.project_per_layer_inputs(hf_embeds, per_layer_inputs)

    # Build 4D stack
    hidden_states_0 = hf_embeds
    target_magnitude = torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
    epsilon_tensor = torch.tensor(1e-5, device=device, dtype=dtype)

    temp_hidden_states = [hidden_states_0]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hidden_states_0)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden_states.append(altup_proj)

    hf_hidden_4d = torch.stack(temp_hidden_states, dim=0)

    # Compute position embeddings
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    position_embeddings = {}
    for layer_type in set(text_config.layer_types):
        position_embeddings[layer_type] = hf_lm.rotary_emb(hf_hidden_4d, position_ids, layer_type)

    # FS2 setup - capture hidden states after each layer
    captured_fs2_states = []

    # Hook each layer to capture its output
    def make_hook(layer_idx):
        original_forward = fs2_model.decoder.layers[layer_idx].forward
        def hooked_forward(seqs, seqs_layout, attn_bias_cache, *, per_layer_input=None, state_bag=None):
            result = original_forward(seqs, seqs_layout, attn_bias_cache, per_layer_input=per_layer_input, state_bag=state_bag)
            captured_fs2_states.append(result.clone())
            return result
        return original_forward, hooked_forward

    original_forwards = []
    for i in range(len(fs2_model.decoder.layers)):
        orig, hooked = make_hook(i)
        original_forwards.append(orig)
        fs2_model.decoder.layers[i].forward = hooked

    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    fs2_logits = fs2_model(input_ids, batch_layout)

    # Restore original forwards
    for i, original_forward in enumerate(original_forwards):
        fs2_model.decoder.layers[i].forward = original_forward

    # Forward through HF layers one by one, comparing after each
    print(f"\nDivergence after each layer:")
    print(f"{'Layer':<6} {'Max Diff [0]':<15} {'Max Diff [1]':<15} {'Max Diff [2]':<15} {'Max Diff [3]':<15}")
    print("-" * 80)

    for layer_idx, decoder_layer in enumerate(hf_lm.layers):
        per_layer_input = per_layer_inputs[:, :, decoder_layer.layer_idx, :]
        hf_hidden_4d = decoder_layer(
            hf_hidden_4d,
            position_embeddings[decoder_layer.attention_type],
            per_layer_input,
        )

        # Compare with FS2 at this layer
        if layer_idx < len(captured_fs2_states):
            fs2_hidden = captured_fs2_states[layer_idx]

            diffs = []
            for i in range(4):
                diff = (hf_hidden_4d[i] - fs2_hidden[i]).abs().max().item()
                diffs.append(f"{diff:.6e}")

            print(f"{layer_idx:<6} {diffs[0]:<15} {diffs[1]:<15} {diffs[2]:<15} {diffs[3]:<15}")

print("="*80)
