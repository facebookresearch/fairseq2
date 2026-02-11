#!/usr/bin/env python3
"""Add hooks to HF layer to capture what actually happens in forward()."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)
input_ids = tokenizer("Hi", return_tensors="pt")["input_ids"].to(device)

# Storage for intermediate activations
activations = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activations[name] = output.clone()
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            activations[name] = output[0].clone()
    return hook

print("="*80)
print("Capture HF Layer Forward with Hooks")
print("="*80)

with torch.no_grad():
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)

    # 4D stack
    target_magnitude = torch.mean(hf_embeds**2, dim=-1, keepdim=True) ** 0.5
    epsilon = torch.tensor(1e-5, device=device, dtype=dtype)
    temp_hidden = [hf_embeds]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hf_embeds)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden.append(altup_proj)
    hf_hidden_4d = torch.stack(temp_hidden, dim=0)

    per_layer_inputs_discrete = hf_lm.get_per_layer_inputs(input_ids)
    per_layer_inputs = hf_lm.project_per_layer_inputs(hf_embeds, per_layer_inputs_discrete)

    # Register hooks
    hf_layer = hf_lm.layers[0]
    hooks = []

    # Try to hook key submodules
    if hasattr(hf_layer, 'altup'):
        hooks.append(hf_layer.altup.register_forward_hook(make_hook('altup_output')))
    if hasattr(hf_layer, 'input_layernorm'):
        hooks.append(hf_layer.input_layernorm.register_forward_hook(make_hook('input_norm')))
    if hasattr(hf_layer, 'self_attn'):
        hooks.append(hf_layer.self_attn.register_forward_hook(make_hook('attention')))
    if hasattr(hf_layer, 'mlp'):
        hooks.append(hf_layer.mlp.register_forward_hook(make_hook('mlp')))

    # Run forward
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding
    rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
    cos, sin = rope(hf_hidden_4d[0], position_ids, "sliding_attention")

    hf_output = hf_layer(
        hidden_states=hf_hidden_4d,
        per_layer_input=per_layer_inputs[:, :, 0, :],
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
        past_key_values=None,
        cache_position=None,
    )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    print("\nCaptured activations:")
    for name, tensor in activations.items():
        print(f"{name:20s}: shape={tensor.shape}, mean={tensor.mean():.6f}, std={tensor.std():.6f}")

    print(f"\nFinal output:")
    print(f"  shape={hf_output.shape}, mean={hf_output.mean():.6f}, std={hf_output.std():.6f}")

    if hf_output.ndim == 4:
        print("\nPer-prediction stats:")
        for i in range(4):
            pred = hf_output[i]
            print(f"  Prediction {i}: mean={pred.mean():.6f}, std={pred.std():.6f}")
