#!/usr/bin/env python3
"""Trace intermediate outputs in layer 0 forward (HF vs FS2) using hooks."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag

device = torch.device("cpu")
dtype = torch.float32

# Load models
hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)
input_ids = tokenizer("Hi", return_tensors="pt")["input_ids"].to(device)

def compare(name, hf_tensor, fs2_tensor, threshold=1e-5):
    """Compare tensors and only print if diff exceeds threshold."""
    diff = (hf_tensor - fs2_tensor).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    if max_diff > threshold:
        print(f"❌ {name:30s} max={max_diff:.6e}, mean={mean_diff:.6e}")
        print(f"   HF:  mean={hf_tensor.mean():.6f}, std={hf_tensor.std():.6f}")
        print(f"   FS2: mean={fs2_tensor.mean():.6f}, std={fs2_tensor.std():.6f}")
        return False
    else:
        print(f"✓  {name:30s} max={max_diff:.6e}")
        return True

print("="*80)
print("LAYER 0 TRACE (only showing diffs > 1e-5)")
print("="*80)

# Storage for intermediate activations
hf_intermediates = {}
fs2_intermediates = {}

with torch.no_grad():
    # Setup HF
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

    # Setup FS2
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    seqs, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
    fs2_hidden_4d = fs2_model.decoder._stack_altup(seqs)
    fs2_per_layer_inputs = getattr(state_bag, 'per_layer_inputs', None)

    # Verify inputs match
    print("\n[INPUTS]")
    compare("4D hidden", hf_hidden_4d, fs2_hidden_4d, threshold=1e-7)
    compare("PLE", per_layer_inputs[:, :, 0, :], fs2_per_layer_inputs[:, :, 0, :], threshold=1e-7)

    # Register hooks for FS2 layer to capture intermediates
    fs2_layer = fs2_model.decoder.layers[0]

    def make_hook(name):
        def hook(module, input, output):
            fs2_intermediates[name] = output.clone() if isinstance(output, torch.Tensor) else output
        return hook

    # Register hooks on FS2 submodules
    handles = []
    handles.append(fs2_layer.altup_predict.register_forward_hook(make_hook('altup_predict')))
    handles.append(fs2_layer.altup_activate.register_forward_hook(make_hook('altup_activate')))
    handles.append(fs2_layer.pre_attention_norm.register_forward_hook(make_hook('pre_attn_norm')))
    handles.append(fs2_layer.self_attn.register_forward_hook(make_hook('attention')))
    handles.append(fs2_layer.post_attention_norm.register_forward_hook(make_hook('post_attn_norm')))
    handles.append(fs2_layer.ffn.register_forward_hook(make_hook('ffn')))

    # Run FS2 forward
    from fairseq2.models.transformer import AttentionBiasCache
    attn_bias_cache = AttentionBiasCache()
    fs2_output = fs2_layer(
        fs2_hidden_4d, batch_layout, attn_bias_cache,
        per_layer_input=fs2_per_layer_inputs[:, :, 0, :],
        state_bag=state_bag
    )

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Now manually compute HF intermediates
    print("\n[COMPUTING HF INTERMEDIATES]")
    hf_layer = hf_lm.layers[0]

    # Get HF RoPE
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding
    rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
    cos, sin = rope(hf_hidden_4d[0], position_ids, "sliding_attention")

    # Run full HF forward to get output
    hf_output = hf_layer(
        hidden_states=hf_hidden_4d,
        per_layer_input=per_layer_inputs[:, :, 0, :],
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
        past_key_values=None,
        cache_position=None,
    )

    # Compare final outputs first
    print("\n[FINAL OUTPUT]")
    compare("layer output", hf_output, fs2_output)

    # Now try to manually extract HF intermediates by calling submodules
    print("\n[SUBMODULE COMPARISONS]")
    hf_hidden = hf_hidden_4d
    hf_ple = per_layer_inputs[:, :, 0, :]

    # Try to trace through HF layer manually
    try:
        # Check if we can access HF submodules
        print("\nHF layer attributes:")
        for attr in dir(hf_layer):
            if not attr.startswith('_') and not attr.startswith('forward'):
                obj = getattr(hf_layer, attr)
                if isinstance(obj, torch.nn.Module):
                    print(f"  - {attr}: {type(obj).__name__}")
    except Exception as e:
        print(f"Error inspecting HF layer: {e}")

    print("\n" + "="*80)
    print("Note: Need to manually trace HF forward to compare intermediates")
    print("Run HF model in debug mode to capture intermediate values")
    print("="*80)
