#!/usr/bin/env python3
"""Check what HF layer returns - full output without [0]."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config

device = torch.device("cpu")

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=torch.float32, device_map=device, local_files_only=True
).eval()

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)
input_ids = tokenizer("Hi", return_tensors="pt")["input_ids"].to(device)

with torch.no_grad():
    hf_lm = hf_model.model.language_model
    hf_hidden = hf_lm.embed_tokens(input_ids)

    # 4D stacking
    target_magnitude = torch.mean(hf_hidden**2, dim=-1, keepdim=True) ** 0.5
    epsilon = torch.tensor(1e-5, device=device, dtype=torch.float32)
    temp_hidden = [hf_hidden]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hf_hidden)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden.append(altup_proj)
    hf_hidden_4d = torch.stack(temp_hidden, dim=0)

    # PLE
    config = get_gemma3n_e2b_config()
    per_layer_embeds = hf_lm.embed_tokens_per_layer(input_ids)
    per_layer_model_projection = hf_lm.per_layer_model_projection(hf_hidden)
    per_layer_projection_scale = hf_lm.per_layer_projection_scale
    per_layer_input_raw = per_layer_embeds + per_layer_model_projection * per_layer_projection_scale
    per_layer_input_raw = per_layer_input_raw.reshape(*input_ids.shape, config.num_layers, -1)
    per_layer_input_scale = hf_lm.per_layer_input_scale
    ple_layer0 = per_layer_input_raw[:, :, 0, :]
    ple_layer0 = hf_lm.per_layer_projection_norm(ple_layer0)
    ple_layer0 = ple_layer0 * per_layer_input_scale

    # Call layer 0 and check return value
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
    layer_type = hf_lm.config.layer_types[0]
    cos, sin = rope(hf_hidden_4d[0], position_ids, layer_type)

    hf_layer = hf_lm.layers[0]

    # Call WITHOUT [0] to see full return
    full_output = hf_layer(
        hidden_states=hf_hidden_4d,
        per_layer_input=ple_layer0,
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
        past_key_values=None,
        cache_position=None,
    )

    print(f"HF layer return type: {type(full_output)}")
    if isinstance(full_output, tuple):
        print(f"Tuple length: {len(full_output)}")
        for i, elem in enumerate(full_output):
            if isinstance(elem, torch.Tensor):
                print(f"  [{i}] Tensor shape: {elem.shape}")
            else:
                print(f"  [{i}] Type: {type(elem)}")
    else:
        print(f"Shape: {full_output.shape}")
