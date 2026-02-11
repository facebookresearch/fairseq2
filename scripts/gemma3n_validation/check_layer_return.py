#!/usr/bin/env python3
"""Check what HF layer forward returns."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from fairseq2.nn import IncrementalStateBag

device = torch.device("cpu")
dtype = torch.float32
model_name = "google/gemma-3n-E2B-it"

print("Loading model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
text = "The quick brown"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

print(f"Input: {text!r}\n")

with torch.no_grad():
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)

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

    # Position embeddings
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    text_config = hf_lm.config
    position_embeddings = {}
    for layer_type in set(text_config.layer_types):
        position_embeddings[layer_type] = hf_lm.rotary_emb(hf_hidden_4d, position_ids, layer_type)

    # Test layer 0 forward
    print("Testing layer 0 forward call...")
    hf_layer0 = hf_lm.layers[0]
    hf_cache = DynamicCache()

    # Get PLE
    from fairseq2.nn import BatchLayout
    from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
    from fairseq2.models.gemma3n.factory import create_gemma3n_model

    config = get_gemma3n_e2b_config()
    fs2_model = create_gemma3n_model(config, device=device, dtype=dtype)
    state_bag = IncrementalStateBag(input_ids.shape[1])
    fs2_embeds, _ = fs2_model.decoder_frontend(
        input_ids, BatchLayout(input_ids.shape, [input_ids.shape[1]], device=device), state_bag=state_bag
    )
    layer_ple = state_bag.per_layer_inputs[:, :, 0, :]

    layer_type = hf_layer0.attention_type

    result = hf_layer0(
        hf_hidden_4d,
        position_embeddings=position_embeddings[layer_type],
        per_layer_input=layer_ple,
        past_key_value=hf_cache,
    )

    print(f"\nLayer 0 forward returned:")
    print(f"  Type: {type(result)}")
    if isinstance(result, tuple):
        print(f"  Tuple length: {len(result)}")
        for i, item in enumerate(result):
            print(f"  Item {i}: type={type(item)}, ", end="")
            if isinstance(item, torch.Tensor):
                print(f"shape={item.shape}")
            else:
                print(f"value={item}")
