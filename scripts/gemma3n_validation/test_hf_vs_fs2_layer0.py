#!/usr/bin/env python3
"""Properly compare HF vs FS2 layer 0 output."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.models.transformer import AttentionBiasCache

device = torch.device("cpu")
dtype = torch.float32  # Force fp32 precision

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)
input_ids = tokenizer("Hi", return_tensors="pt")["input_ids"].to(device)

with torch.no_grad():
    # HF setup
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)
    per_layer_inputs_discrete = hf_lm.get_per_layer_inputs(input_ids)
    per_layer_inputs = hf_lm.project_per_layer_inputs(hf_embeds, per_layer_inputs_discrete)

    # 4D stack
    target_magnitude = torch.mean(hf_embeds**2, dim=-1, keepdim=True) ** 0.5
    epsilon = torch.tensor(1e-5, device=device, dtype=torch.float32)
    temp_hidden = [hf_embeds]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hf_embeds)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden.append(altup_proj)
    hf_hidden_4d = torch.stack(temp_hidden, dim=0)

    # FS2 setup (FRESH state_bag, crucial!)
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    seqs, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
    fs2_hidden_4d = fs2_model.decoder._stack_altup(seqs)
    fs2_per_layer_inputs = getattr(state_bag, 'per_layer_inputs', None)

    print("="*80)
    print("Input check (only showing if diff > 1e-7)")
    print("="*80)

    diff_4d = (hf_hidden_4d - fs2_hidden_4d).abs()
    if diff_4d.max() > 1e-7:
        print(f"❌ 4D inputs: max={diff_4d.max():.6e}")
    else:
        print(f"✓ 4D inputs match")

    diff_ple = (per_layer_inputs[:, :, 0, :] - fs2_per_layer_inputs[:, :, 0, :]).abs()
    if diff_ple.max() > 1e-7:
        print(f"❌ PLE inputs: max={diff_ple.max():.6e}")
    else:
        print(f"✓ PLE inputs match")

    print("\n" + "="*80)
    print("Layer 0 forward pass")
    print("="*80)

    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
    cos, sin = rope(hf_hidden_4d[0], position_ids, "sliding_attention")

    hf_layer = hf_lm.layers[0]
    hf_output = hf_layer(
        hidden_states=hf_hidden_4d,
        per_layer_input=per_layer_inputs[:, :, 0, :],
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
        past_key_values=None,
        cache_position=None,
    )

    attn_bias_cache = AttentionBiasCache()
    fs2_layer = fs2_model.decoder.layers[0]
    fs2_output = fs2_layer(
        fs2_hidden_4d, batch_layout, attn_bias_cache,
        per_layer_input=fs2_per_layer_inputs[:, :, 0, :],
        state_bag=state_bag
    )

    diff = (hf_output - fs2_output).abs()
    print(f"Output diff: max={diff.max():.6e}, mean={diff.mean():.6e}")

    if diff.max() < 1e-3:
        print("\n✓ Excellent parity!")
    elif diff.max() < 0.1:
        print("\n✓ Good parity")
    else:
        print(f"\n⚠️  Poor parity: max diff {diff.max():.2f}")
