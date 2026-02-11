#!/usr/bin/env python3
"""Check output shapes from HF vs FS2 layer."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.models.transformer import AttentionBiasCache

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)
input_ids = tokenizer("Hi", return_tensors="pt")["input_ids"].to(device)

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

    # HF layer forward
    hf_layer = hf_lm.layers[0]
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

    # FS2 layer forward
    attn_bias_cache = AttentionBiasCache()
    fs2_layer = fs2_model.decoder.layers[0]
    fs2_output = fs2_layer(
        fs2_hidden_4d, batch_layout, attn_bias_cache,
        per_layer_input=fs2_per_layer_inputs[:, :, 0, :],
        state_bag=state_bag
    )

    print("="*80)
    print("Output Shapes")
    print("="*80)
    print(f"HF output shape: {hf_output.shape}")
    print(f"HF output mean: {hf_output.mean():.6f}, std: {hf_output.std():.6f}")
    print(f"\nFS2 output shape: {fs2_output.shape}")
    print(f"FS2 output mean: {fs2_output.mean():.6f}, std: {fs2_output.std():.6f}")

    print("\n" + "="*80)
    print("Comparison")
    print("="*80)

    if hf_output.shape == fs2_output.shape:
        print("✓ Shapes match")
        diff = (hf_output - fs2_output).abs()
        print(f"Diff: max={diff.max():.6e}, mean={diff.mean():.6e}")
    else:
        print(f"❌ Shape mismatch: HF {hf_output.shape} vs FS2 {fs2_output.shape}")

        # Try comparing first prediction
        if hf_output.dim() == 4:
            hf_first = hf_output[0]
            print(f"\nHF first prediction: {hf_first.shape}")
            if hf_first.shape == fs2_output.shape:
                diff = (hf_first - fs2_output).abs()
                print(f"HF[0] vs FS2: max={diff.max():.6e}, mean={diff.mean():.6e}")
            elif fs2_output.dim() == 4:
                fs2_first = fs2_output[0]
                print(f"FS2 first prediction: {fs2_first.shape}")
                diff = (hf_first - fs2_first).abs()
                print(f"HF[0] vs FS2[0]: max={diff.max():.6e}, mean={diff.mean():.6e}")
