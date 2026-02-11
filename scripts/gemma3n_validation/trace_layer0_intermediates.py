#!/usr/bin/env python3
"""Trace all intermediate outputs in layer 0 forward pass (HF vs FS2)."""

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
print("LAYER 0 INTERMEDIATE TRACE (only showing diffs > 1e-5)")
print("="*80)

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

    # Get layers
    hf_layer = hf_lm.layers[0]
    fs2_layer = fs2_model.decoder.layers[0]

    # Prepare HF RoPE
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding
    rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
    cos, sin = rope(hf_hidden_4d, position_ids, "sliding_attention")

    print("\n[LAYER 0 FORWARD - TRACING INTERMEDIATES]")

    # === HF FORWARD (manually step through) ===
    hf_hidden = hf_hidden_4d
    hf_ple = per_layer_inputs[:, :, 0, :]

    # Step 1: AltUp predictions + active
    hf_predictions = hf_layer.altup_predict(hf_hidden)
    hf_active = hf_layer.altup_activate(hf_predictions)

    # Step 2: Apply PLE gating
    hf_ple_gate = hf_layer.per_layer_projection_gate(hf_ple)
    hf_ple_gate = torch.nn.functional.softmax(hf_ple_gate, dim=-1)
    hf_ple_proj = hf_layer.per_layer_projection(hf_ple)
    hf_ple_proj = hf_ple_gate * hf_ple_proj
    hf_ple_proj = hf_ple_proj.reshape(*hf_ple.shape[:-1], -1)
    hf_active = hf_active + hf_ple_proj

    # Step 3: Pre-attention norm
    hf_active_normed = hf_layer.pre_attention_norm(hf_active)

    # Step 4: LAuReL residual
    hf_laurel = hf_hidden + hf_active_normed

    # Step 5: Attention
    from transformers.models.gemma3n.modeling_gemma3n import apply_rotary_pos_emb

    # Get Q, K, V projections
    hf_qkv = hf_layer.self_attn.qkv_proj(hf_laurel)
    bsz, q_len, _ = hf_laurel.shape
    num_heads = hf_layer.self_attn.num_heads
    num_key_value_heads = hf_layer.self_attn.num_key_value_heads
    head_dim = hf_layer.self_attn.head_dim

    hf_qkv = hf_qkv.view(bsz, q_len, num_heads + 2 * num_key_value_heads, head_dim)
    hf_qkv = hf_qkv.transpose(1, 2)
    hf_q, hf_k, hf_v = hf_qkv.split([num_heads, num_key_value_heads, num_key_value_heads], dim=1)

    # Apply QK norm
    hf_q = hf_layer.self_attn.q_norm(hf_q)
    hf_k = hf_layer.self_attn.k_norm(hf_k)

    # Apply RoPE
    hf_q, hf_k = apply_rotary_pos_emb(hf_q, hf_k, cos, sin)

    # Apply V norm
    hf_v = hf_layer.self_attn.v_norm(hf_v)

    # Expand K, V for GQA
    hf_k = hf_k.repeat_interleave(num_heads // num_key_value_heads, dim=1)
    hf_v = hf_v.repeat_interleave(num_heads // num_key_value_heads, dim=1)

    # SDPA
    hf_attn_output = torch.nn.functional.scaled_dot_product_attention(
        hf_q, hf_k, hf_v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
    )
    hf_attn_output = hf_attn_output.transpose(1, 2).contiguous()
    hf_attn_output = hf_attn_output.view(bsz, q_len, -1)
    hf_attn_output = hf_layer.self_attn.o_proj(hf_attn_output)

    # Step 6: Post-attention norm
    hf_post_attn = hf_layer.post_attention_norm(hf_attn_output)

    # Step 7: FFN
    hf_ffn_input = hf_laurel + hf_post_attn
    hf_gate = hf_layer.mlp.gate_proj(hf_ffn_input)

    # Apply Gaussian top-k (95% sparsity for layer 0)
    target_sparsity = torch.tensor(0.95, dtype=torch.float32, device=device)
    normal_dist = torch.distributions.normal.Normal(0, 1)
    std_multiplier = normal_dist.icdf(target_sparsity).to(hf_gate.dtype)
    gate_mean = torch.mean(hf_gate, dim=-1, keepdim=True)
    gate_std = torch.std(hf_gate, dim=-1, keepdim=True, unbiased=False)
    cutoff = gate_mean + gate_std * std_multiplier
    hf_gate = torch.nn.functional.relu(hf_gate - cutoff)

    hf_gate = torch.nn.functional.gelu(hf_gate, approximate="tanh")
    hf_up = hf_layer.mlp.up_proj(hf_ffn_input)
    hf_ffn_output = hf_layer.mlp.down_proj(hf_gate * hf_up)

    # Step 8: Final output
    hf_output = hf_ffn_input + hf_ffn_output

    # === FS2 FORWARD (manually step through) ===
    fs2_hidden = fs2_hidden_4d
    fs2_ple = fs2_per_layer_inputs[:, :, 0, :]

    # Step 1: AltUp predictions + active
    fs2_predictions = fs2_layer.altup_predict(fs2_hidden)
    fs2_active = fs2_layer.altup_activate(fs2_predictions)

    # Step 2: Apply PLE gating
    fs2_ple_gate = fs2_layer.per_layer_projection_gate(fs2_ple)
    fs2_ple_gate = torch.nn.functional.softmax(fs2_ple_gate, dim=-1)
    fs2_ple_proj = fs2_layer.per_layer_projection(fs2_ple)
    fs2_ple_proj = fs2_ple_gate * fs2_ple_proj
    fs2_ple_proj = fs2_ple_proj.reshape(*fs2_ple.shape[:-1], -1)
    fs2_active = fs2_active + fs2_ple_proj

    # Step 3: Pre-attention norm
    fs2_active_normed = fs2_layer.pre_attention_norm(fs2_active)

    # Step 4: LAuReL residual
    fs2_laurel = fs2_hidden + fs2_active_normed

    # Step 5: Attention (call self_attn directly)
    fs2_attn_output = fs2_layer.self_attn(fs2_laurel, batch_layout, state_bag=state_bag)

    # Step 6: Post-attention norm
    fs2_post_attn = fs2_layer.post_attention_norm(fs2_attn_output)

    # Step 7: FFN
    fs2_ffn_input = fs2_laurel + fs2_post_attn
    fs2_ffn_output = fs2_layer.ffn(fs2_ffn_input)

    # Step 8: Final output
    fs2_output = fs2_ffn_input + fs2_ffn_output

    # === COMPARE ALL STAGES ===
    print("\n[AltUp Stage]")
    compare("predictions", hf_predictions, fs2_predictions)
    compare("active (before PLE)", hf_layer.altup_activate(hf_predictions), fs2_layer.altup_activate(fs2_predictions))
    compare("active (after PLE)", hf_active, fs2_active)

    print("\n[Pre-Attention]")
    compare("pre_attention_norm", hf_active_normed, fs2_active_normed)
    compare("LAuReL output", hf_laurel, fs2_laurel)

    print("\n[Attention]")
    compare("attn output", hf_attn_output, fs2_attn_output)

    print("\n[Post-Attention]")
    compare("post_attention_norm", hf_post_attn, fs2_post_attn)
    compare("FFN input", hf_ffn_input, fs2_ffn_input)

    print("\n[FFN]")
    compare("FFN output", hf_ffn_output, fs2_ffn_output)

    print("\n[Final]")
    compare("layer output", hf_output, fs2_output)

    print("\n" + "="*80)
    print("SUMMARY: First divergence point shows where the bug is")
    print("="*80)
