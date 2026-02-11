"""Test 4: Isolate AltUp FFN divergence."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag

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
print(f"Input: {text!r}\n")

seq_lens = [input_ids.shape[1]]
batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
state_bag = IncrementalStateBag(input_ids.shape[1])

print("=" * 80)
print("TEST 4: Isolate AltUp FFN Divergence")
print("=" * 80)

with torch.no_grad():
    # Get to the point just before FFN
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)
    fs2_embeds, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)

    # Create 4D
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

    # Setup layers
    hf_layer0 = hf_lm.layers[0]
    fs2_layer0 = fs2_model.decoder.layers[0]

    # Get to pre-FFN input (we know this matches from Test 3)
    import math
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    text_config = hf_model.model.language_model.config
    position_embeddings = {}
    for layer_type in set(text_config.layer_types):
        position_embeddings[layer_type] = hf_lm.rotary_emb(hf_hidden_4d, position_ids, layer_type)

    from fairseq2.models.transformer import AttentionBiasCache
    attn_bias_cache = AttentionBiasCache()

    # HF path
    hf_predictions = hf_layer0.altup.predict(hf_hidden_4d)
    hf_active = hf_predictions[0]
    hf_normed = hf_layer0.input_layernorm(hf_active)
    hf_laurel = hf_layer0.laurel(hf_normed)
    hf_attn, _ = hf_layer0.self_attn(
        hidden_states=hf_normed,
        position_embeddings=position_embeddings[hf_layer0.attention_type],
    )
    hf_attn_norm = hf_layer0.post_attention_layernorm(hf_attn)
    hf_attn_gated = hf_active + hf_attn_norm
    hf_attn_laurel = (hf_attn_gated + hf_laurel) / math.sqrt(2)
    hf_pre_ffn = hf_layer0.pre_feedforward_layernorm(hf_attn_laurel)

    # FS2 path
    fs2_predictions = fs2_layer0.altup(fs2_hidden_4d)
    fs2_active = fs2_predictions[0]
    fs2_normed = fs2_layer0.input_layernorm(fs2_active)
    fs2_laurel = fs2_layer0.laurel(fs2_normed)
    fs2_attn = fs2_layer0.self_attn(
        fs2_normed, batch_layout,
        keys=fs2_normed, keys_layout=batch_layout, values=fs2_normed,
        bias_cache=attn_bias_cache, state_bag=state_bag,
    )
    fs2_attn_norm = fs2_layer0.post_attention_layernorm(fs2_attn)
    fs2_attn_gated = fs2_active + fs2_attn_norm
    fs2_attn_laurel = (fs2_attn_gated + fs2_laurel) / math.sqrt(2.0)
    fs2_pre_ffn = fs2_layer0.pre_feedforward_layernorm(fs2_attn_laurel)

    print("\nFFN Input:")
    diff = (hf_pre_ffn - fs2_pre_ffn).abs()
    print(f"  Max diff: {diff.max().item():.6e}")
    print(f"  Shape: {hf_pre_ffn.shape}")
    print()

    # ========================================================================
    # Detailed FFN inspection
    # ========================================================================
    print("=" * 80)
    print("HF FFN (Gemma3nTextMLP)")
    print("=" * 80)

    hf_mlp = hf_layer0.mlp
    print(f"Config:")
    print(f"  hidden_size: {hf_mlp.hidden_size}")
    print(f"  intermediate_size: {hf_mlp.intermediate_size}")
    print(f"  activation_sparsity: {hf_mlp.activation_sparsity}")
    print(f"  act_fn: {hf_mlp.act_fn}")
    print()

    # Manual HF FFN forward
    print("Manual HF FFN forward:")
    hf_gate = hf_mlp.gate_proj(hf_pre_ffn)
    print(f"  gate_proj output: shape={hf_gate.shape}, range=[{hf_gate.min():.4f}, {hf_gate.max():.4f}]")

    hf_up = hf_mlp.up_proj(hf_pre_ffn)
    print(f"  up_proj output: shape={hf_up.shape}, range=[{hf_up.min():.4f}, {hf_up.max():.4f}]")

    hf_act = hf_mlp.act_fn(hf_gate)
    print(f"  activation output: shape={hf_act.shape}, range=[{hf_act.min():.4f}, {hf_act.max():.4f}]")

    hf_gated = hf_act * hf_up
    print(f"  gated (act * up): shape={hf_gated.shape}, range=[{hf_gated.min():.4f}, {hf_gated.max():.4f}]")

    # Check for sparsity
    if hf_mlp.activation_sparsity is not None and hf_mlp.activation_sparsity > 0:
        print(f"\n  Applying sparsity: {hf_mlp.activation_sparsity}")
        # HF sparsity implementation - need to check what it does
        k = int((1 - hf_mlp.activation_sparsity) * hf_mlp.intermediate_size)
        print(f"  Top-k: {k} out of {hf_mlp.intermediate_size}")

        # Get top-k values
        topk_values, topk_indices = torch.topk(hf_gated, k=k, dim=-1)
        print(f"  Top-k range: [{topk_values.min():.4f}, {topk_values.max():.4f}]")

        # Apply sparsity mask
        hf_sparse = torch.zeros_like(hf_gated)
        hf_sparse.scatter_(-1, topk_indices, topk_values)
        print(f"  Sparse gated: shape={hf_sparse.shape}, range=[{hf_sparse.min():.4f}, {hf_sparse.max():.4f}]")
        print(f"  Non-zero fraction: {(hf_sparse != 0).float().mean():.4f}")

        hf_down = hf_mlp.down_proj(hf_sparse)
    else:
        print(f"\n  No sparsity applied")
        hf_down = hf_mlp.down_proj(hf_gated)

    print(f"  down_proj output: shape={hf_down.shape}, range=[{hf_down.min():.4f}, {hf_down.max():.4f}]")
    print()

    # ========================================================================
    print("=" * 80)
    print("FS2 FFN (StandardFeedForwardNetwork)")
    print("=" * 80)

    fs2_ffn = fs2_layer0.ffn
    print(f"Type: {type(fs2_ffn)}")
    print(f"Attributes: {[a for a in dir(fs2_ffn) if not a.startswith('_')]}")
    print()

    # Manual FS2 FFN forward
    print("Manual FS2 FFN forward:")
    fs2_output = fs2_ffn(fs2_pre_ffn)
    print(f"  FFN output: shape={fs2_output.shape}, range=[{fs2_output.min():.4f}, {fs2_output.max():.4f}]")
    print()

    # ========================================================================
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)

    diff = (hf_down - fs2_output).abs()
    print(f"Max diff: {diff.max().item():.6e}")
    print(f"Mean diff: {diff.mean().item():.6e}")
    print()

    if diff.max() > 1e-3:
        print("❌ MASSIVE DIVERGENCE in FFN")
        print(f"\nHF output sample [0, 0, :5]: {hf_down[0, 0, :5]}")
        print(f"FS2 output sample [0, 0, :5]: {fs2_output[0, 0, :5]}")
        print(f"Diff [0, 0, :5]: {diff[0, 0, :5]}")
    else:
        print("✅ FFN outputs match")
