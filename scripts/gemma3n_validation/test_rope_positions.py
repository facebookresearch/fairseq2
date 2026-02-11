#!/usr/bin/env python3
"""Compare RoPE position handling in HF vs FS2 with fp32 precision."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag

device = torch.device("cpu")
dtype = torch.float32  # Force fp32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)
input_ids = tokenizer("Hi", return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("RoPE Position Comparison (only showing diffs > 1e-6)")
print("="*80)

with torch.no_grad():
    # Setup HF
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding
    rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
    cos, sin = rope(hf_embeds, position_ids, "sliding_attention")

    # Setup FS2
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    seqs, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)

    fs2_layer = fs2_model.decoder.layers[0]
    pos_encoder = fs2_layer.self_attn.pos_encoder

    # Extract FS2 cos/sin
    if not fs2_model.training and state_bag is not None:
        start_step_for_indexing = 1 + state_bag.step_nr
    else:
        start_step_for_indexing = 1

    batch_width = batch_layout.width
    fs2_cos = pos_encoder.cos_freqs[start_step_for_indexing : start_step_for_indexing + batch_width]
    fs2_sin = pos_encoder.sin_freqs[start_step_for_indexing : start_step_for_indexing + batch_width]

    # Compare position embeddings
    hf_cos_seq = cos[0]  # [S, head_dim]
    hf_sin_seq = sin[0]  # [S, head_dim]

    cos_diff = (hf_cos_seq - fs2_cos).abs()
    sin_diff = (hf_sin_seq - fs2_sin).abs()

    if cos_diff.max() > 1e-6 or sin_diff.max() > 1e-6:
        print(f"❌ cos diff: max={cos_diff.max():.6e}, mean={cos_diff.mean():.6e}")
        print(f"❌ sin diff: max={sin_diff.max():.6e}, mean={sin_diff.mean():.6e}")
    else:
        print(f"✓ RoPE position embeddings match (max cos diff: {cos_diff.max():.6e})")

    # Test RoPE application
    test_q = torch.randn(1, input_ids.shape[1], pos_encoder.encoding_dim, dtype=dtype, device=device)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    hf_q_rotated = (test_q * cos) + (rotate_half(test_q) * sin)
    fs2_q_rotated = pos_encoder(test_q, batch_layout, state_bag=state_bag)

    q_diff = (hf_q_rotated - fs2_q_rotated).abs()

    if q_diff.max() > 1e-5:
        print(f"❌ RoPE application diff: max={q_diff.max():.6e}, mean={q_diff.mean():.6e}")
    else:
        print(f"✓ RoPE application matches (max diff: {q_diff.max():.6e})")
