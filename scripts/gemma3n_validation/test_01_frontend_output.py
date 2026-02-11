"""Test 1: Verify embeddings + PLE → Layer 0 input match between HF and FS2."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag

device = torch.device("cpu")
dtype = torch.float32
model_name = "google/gemma-3n-E2B-it"

# Load models
print("Loading models...")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

# Tokenize input
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
text = "The quick brown"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
print(f"Input: {text!r}")
print(f"Shape: {input_ids.shape}")
print()

# Prepare FS2 batch layout
seq_lens = [input_ids.shape[1]]
batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)

print("=" * 80)
print("TEST 1: Frontend Output (Embeddings + PLE)")
print("=" * 80)

# HF: Get embedding output
with torch.no_grad():
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)
    print(f"HF embeddings shape: {hf_embeds.shape}")
    print(f"HF embeddings[0, 0, :5]: {hf_embeds[0, 0, :5]}")
    print()

# FS2: Get frontend output (embedding + PLE injection)
with torch.no_grad():
    state_bag = IncrementalStateBag(input_ids.shape[1])
    fs2_frontend_out, padding_mask = fs2_model.decoder_frontend(
        input_ids, batch_layout, state_bag=state_bag
    )
    print(f"FS2 frontend output shape: {fs2_frontend_out.shape}")
    print(f"FS2 frontend output[0, 0, :5]: {fs2_frontend_out[0, 0, :5]}")
    print()

# Compare
print("=" * 80)
print("COMPARISON: HF embeddings vs FS2 frontend output")
print("=" * 80)

diff = (hf_embeds - fs2_frontend_out).abs()
print(f"Max diff:  {diff.max().item():.6e}")
print(f"Mean diff: {diff.mean().item():.6e}")
print(f"Shape match: {hf_embeds.shape == fs2_frontend_out.shape}")

if diff.max() < 1e-5:
    print("✅ MATCH - Frontend outputs are identical")
else:
    print(f"❌ DIVERGENCE - Frontend outputs differ")
    print()
    print("First 3 positions, first 5 dims:")
    for pos in range(min(3, input_ids.shape[1])):
        print(f"\nPosition {pos}:")
        print(f"  HF:  {hf_embeds[0, pos, :5]}")
        print(f"  FS2: {fs2_frontend_out[0, pos, :5]}")
        print(f"  Diff: {diff[0, pos, :5]}")
