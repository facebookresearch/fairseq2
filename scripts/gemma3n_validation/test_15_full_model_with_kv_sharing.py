"""Test 15: Full model parity with KV sharing enabled."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict

device = torch.device("cpu")
dtype = torch.float32
model_name = "google/gemma-3n-E2B-it"

print("Loading models...")
try:
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, local_files_only=True
    ).eval()
except OSError:
    print("Model not cached, downloading...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    ).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
except OSError:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

print("=" * 80)
print("TEST 15: Full Model Parity with KV Sharing")
print("=" * 80)

# Test input
test_text = "The quick brown fox jumps over the"
input_ids = tokenizer(test_text, return_tensors="pt").input_ids.to(device)
print(f"Input: {test_text}")
print(f"Input shape: {input_ids.shape}")
print()

with torch.no_grad():
    # HuggingFace forward (with use_cache=True - this uses KV sharing)
    print("Running HuggingFace model (use_cache=True)...")
    hf_output = hf_model(input_ids, use_cache=True)
    hf_logits = hf_output.logits
    hf_next_token_logits = hf_logits[0, -1, :]
    hf_top_token = hf_next_token_logits.argmax()
    hf_next_word = tokenizer.decode([hf_top_token])

    # fairseq2 forward (should now use KV sharing)
    print("Running fairseq2 model...")
    fs2_logits = fs2_model(input_ids)
    fs2_next_token_logits = fs2_logits[0, -1, :]
    fs2_top_token = fs2_next_token_logits.argmax()
    fs2_next_word = tokenizer.decode([fs2_top_token])

print()
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"HF next token logits (first 10): {hf_next_token_logits[:10].tolist()}")
print(f"FS2 next token logits (first 10): {fs2_next_token_logits[:10].tolist()}")
print()

logit_diff = (hf_next_token_logits - fs2_next_token_logits).abs()
print(f"Max logit diff: {logit_diff.max().item():.6e}")
print(f"Mean logit diff: {logit_diff.mean().item():.6e}")
print()

print(f"HF predicts: '{hf_next_word}' (token {hf_top_token.item()})")
print(f"FS2 predicts: '{fs2_next_word}' (token {fs2_top_token.item()})")
print()

# Full logits comparison
full_diff = (hf_logits - fs2_logits).abs()
print(f"Full logits max diff: {full_diff.max().item():.6e}")
print(f"Full logits mean diff: {full_diff.mean().item():.6e}")
print()

if logit_diff.max() < 1e-4:
    print("✅ PARITY ACHIEVED: FS2 matches HF with KV sharing!")
else:
    print(f"❌ PARITY FAILED: Max diff {logit_diff.max().item():.6e} exceeds threshold")

print()
print("=" * 80)
print("KV SHARING VERIFICATION")
print("=" * 80)

# Verify KV sharing configuration
print("Checking KV sharing configuration...")
from fairseq2.models.gemma3n.config import get_kv_sharing_config

num_source_layers = 0
num_shared_layers = 0

for layer_idx in range(config.num_layers):
    is_shared, source_idx, is_source = get_kv_sharing_config(
        layer_idx, config.num_layers, config.num_kv_shared_layers
    )
    if is_source:
        num_source_layers += 1
        print(f"  Layer {layer_idx}: KV source")
    if is_shared:
        num_shared_layers += 1
        print(f"  Layer {layer_idx}: KV shared (from layer {source_idx})")

print()
print(f"Total source layers: {num_source_layers}")
print(f"Total shared layers: {num_shared_layers}")
print(f"Expected shared layers: {config.num_kv_shared_layers}")

if num_shared_layers == config.num_kv_shared_layers:
    print("✅ KV sharing configuration correct")
else:
    print(f"❌ KV sharing configuration mismatch")
