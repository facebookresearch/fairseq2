#!/usr/bin/env python3
"""Compare full model forward HF vs FS2."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict

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

print("="*80)
print("Full Model Forward: HF vs FS2")
print("="*80)

with torch.no_grad():
    # HF full forward
    hf_outputs = hf_model(input_ids, output_hidden_states=True)
    hf_logits = hf_outputs.logits
    hf_hidden_states = hf_outputs.hidden_states  # Tuple of hidden states

    print(f"\nHF logits: shape={hf_logits.shape}, mean={hf_logits.mean():.6f}")
    print(f"HF hidden states: {len(hf_hidden_states)} layers")
    print(f"HF final hidden: shape={hf_hidden_states[-1].shape}, mean={hf_hidden_states[-1].mean():.6f}")

    # FS2 full forward
    from fairseq2.nn import BatchLayout
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)

    fs2_logits = fs2_model(input_ids, batch_layout)
    print(f"\nFS2 logits: shape={fs2_logits.shape}, mean={fs2_logits.mean():.6f}")

    # Compare logits
    diff = (hf_logits - fs2_logits).abs()
    print(f"\nLogits diff: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Token predictions
    hf_tokens = hf_logits.argmax(dim=-1)
    fs2_tokens = fs2_logits.argmax(dim=-1)
    match_rate = (hf_tokens == fs2_tokens).float().mean()

    print(f"\nToken match rate: {match_rate:.2%}")
    print(f"HF tokens: {hf_tokens}")
    print(f"FS2 tokens: {fs2_tokens}")

    if match_rate < 1.0:
        print("\n❌ Token mismatch detected")
    else:
        print("\n✓ All tokens match!")
