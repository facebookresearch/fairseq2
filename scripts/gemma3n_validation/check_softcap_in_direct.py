#!/usr/bin/env python3
"""
Check if HF layer uses attn_logit_softcapping by inspecting the config
from already loaded model. Builds on direct_attention_comparison.py setup.
"""

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

print("="*80)
print("CHECK ATTENTION SPECIAL FEATURES")
print("="*80)

layer0 = hf_model.model.language_model.layers[0]

print(f"\n[Layer 0 self_attn config]")
cfg = layer0.self_attn.config

# Check critical attention parameters
attrs_to_check = [
    'attn_logit_softcapping',
    'query_pre_attn_scalar',
    'final_logit_softcapping',
    'sliding_window',
    'attention_dropout',
    'num_attention_heads',
    'num_key_value_heads',
]

for attr in attrs_to_check:
    if hasattr(cfg, attr):
        val = getattr(cfg, attr)
        print(f"  {attr}: {val}")

        # CRITICAL: Check if softcapping is enabled
        if attr == 'attn_logit_softcapping' and val is not None:
            print(f"\n⚠️  ATTENTION LOGIT SOFTCAPPING IS ENABLED: {val}")
            print(f"   This means HF clamps attention logits: tanh(logits / {val}) * {val}")
            print(f"   FS2 likely does NOT implement this!")

# Check layer 0 self_attn direct attributes
print(f"\n[Layer 0 self_attn attributes]")
direct_attrs = ['sliding_window', 'attention_dropout', 'attn_logit_softcapping']
for attr in direct_attrs:
    if hasattr(layer0.self_attn, attr):
        val = getattr(layer0.self_attn, attr)
        print(f"  {attr}: {val}")

# Check if FS2 has softcap support
print(f"\n[FS2 Layer 0 config]")
fs2_layer0 = fs2_model.decoder.layers[0]
print(f"  FS2 self_attn class: {type(fs2_layer0.self_attn).__name__}")

if hasattr(fs2_layer0.self_attn, 'attn_logit_softcapping'):
    print(f"  ✓ FS2 has attn_logit_softcapping: {fs2_layer0.self_attn.attn_logit_softcapping}")
else:
    print(f"  ❌ FS2 does NOT have attn_logit_softcapping attribute")
    print(f"     This could be the root cause of divergence!")

print("="*80)
