#!/usr/bin/env python3
"""
Quick test to verify final logit softcapping is applied.
"""

import torch
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model

device = torch.device("cpu")
dtype = torch.float32

print("="*80)
print("VERIFY FINAL LOGIT SOFTCAPPING")
print("="*80)

config = get_gemma3n_e2b_config()
model = create_gemma3n_model(config, device=device, dtype=dtype).eval()

print(f"\n[Gemma3nLM Configuration]")
print(f"  Config final_logit_soft_cap: {config.final_logit_soft_cap}")
print(f"  Config final_logit_softcapping (alias): {config.final_logit_softcapping}")

if hasattr(model, 'final_logit_softcapping'):
    print(f"  ✅ Model has final_logit_softcapping: {model.final_logit_softcapping}")
else:
    print(f"  ❌ Model does NOT have final_logit_softcapping attribute!")

print("\n" + "="*80)

if hasattr(model, 'final_logit_softcapping') and model.final_logit_softcapping == 30.0:
    print("✅ Final logit softcapping IS configured correctly (30.0)")
else:
    print("❌ Final logit softcapping is NOT configured!")

print("="*80)
