# Gemma3n Parity Implementation Status

## ✅ PARITY ACHIEVED (2026-02-11)

Full inference parity between fairseq2 and HuggingFace Transformers implementations.

### Final Parity Results
- **Token prediction agreement**: 100%
- **Max absolute diff**: 1.39e-04
- **Max relative diff**: 2.67e-03
- **Test**: `scripts/gemma3n_validation/test_parity.py`

## Critical Components for Parity

### 1. KV Projection Sharing ⭐ MOST COMPLEX
Consumer layers reuse K/V projections from source layers instead of computing their own.

**Configuration**:
- `num_kv_shared_layers = 10` (NOT 15 - critical bug!)
- SOURCE layers: Layer 18 (local), Layer 19 (global)
- CONSUMER layers: Layers 20-29 (10 layers total)

**Implementation**:
- `KVProjectionType` enum (LOCAL/GLOBAL) for type-safe slot access
- `KVProjectionRole` enum (SOURCE/CONSUMER/NONE) for layer roles
- Decoder creates 2-slot dict: `{KVProjectionType.LOCAL: None, KVProjectionType.GLOBAL: None}`
- SOURCE layers store via callback: `kv_projection_slots.update({slot_key: (k, v)})`
- CONSUMER layers retrieve: `pre_computed_kv = kv_projection_slots[slot_key]`

**Key Insight**: Each implementation stores K/V in its native format (HF: B,H,S,D; FS2: B,S,H,D).

### 2. Activation Sparsity
Required for both local AND global layers (not in paper, found via debugging).

**Configuration**:
- First 10 layers: `activation_sparsity = 0.95`
- Remaining layers: `activation_sparsity = 0.0`
- Applied to BOTH AltUpFeedForwardNetwork and GLUFeedForwardNetwork

**Implementation**: Gaussian top-k selection in FFN.

### 3. AltUp Predict/Correct
4D tensor processing with predict → process → correct pattern.

**Flow**:
1. Stack embeddings to 4D: `[4, batch, seq, dim]`
2. Each layer: `predict()` → attention+FFN → `correct()` all versions
3. Unstack: average 4 versions back to 3D

**Config**: `altup_num_inputs=4`, `altup_active_idx=0`, `altup_coef_clip=120.0`

### 4. Per-Layer Embeddings (PLE)
Separate embeddings per layer that augment hidden states.

**Config**: `vocab_size_per_layer_input=262144`, `hidden_size_per_layer_input=256`

### 5. LAuReL (Learned Augmented Residual)
Low-rank transformation: `model_dim → rank → model_dim`
Combined: `(attn_gated + laurel_output) / sqrt(2)`

**Config**: `laurel_rank=64`

### 6. RoPE with Dual Theta
- Local layers: `rope_theta = 10,000`
- Global layers: `rope_theta_global = 1,000,000`

### 7. QK Normalization with scale=1.0
RMSNorm on Q/K/V + SDPA with `scale=1.0` (no 1/sqrt(d_k) scaling)

### 8. Attention Softcapping
Tanh-based softcapping: `final_logit_soft_cap = 30.0`

### 9. Sliding Window Attention
Local layers: 512-token window
Global layers: Full causal

### 10. GELU with Tanh Approximation
`F.gelu(x, approximate="tanh")`

## Debugging Journey

### Issue 1: Activation Sparsity
**Symptom**: Layer 0 FFN divergence
**Root Cause**: HF had sparsity enabled, FS2 didn't
**Fix**: Added `activation_sparsity=0.95` to first 10 layers

### Issue 2: num_kv_shared_layers
**Symptom**: Consumer layer 20 exploded (61.9 max diff)
**Root Cause**: FS2 used 15, HF used 10
**Fix**: Changed to `num_kv_shared_layers = 10`

### Issue 3: Test Parameter Name
**Symptom**: HF not storing K/V
**Root Cause**: Test used `past_key_value` instead of `past_key_values` (plural)
**Fix**: Corrected parameter name
**Result**: Layer 20 went from 61.9 → 9.16e-05 (675,000x improvement!)

## Layer Configuration (30 layers)

| Layers | Type | Attention | FFN | RoPE | KV Role | Sparsity |
|--------|------|-----------|-----|------|---------|----------|
| 0-3 | Local | Sliding | AltUp | 10k | NONE | 0.95 |
| 4 | Global | Full | GLU | 1M | NONE | 0.95 |
| 5-8 | Local | Sliding | AltUp | 10k | NONE | 0.95 |
| 9 | Global | Full | GLU | 1M | NONE | 0.95 |
| 10-13 | Local | Sliding | AltUp | 10k | NONE | 0.0 |
| 14 | Global | Full | GLU | 1M | NONE | 0.0 |
| 15-17 | Local | Sliding | AltUp | 10k | NONE | 0.0 |
| **18** | **Local** | **Sliding** | **AltUp** | **10k** | **SOURCE** | **0.0** |
| **19** | **Global** | **Full** | **GLU** | **1M** | **SOURCE** | **0.0** |
| 20-23 | Local | Sliding | AltUp | 10k | CONSUMER (18) | 0.0 |
| 24 | Global | Full | GLU | 1M | CONSUMER (19) | 0.0 |
| 25-28 | Local | Sliding | AltUp | 10k | CONSUMER (18) | 0.0 |
| 29 | Global | Full | GLU | 1M | CONSUMER (19) | 0.0 |

## Files Modified

### Core Implementation
- `src/fairseq2/models/gemma3n/config.py` - Config with KV sharing
- `src/fairseq2/models/gemma3n/decoder.py` - Decoder with AltUp 4D
- `src/fairseq2/models/gemma3n/decoder_layer.py` - Layer with LAuReL, AltUp, PLE
- `src/fairseq2/models/gemma3n/factory.py` - Factory for components
- `src/fairseq2/models/gemma3n/frontend.py` - Frontend with PLE
- `src/fairseq2/models/gemma3n/model.py` - Model with softcapping
- `src/fairseq2/models/gemma3n/altup.py` - AltUp implementation
- `src/fairseq2/models/gemma3n/kv_projection.py` - KV sharing enums
- `src/fairseq2/models/gemma3n/interop.py` - Checkpoint conversion

### Infrastructure
- `src/fairseq2/models/transformer/multihead_attention.py` - pre_computed_kv support
- `src/fairseq2/models/transformer/ffn.py` - AltUpFeedForwardNetwork
- `src/fairseq2/models/transformer/sdpa/naive.py` - Softcapping
- `src/fairseq2/models/transformer/sdpa/torch.py` - Softcapping

## Testing

```bash
python scripts/gemma3n_validation/test_parity.py
```

Expected: 100% token agreement, max abs diff < 1e-3, max rel diff < 1e-2
