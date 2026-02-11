# Gemma3n Parity Status - RESOLVED

## **ROOT CAUSE IDENTIFIED AND FIXED** (2026-02-10)

### The Issue: Attention Scaling

**HF uses `scaling=1.0` (no scaling) because Gemma3n uses QK normalization.**
**FS2 was applying default `1/sqrt(d_k)` scaling in SDPA implementations.**

This caused attention outputs to diverge significantly (max diff ~2.88) despite Q, K, V matching perfectly.

### The Fix

Added configurable `scale` parameter to SDPA implementations:

1. **`NaiveSDPA`** (`src/fairseq2/models/transformer/sdpa/naive.py`):
   - Added `scale: float | None` parameter (default None = 1/sqrt(d_k))
   - Modified `naive_scaled_dot_product_attention()` to use custom scaling
   - Updated `extra_repr()` to show scale value

2. **`TorchSDPA`** (`src/fairseq2/models/transformer/sdpa/torch.py`):
   - Added `scale: float | None` parameter (default None = 1/sqrt(d_k))
   - Pre-scales Q by `scale * sqrt(d_k)` to cancel PyTorch's built-in 1/sqrt(d_k)
   - Updated `extra_repr()` to show scale value

3. **Gemma3n Factory** (`src/fairseq2/models/gemma3n/factory.py`):
   - Updated SDPA creation to pass `scale=1.0`
   - Added comment explaining why (QK normalization)

### Test Results

**Before fix**:
```
Max diff: 2.879671e+00
Mean diff: 2.395230e-01
❌ ATTENTION OUTPUTS DIVERGE
```

**After fix**:
```
Max diff: 2.622604e-06
Mean diff: 2.092344e-07
✅ ATTENTION OUTPUTS MATCH!
```

## Key Technical Details

### Why Gemma3n Uses No Scaling

Gemma3n applies RMSNorm to Q, K, V after projection:
- `q_norm = RMSNorm(head_dim)` on queries
- `k_norm = RMSNorm(head_dim)` on keys
- `v_norm = RMSNorm(head_dim, elementwise_affine=False)` on values

This QK normalization eliminates the need for 1/sqrt(d_k) scaling because:
- Normalized Q and K already have controlled magnitudes
- The dot product is already properly scaled
- Additional 1/sqrt(d_k) would under-scale the attention logits

### HF Implementation

From `transformers/models/gemma3n/modeling_gemma3n.py`:

```python
class Gemma3nTextAttention(nn.Module):
    def __init__(self, config, layer_idx):
        # ...
        self.scaling = 1.0  # No scaling!

    def forward(self, ...):
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=...,
            scaling=self.scaling,  # Passes 1.0
            ...
        )
```

The `eager_attention_forward` function then uses this scaling value:

```python
def eager_attention_forward(module, query, key, value, attention_mask,
                           dropout=0.0, scaling=None, ...):
    if scaling is None:
        scaling = module.head_dim**-0.5  # Default

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling  # Uses 1.0!
    # ...
```

### Investigation Journey

1. **Verified Q, K, V match** (~1e-7 precision) at SDPA input
2. **Verified GQA expansion match** (both use same pattern)
3. **Verified manual SDPA computation matches HF** perfectly
4. **Found divergence persisted** in actual forward pass
5. **Investigated softcapping** - not present in text attention
6. **Checked HF attention config** - found `self.scaling = 1.0`!
7. **Tested scaling hypothesis** - pre-scaling Q by sqrt(d_k) fixed divergence
8. **Implemented proper fix** - added scale parameter to SDPA

## Component Status

✅ **Embedding & Scaling**: Frontend applies sqrt(model_dim) scaling
✅ **AltUp (4D stacking)**: Predict/correct logic matches
✅ **PLE (Per-Layer Embeddings)**: Gating and projection match
✅ **RoPE**: ReferenceRotaryEncoder with halved format matches
✅ **QKV Projections**: Linear projections match
✅ **QKV Normalization**: RMSNorm with correct settings
✅ **GQA Expansion**: repeat_interleave matches HF repeat_kv
✅ **Attention Masks**: CausalAttentionBias with sliding window
✅ **SDPA**: NaiveSDPA with scale=1.0 matches HF eager path
✅ **Output Projection**: Linear layer matches
✅ **LAuReL (Learned Augmented Residual)**: Low-rank residual augmentation
✅ **FFN Sparsity**: Gaussian top-k (95% sparsity) for first 10 layers
✅ **Layer Normalization**: RMSNorm at all normalization points

## Next Steps

1. **Run full 30-layer parity test** to verify end-to-end matching
2. **Test with longer sequences** to verify scaling holds
3. **Test with different model sizes** (E2B, E4B, etc.)
4. **Run generation tests** to verify inference quality
5. **Commit changes** with proper documentation

## Files Modified

### Core Implementation
- `src/fairseq2/models/transformer/sdpa/naive.py` - Added scale parameter
- `src/fairseq2/models/transformer/sdpa/torch.py` - Added scale parameter
- `src/fairseq2/models/gemma3n/factory.py` - Pass scale=1.0 for Gemma3n

### Test Files Created (This Session)
- `scripts/gemma3n_validation/direct_attention_comparison.py` - Clean HF vs FS2 attention test
- `scripts/gemma3n_validation/check_layer0_softcap_value.py` - Verify no softcapping in text layers
- `scripts/gemma3n_validation/test_disable_scaling.py` - Proof of concept for scaling fix
- `scripts/gemma3n_validation/check_softcap_in_direct.py` - Check softcap in direct comparison

### Previous Test Files
- Multiple debug scripts for systematic component verification
- All component-level tests passed before identifying scaling issue
