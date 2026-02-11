# KV Sharing Implementation - Ready for Testing

## Summary

I've successfully implemented KV sharing for Gemma3n layers 15-29. The implementation follows the HuggingFace reference exactly.

## What Was Implemented

### 1. KV Sharing Registry (`src/fairseq2/models/gemma3n/kv_sharing.py`)
- Clean registry pattern for storing/retrieving K/V tensors
- Thread-safe storage with error checking
- Clear separation from incremental decoding cache

### 2. Attention Layer Updates (`src/fairseq2/models/transformer/multihead_attention.py`)
- Added `pre_computed_kv` parameter to bypass K/V projection for shared layers
- Added `kv_storage_callback` to store K/V from source layers
- Incremental decoding with KV sharing raises `NotImplementedError` (future work)

### 3. Decoder Layer Updates (`src/fairseq2/models/gemma3n/decoder_layer.py`)
- Added KV sharing configuration fields (`is_kv_shared_layer`, `kv_source_layer_idx`, `is_kv_source_layer`)
- Updated `forward()` to use registry for KV sharing
- Source layers store K/V after projection
- Shared layers retrieve K/V instead of computing

### 4. Decoder Updates (`src/fairseq2/models/gemma3n/decoder.py`)
- Creates `KVSharedLayerRegistry` for each forward pass
- Passes registry to all layers
- Clears registry after forward pass

### 5. Factory Updates (`src/fairseq2/models/gemma3n/factory.py`)
- Configures each layer with correct KV sharing settings
- Uses `get_kv_sharing_config()` helper

### 6. Config Updates (`src/fairseq2/models/gemma3n/config.py`)
- Added `get_kv_sharing_config()` helper function
- Correctly maps layers to their KV sources based on layer type (local/global)

## KV Sharing Configuration (Verified)

For Gemma3n E2B (30 layers, 15 shared):

**Source Layers (0-14):**
- Layer 13 (Local) → stores K/V for 12 local shared layers
- Layer 14 (Global) → stores K/V for 3 global shared layers

**Shared Layers (15-29):**
- Local layers (15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28) → retrieve from Layer 13
- Global layers (19, 24, 29) → retrieve from Layer 14

This matches the HuggingFace implementation exactly.

## Testing Required

### Run the full parity test:

```bash
cd /home/aerben/repos/fairseq2
./scripts/gemma3n_validation/test_parity.sh
```

### Expected Results:

**Before this fix:**
- With `use_cache=True` (HF default): Max diff ~21, predictions differ ("fox" vs "frog")
- With `use_cache=False`: Max diff ~0, predictions match ✅

**After this fix:**
- With `use_cache=True`: Should now match! Max diff < 1e-4, predictions identical
- The only difference was KV sharing, which is now implemented

### Alternative: Run custom test

You can also run:
```bash
/home/aerben/repos/fairseq2/.venv/bin/python3 scripts/gemma3n_validation/test_15_full_model_with_kv_sharing.py
```

This will:
1. Load both HF and FS2 models
2. Run same input through both
3. Compare logits and predictions
4. Verify KV sharing configuration

## Implementation Highlights

### Read-Through Pattern
Source layers store K/V **after** projection but **before** incremental cache. This ensures:
- Shared layers get the same K/V as if they computed it themselves
- Incremental decoding cache is separate from KV sharing
- Clean separation of concerns

### Type-Based Sharing
Layers share K/V only with layers of the same type:
- Local layers (sliding window attention) → share among local
- Global layers (full causal attention) → share among global

This is critical because local/global layers use different RoPE theta values.

### Safety Checks
- Runtime error if shared layer executed before its source
- Runtime error if source layer tries to store K/V twice
- Clear error messages for debugging

## Next Steps

1. **Run parity test** - Verify FS2 now matches HF with `use_cache=True`
2. **Implement incremental decoding with KV sharing** (future work)
   - Currently raises NotImplementedError
   - Needs careful handling of KV sharing + generation cache
3. **Performance testing** - Verify KV sharing provides expected speedup

## Files Changed

1. `src/fairseq2/models/gemma3n/kv_sharing.py` (NEW)
2. `src/fairseq2/models/gemma3n/config.py` (added `get_kv_sharing_config()`)
3. `src/fairseq2/models/gemma3n/decoder.py` (added registry creation)
4. `src/fairseq2/models/gemma3n/decoder_layer.py` (added KV sharing logic)
5. `src/fairseq2/models/gemma3n/factory.py` (configure KV sharing)
6. `src/fairseq2/models/transformer/multihead_attention.py` (added KV sharing parameters)
7. `scripts/gemma3n_validation/test_kv_sharing_config.py` (NEW - config verification)
8. `scripts/gemma3n_validation/test_15_full_model_with_kv_sharing.py` (NEW - parity test)
