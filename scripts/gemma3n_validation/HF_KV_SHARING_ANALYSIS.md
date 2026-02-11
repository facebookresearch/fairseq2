"""
HF KV SHARING IMPLEMENTATION - LINE BY LINE ANALYSIS
=====================================================

SOURCE: transformers/models/gemma3n/modeling_gemma3n.py lines 1300-1370

## LAYER TYPES

### Layer 13: SOURCE (local, store_full_length_kv=True, is_kv_shared_layer=False)
### Layer 14: SOURCE (global, store_full_length_kv=True, is_kv_shared_layer=False)
### Layer 15: SHARED (local, store_full_length_kv=False, is_kv_shared_layer=True, kv_shared_layer_index=13)


## SCENARIO 1: First Forward Pass (past_key_values=None)
--------------------------------------------------------

### Layer 13 (SOURCE):

Line 1309-1312: Compute Q, apply norm, RoPE, transpose
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, cos, sin)
    query_states = query_states.transpose(1, 2)

Line 1314-1319: Check if shared layer
    if self.is_kv_shared_layer and past_key_values is not None:
        # FALSE (is_kv_shared_layer=False), so skip this block

Line 1320-1328: Compute K/V (this executes)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    key_states = self.k_norm(key_states)
    key_states = apply_rotary_pos_emb(key_states, cos, sin)
    key_states = key_states.transpose(1, 2)

    value_states = self.v_proj(hidden_states).view(hidden_shape)
    value_states = self.v_norm(value_states)
    value_states = value_states.transpose(1, 2)

Line 1330-1345: Handle caching
    if past_key_values is not None:
        # FALSE (past_key_values=None), so skip entire block
        # NO update() call
        # NO storage in shared_layers

Line 1347-1360: Run SDPA with computed K/V
    attn_output, attn_weights = attention_interface(query_states, key_states, value_states, ...)

**RESULT**: K/V computed, used directly, NOT stored (because past_key_values=None)


### Layer 15 (SHARED):

Line 1314-1319: Check if shared layer
    if self.is_kv_shared_layer and past_key_values is not None:
        # FALSE (past_key_values=None), so skip this block
        # DOES NOT retrieve shared K/V!

Line 1320-1328: Compute K/V (this executes)
    # Computes its own K/V just like a normal layer

Line 1330-1345: Handle caching
    if past_key_values is not None:
        # FALSE, skip

Line 1347-1360: Run SDPA with its own computed K/V

**RESULT**: Shared layer computes its own K/V when past_key_values=None!
**CRITICAL**: KV SHARING ONLY ACTIVATES WHEN past_key_values IS NOT NONE!


## SCENARIO 2: With Cache (past_key_values=Cache object, use_cache=True)
-------------------------------------------------------------------------

### Layer 13 (SOURCE):

Line 1320-1328: Compute K/V (shape: [batch, num_kv_heads, seq_len, head_dim])

Line 1330-1345: Handle caching
    if past_key_values is not None:  # TRUE
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "sliding_window": 512}

        if not self.is_kv_shared_layer:  # TRUE (not a shared layer)
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            # For first pass: returns same K/V
            # For generation: concatenates new token's K/V with cached K/V

        if self.store_full_length_kv:  # TRUE (is a source layer)
            if not hasattr(past_key_values, "shared_layers"):
                past_key_values.shared_layers = {}
            past_key_values.shared_layers[self.layer_idx] = key_states, value_states
            # Stores K/V AFTER update() call

**RESULT**: Source layer stores UPDATED K/V (with incremental cache if generation)


### Layer 15 (SHARED):

Line 1314-1319: Check if shared layer
    if self.is_kv_shared_layer and past_key_values is not None:  # TRUE
        key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
        # Retrieves K/V from layer 13
        # This is the K/V that was stored AFTER layer 13's update() call

Line 1320-1328: Skipped (didn't enter else block)

Line 1330-1345: Handle caching
    if past_key_values is not None:  # TRUE
        if not self.is_kv_shared_layer:  # FALSE (is a shared layer)
            # SKIP update() call - shared layers don't update cache

        if self.store_full_length_kv:  # FALSE (not a source)
            # SKIP storage

**RESULT**: Shared layer uses K/V from source, doesn't call update(), doesn't store


## KEY INSIGHTS
--------------

1. **KV sharing ONLY activates when past_key_values is not None**
   - Without cache: All layers compute their own K/V
   - With cache: Shared layers retrieve from sources

2. **Source layers store K/V AFTER past_key_values.update()**
   - First forward: update() is no-op, stores original K/V
   - Generation: update() concatenates, stores concatenated K/V

3. **Shared layers NEVER call past_key_values.update()**
   - They use pre-computed K/V directly
   - No incremental state management for shared layers

4. **The condition matters**:
   - `if self.is_kv_shared_layer and past_key_values is not None:`
   - BOTH conditions must be true to retrieve shared K/V


## MAPPING TO FAIRSEQ2
---------------------

Our problem: We ALWAYS pass state_bag (for PLE storage), so it's never None.
This is equivalent to HF always having past_key_values.

So our implementation should:
1. Always activate KV sharing (because state_bag is never None)
2. Source layers: call state management, then store
3. Shared layers: retrieve, skip state management

But we verified parity works with use_cache=False (past_key_values=None).
This means in that test, HF shared layers computed their own K/V!

**THE REAL QUESTION**:
When HF runs with use_cache=False, does it disable KV sharing?
Let me check what past_key_values is when use_cache=False...
"""
