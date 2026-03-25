# Qwen 3.5 HF Parity: Issue, Investigation, and Fix

**Date:** 2026-03-25
**Model:** Qwen3.5-0.8B (`Qwen/Qwen3.5-0.8B`)
**Result:** ✅ PASS — max abs diff 7.63e-06

---

## Problem Statement

After implementing the Qwen 3.5 model family in fairseq2 (Phases 1–4: GatedDeltaNet, Qwen35Attention, decoder layer, factory, interop, MoE), we needed to verify that loading a HuggingFace checkpoint and running inference produces numerically identical logits.

Initial parity test showed:
- State dict loaded with `strict=True` — all 752M parameters matched
- But logits diverged: **max abs diff = 3.79**, **cosine similarity = 0.989**
- HF predicted ` Paris`, fairseq2 predicted `:` for "The capital of France is"

---

## Investigation Approach

### Step 1: Layer-by-Layer Hidden State Comparison

Ran both HF (with `output_hidden_states=True`) and fairseq2 (manually iterating decoder layers), collecting the hidden state tensor after every layer. Compared each pair:

```
Layer  Type                 MaxDiff     CosSim
0      embed                0.00e+00    1.000000   ← perfect
1      linear_attention     0.00e+00    1.000000   ← perfect
2      linear_attention     0.00e+00    1.000000   ← perfect
3      linear_attention     0.00e+00    1.000000   ← perfect
4      full_attention       5.33e-02    0.993110   ← DIVERGENCE STARTS
5-23   mixed                0.04-0.24   ~0.995     ← error accumulates
24     final_norm+proj      3.27e+01    0.937      ← catastrophic
```

**Key finding**: GatedDeltaNet (linear attention) layers 0–3 are **perfect** (zero diff). The error originates at **layer 4, the first full attention layer** (`Qwen35Attention`).

### Step 2: M-RoPE Hypothesis (Red Herring)

Qwen 3.5 uses M-RoPE (Multi-Resolution Rotary Position Embedding) with `mrope_section=[11,11,10]` and `mrope_interleaved=True`. I hypothesized this produced different cos/sin patterns than fairseq2's standard `ReferenceRotaryEncoder`.

**Approach**: Extracted `inv_freq` from both HF and fairseq2, computed cos/sin manually at the same positions, and compared.

**Result**: The `inv_freq` values were **identical**, and the cos/sin values matched to floating-point precision. For text-only input (where all 3 position grids are the same), M-RoPE's interleaving is a **no-op** — replacing values with identical values. This was a dead end.

### Step 3: Sub-Operation Comparison Within Layer 4

With the input to layer 4 confirmed identical (0.00 diff from step 1), I compared every sub-operation of the first full attention layer against HF:

```
Operation                  MaxDiff    Status
Pre-layer3 input           0.00e+00   ✅
After pre-attn norm        0.00e+00   ✅
After q_proj               0.00e+00   ✅
Query/Gate split           0.00e+00   ✅
After q_norm               1.01e+01   ❌  ← BUG HERE
After k_norm               6.06e+00   ❌
After RoPE Q               9.22e+00   ❌  (downstream of norm bug)
After RoPE K               6.06e+00   ❌  (downstream of norm bug)
```

Everything up to the query/gate split was perfect. The **QK-Norm** step introduced massive error (10.1 max diff). RoPE was merely propagating this upstream error.

### Step 4: Root Cause Identification

Inspected the q_norm weights: HF and fairseq2 had **identical** raw values (0.00 diff). The HF weight mean was ~0.43, not ~0.0 (meaning these are trained, not zero-init).

The critical realization: **HF's `Qwen3_5RMSNorm` uses the `(1+weight)` formula for ALL norms**, including q_norm and k_norm:

| | HF computation | fairseq2 computation |
|---|---|---|
| **Formula** | `norm(x) * (1 + 0.43) = norm(x) * 1.43` | `norm(x) * 0.43` |
| **Ratio** | **3.3× difference** | |

The interop weight conversion (`weight += 1.0`) was only applied to:
- `self_attn_layer_norm.weight` ✅
- `ffn_layer_norm.weight` ✅
- `decoder.layer_norm.weight` ✅

But **missed**:
- `self_attn.q_norm.weight` ❌
- `self_attn.k_norm.weight` ❌

---

## Fix

Added two entries to `_QWEN35_RMSNORM_KEYS` in `src/fairseq2/models/qwen/interop.py`:

```python
_QWEN35_RMSNORM_KEYS = (
    "self_attn_layer_norm.weight",
    "ffn_layer_norm.weight",
    "decoder.layer_norm.weight",
    "self_attn.q_norm.weight",    # ← added
    "self_attn.k_norm.weight",    # ← added
)
```

---

## Result After Fix

```
Full-seq logit max  abs diff: 7.63e-06 < 1e-04   ✅ PASS
Cosine similarity:            1.00000048
Top-1 token:                  'Paris' == 'Paris'
Top-5 tokens:                 [Paris, the, located, :, \n] == [Paris, the, located, :, \n]
Unit tests:                   29/29 passing
```

---

## Lessons

1. **Layer-by-layer comparison** is the most powerful diagnostic — it immediately isolates which component is wrong and prevents chasing false hypotheses.

2. **M-RoPE was a red herring** — for text-only inference, the multi-resolution interleaving collapses to standard RoPE. The cos/sin comparison confirmed this quickly.

3. **The `(1+weight)` RMSNorm pattern** in Qwen 3.5 applies to ALL `Qwen3_5RMSNorm` instances, not just the ones explicitly called "layernorm". The original implementation plan correctly identified which norms to convert but missed q_norm/k_norm because they were added as part of the attention module, not the decoder layer.

4. **Sub-operation comparison** (projections → split → norm → RoPE) is essential when layer-level comparison points to a specific layer — it narrows from "something in attention is wrong" to "the norm step specifically" in one diagnostic run.
