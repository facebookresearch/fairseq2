# Qwen 3.5: Architecture & Theory

> Reference: `~/transformers/src/transformers/models/qwen3_5/` and `qwen3_5_moe/`

## 1. What is Qwen 3.5?

Qwen 3.5 is a **hybrid** language model that mixes two types of layers:

- **Linear Attention (75% of layers):** Gated DeltaNet — a recurrent mechanism using causal convolution + gated delta rule. Constant memory during decoding (no KV cache growth).
- **Full Attention (25% of layers):** Standard multi-head attention with **output gating** and **partial rotary embeddings**.

It comes in two variants: **dense** (standard FFN) and **MoE** (sparse experts + shared expert).

## 2. Layer Pattern

```python
# configuration_qwen3_5.py lines 108-111
layer_types = [
    "linear_attention" if bool((i + 1) % interval) else "full_attention"
    for i in range(num_layers)
]
# → [linear, linear, linear, FULL, linear, linear, linear, FULL, ...]
```

Every 4th layer is full attention. The rest are linear attention (GatedDeltaNet).

## 3. RMSNorm: Non-Standard `(1 + weight)` Formula

Unlike Llama's `weight * norm(x)` with weight=ones, Qwen 3.5 uses:

```python
# modeling_qwen3_5.py lines 798-812
self.weight = nn.Parameter(torch.zeros(dim))  # zeros, not ones
output = norm(x) * (1.0 + self.weight)        # (1+w), not w
```

This is mathematically equivalent at init (both produce `norm(x) * 1.0`), but the parameter space is centered at zero instead of one. All layer norms (`input_layernorm`, `post_attention_layernorm`, final `norm`) use this variant. The GatedDeltaNet's internal `RMSNormGated` does NOT — it uses standard `weight=ones`.

## 4. Partial Rotary Position Embeddings

`head_dim = 256`, but `partial_rotary_factor = 0.25`. Only **64 of 256 dimensions** are rotated:

```python
# configuration_qwen3_5.py line 105
partial_rotary_factor = 0.25  # → rotary_dim = 256 * 0.25 = 64

# modeling_qwen3_5.py lines 654-666
q_rot, q_pass = q[..., :64], q[..., 64:]    # split at rotary_dim
k_rot, k_pass = k[..., :64], k[..., 64:]
q_embed = torch.cat([rotate(q_rot, cos, sin), q_pass], dim=-1)
k_embed = torch.cat([rotate(k_rot, cos, sin), k_pass], dim=-1)
```

The unrotated 192 dims act as position-independent features — useful for retrieval-like behavior.

## 5. Gated Full Attention

```
┌─────────────────────────────────────────────────┐
│ x ──→ q_proj(x) ──→ [query | gate]  (split)    │
│       k_proj(x) ──→  key                        │
│       v_proj(x) ──→  value                      │
│                                                  │
│ query, key ──→ QK-Norm ──→ Partial RoPE         │
│            ──→ SDPA(query, key, value)           │
│            ──→ attn_output * sigmoid(gate)       │
│            ──→ o_proj(gated_output)              │
└─────────────────────────────────────────────────┘
```

Key details from HF `Qwen3_5Attention` (lines 707-779):

| Feature | Detail | HF Line |
|---------|--------|---------|
| Q projection | `num_heads * head_dim * 2` (doubled) | 719 |
| Split | `query, gate = chunk(q_proj(x), 2, dim=-1)` | 745-746 |
| Gate shape | `(*batch, seq, num_heads * head_dim)` — flattened heads | 748 |
| QK-Norm | `RMSNorm(head_dim)` per head dim (not full dim) | 731-732 |
| Partial RoPE | 64/256 dims rotated | 755 |
| Output gate | `attn_output * sigmoid(gate)` | 776 |
| Bias | None (`attention_bias=False`) | 89, 719-727 |

## 6. Gated DeltaNet (Linear Attention)

```
┌──────────────────────────────────────────────────────────┐
│ x ──→ in_proj_qkv(x) ──→ Causal Conv1d(silu) ──→ [Q,K,V]│
│       in_proj_z(x) ──→ z (gate for output norm)          │
│       in_proj_b(x) ──→ beta = sigmoid(b)                 │
│       in_proj_a(x) ──→ g = -exp(A_log) * softplus(a+dt)  │
│                                                           │
│ Q, K ──→ GQA expand (if num_v > num_k)                   │
│       ──→ l2norm(Q), l2norm(K)                            │
│       ──→ chunk_gated_delta_rule(Q,K,V,g,beta) [prefill]  │
│       or  fused_recurrent_gated_delta_rule     [decode]    │
│       ──→ RMSNormGated(output, silu(z))                   │
│       ──→ out_proj                                        │
└──────────────────────────────────────────────────────────┘
```

### 6.1 Dimensions

| Param | Default | Derivation |
|-------|---------|------------|
| `linear_num_key_heads` | 16 | config |
| `linear_num_value_heads` | 32 | config |
| `linear_key_head_dim` | 128 | config |
| `linear_value_head_dim` | 128 | config |
| `key_dim` | 2048 | 16 × 128 |
| `value_dim` | 4096 | 32 × 128 |
| `conv_dim` | 8192 | 2048×2 + 4096 |

### 6.2 Projections (lines 506-509)

```python
in_proj_qkv = Linear(hidden_size, key_dim*2 + value_dim, bias=False)  # → 8192
in_proj_z   = Linear(hidden_size, value_dim, bias=False)               # → 4096
in_proj_b   = Linear(hidden_size, num_v_heads, bias=False)             # → 32
in_proj_a   = Linear(hidden_size, num_v_heads, bias=False)             # → 32
```

### 6.3 Causal Conv1d (lines 464-471)

Depthwise convolution over the QKV stream:
```python
Conv1d(conv_dim, conv_dim, kernel_size=4, groups=conv_dim, bias=False, padding=3)
```
Followed by `silu` activation. During decode, uses a sliding conv state buffer of size `(B, conv_dim, 3)`.

### 6.4 Gating Formulas (lines 578-580)

```python
beta = sigmoid(b)                                    # write gate ∈ (0,1)
g = -exp(A_log) * softplus(a + dt_bias)              # forget gate (negative log-space)
# A_log init: log(uniform(0, 16)), dt_bias init: ones
```

### 6.5 GQA in DeltaNet (line 581-583)

When `num_v_heads (32) > num_k_heads (16)`, Q and K are expanded:
```python
query = query.repeat_interleave(num_v_heads // num_k_heads, dim=2)  # 16→32 heads
key = key.repeat_interleave(num_v_heads // num_k_heads, dim=2)
```

### 6.6 Delta Rule Core

**Prefill** uses chunked computation (`chunk_gated_delta_rule`, chunk_size=64):
- Splits sequence into chunks, processes intra-chunk with triangular attention
- Maintains inter-chunk state as `(num_v_heads, head_k_dim, head_v_dim)` matrix
- Always applies `l2norm` to Q and K before computation

**Decode** uses step-by-step recurrence (`fused_recurrent_gated_delta_rule`):
```python
state = state * exp(g)                       # forget
kv_mem = (state * k).sum(dim=-2)             # retrieve
delta = (v - kv_mem) * beta                  # compute update
state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)  # write
output = (state * q).sum(dim=-2)             # read
```

### 6.7 Output Norm (lines 264-279)

`Qwen3_5RMSNormGated` — **silu gating** (not sigmoid!):
```python
output = RMSNorm(x) * silu(z)   # z from in_proj_z
# weight initialized to ones, standard RMSNorm formula
```

## 7. MoE Architecture

### 7.1 Sparse MoE Block (lines 860-879)

```
┌────────────────────────────────────────────────────┐
│ x ──→ Router(x) ──→ softmax ──→ top-k ──→ renorm  │
│    ──→ Experts(x, indices, weights)                 │
│                                                     │
│ x ──→ SharedExpert(x) * sigmoid(SharedGate(x))     │
│                                                     │
│ output = expert_out + shared_expert_out             │
└────────────────────────────────────────────────────┘
```

### 7.2 Router (lines 841-857)

```python
logits = F.softmax(F.linear(x, self.weight), dtype=float, dim=-1)  # softmax!
top_values, top_indices = torch.topk(logits, top_k, dim=-1)
top_values /= top_values.sum(dim=-1, keepdim=True)  # renormalize
# NOTE: NOT sigmoid like Llama4. Cannot reuse Llama4 MoE.
```

### 7.3 Expert Storage (lines 802-838)

```python
gate_up_proj = Parameter(num_experts, 2 * moe_intermediate, hidden_size)
down_proj = Parameter(num_experts, hidden_size, moe_intermediate)
```
Per-expert forward: `silu(gate) * up → down`, using `F.linear` with indexed 2D slices.

### 7.4 Shared Expert (lines 860-879)

Standard GLU MLP gated by a learned scalar:
```python
shared_out = shared_expert_mlp(x)
gate_score = sigmoid(Linear(x, out_features=1))
shared_out = gate_score * shared_out
```

### 7.5 MoE Config Defaults

| Param | Default |
|-------|---------|
| `num_experts` | 256 |
| `num_experts_per_tok` | 8 |
| `moe_intermediate_size` | 512 |
| `shared_expert_intermediate_size` | 512 |
| `router_aux_loss_coef` | 0.001 |

## 8. Dual Cache System (lines 69-156)

```python
class Qwen3_5DynamicCache:
    conv_states: list[Tensor | None]       # GDN: (B, conv_dim, kernel-1)
    recurrent_states: list[Tensor | None]  # GDN: (B, num_v_heads, k_dim, v_dim)
    key_cache: list[Tensor | None]         # Attn: (B, heads, seq, head_dim)
    value_cache: list[Tensor | None]       # Attn: (B, heads, seq, head_dim)
```

Linear attention layers only touch conv/recurrent states. Full attention layers only touch key/value caches. Sequence length is tracked via full attention layers only.

## 9. Dense Config Defaults (`Qwen3_5TextConfig`)

| Param | Value | Note |
|-------|-------|------|
| `vocab_size` | 248320 | Much larger than Qwen3 (151936) |
| `hidden_size` | 4096 | |
| `intermediate_size` | 12288 | |
| `num_hidden_layers` | 32 | |
| `num_attention_heads` | 16 | |
| `num_key_value_heads` | 4 | GQA 4:1 |
| `head_dim` | 256 | Explicit (not hidden/heads) |
| `partial_rotary_factor` | 0.25 | 64 of 256 dims rotated |
| `attention_bias` | False | |
| `rms_norm_eps` | 1e-6 | |
| `hidden_act` | silu | |
| `tie_word_embeddings` | False | |

**MoE overrides:** `hidden=2048, layers=40, kv_heads=2`
