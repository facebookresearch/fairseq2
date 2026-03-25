# Qwen 3.5 Key Design Decisions

Detailed rationale for every architectural and implementation decision.

---

## Decision 1: Same `qwen/` Directory, Separate Model Families

### What we decided

All Qwen 3.5 code lives in `src/fairseq2/models/qwen/` alongside Qwen 2.5/3.0. Registers as separate families (`"qwen3_5"`, `"qwen3_5_moe"`) distinct from `"qwen"`.

### Why Qwen 2.5 and 3.0 share a family but 3.5 cannot

Qwen 2.5 → 3.0 differs only in **config values** on the same `QwenConfig`:
- `qkv_proj_bias`: True → False
- `q_norm` / `k_norm`: False → True
- `head_dim`: None → 128

Same factory, same `StandardMultiheadAttention`, same `StandardTransformerLMDecoderLayer`.

Qwen 3.5 introduces **new module types**:

| Aspect | Qwen 2.5/3.0 | Qwen 3.5 |
|--------|---------------|----------|
| Token mixer | `StandardMultiheadAttention` (all layers) | `Qwen35Attention` OR `GatedDeltaNet` (per-layer) |
| Decoder layer | `StandardTransformerLMDecoderLayer` | `Qwen35DecoderLayer` (hybrid dispatch) |
| RoPE | Full rotation | Partial (25% of dims) |
| RMSNorm | `weight * norm(x)` | `(1+weight) * norm(x)` — needs interop conversion |
| State dict keys | Only `self_attn.*` | Both `self_attn.*` AND `linear_attn.*` |
| Cache | KV cache only | Dual: KV cache + conv/recurrent state |

**Rule:** Same config values → same family. Different module classes → separate family.

### Alternatives considered

1. **3 separate directories** (`qwen/`, `qwen3_5/`, `qwen3_5_moe/`): Duplicates tokenizer, hub, parts of interop. Rejected.
2. **Single mega-family**: 35+ config fields, half irrelevant per model, tangled factory conditionals. Rejected.

### What's shared vs separate

- **Shared:** `tokenizer.py`, hub patterns, directory, `__init__.py`
- **Separate:** config, factory, interop converter, family registration

---

## Decision 2: RMSNorm `(1+weight)` Handled in Interop

### The problem

HF `Qwen3_5RMSNorm`: `weight = zeros`, formula = `norm(x) * (1.0 + weight)`.
fairseq2 `RMSNorm`: `weight = ones`, formula = `norm(x) * weight`.
Both produce `norm(x) * 1.0` at init, but parameter spaces differ.

### What we decided

Convert in `interop.py` during checkpoint loading: `weight += 1.0`. The factory creates standard `RMSNorm`.

### Why not a custom `Qwen35RMSNorm` class?

A custom norm would ripple through:
- `create_layer_norm()` in factory needs to know which variant
- `RMSNormGated` wraps `RMSNorm` — would need variant awareness
- FSDP, activation checkpointing, compilation need to recognize new class
- `isinstance(module, RMSNorm)` checks elsewhere would miss it

Interop approach is **invisible** — factory creates standard `RMSNorm`, conversion happens once at load time.

### Which weights are converted

| Weight | Converted? | Reason |
|--------|-----------|--------|
| `input_layernorm.weight` | ✅ Yes | HF `(1+w)` formula |
| `post_attention_layernorm.weight` | ✅ Yes | HF `(1+w)` formula |
| `model.norm.weight` (final) | ✅ Yes | HF `(1+w)` formula |
| `linear_attn.norm.weight` (GDN) | ❌ No | Standard `w` formula, `weight=ones` |

Enforced by `_QWEN35_RMSNORM_KEYS` tuple matching specific suffixes.

### Caveat

One-way conversion only. HF export needs `weight -= 1.0` reverse path — tracked for Phase 5.

### Comparison with OLMo2 RMSNorm (a different problem entirely)

OLMo2 also has a non-standard RMSNorm, but the issue is **completely different**:

| | **Qwen 3.5** | **OLMo2** | **LLaMA (standard)** |
|---|---|---|---|
| **HF weight init** | `torch.zeros(dim)` | `torch.ones(dim)` | `torch.ones(dim)` |
| **HF forward** | `norm(x) * (1.0 + w)` | `(w * norm(x)).to(dtype)` | `w * norm(x).to(dtype)` |
| **What's different** | Weight **parameterization** | Dtype **casting order** | (baseline) |
| **fairseq2 solution** | Interop weight conversion `+= 1.0` | Custom `OLMORMSNorm` class | Standard `RMSNorm` |
| **Weight conversion needed?** | ✅ Yes | ❌ No | ❌ No |
| **Custom norm class needed?** | ❌ No | ✅ Yes | ❌ No |

**Qwen 3.5's problem** is a mathematical reparameterization: `(1 + w)` with `w=0` equals `w` with `w=1`. Converting once at load time (`weight += 1.0`) makes the checkpoint compatible with standard `RMSNorm`. No runtime difference.

**OLMo2's problem** is a runtime precision difference:
- LLaMA: `norm(x.float()).to(fp16) * weight` — cast to fp16 THEN multiply (weight stays fp32, result is fp16)
- OLMo2: `(weight * norm(x.float())).to(fp16)` — multiply in fp32 THEN cast (more precise)

This cannot be solved by weight conversion — it's a different computation order. So fairseq2 created `OLMORMSNorm` in `src/fairseq2/models/olmo/normalization.py` as a separate class.

**Why Qwen 3.5 doesn't need a custom class:** The `(1+w)` formula is algebraically equivalent to standard `w` multiplication after converting the weights. The forward pass is identical. OLMo2's casting-order difference produces numerically different results even with identical weights.

---

## Decision 3: Partial RoPE Handled Inside Attention

### The problem

`head_dim = 256`, `partial_rotary_factor = 0.25` → only first **64 of 256** dims rotated. fairseq2's `RotaryEncoder` doesn't support partial rotation.

### What we decided

Split/rotate/concat inside `Qwen35Attention.forward()`:

```python
if encoding_dim < self.head_dim:
    q_rot, q_pass = q[..., :encoding_dim], q[..., encoding_dim:]
    k_rot, k_pass = k[..., :encoding_dim], k[..., encoding_dim:]
    q_rot = self.pos_encoder(q_rot, ...)
    k_rot = self.pos_encoder(k_rot, ...)
    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1)
```

Factory creates `ReferenceRotaryEncoder(encoding_dim=64)`.

### Alternatives considered

1. **`PartialRotaryEncoder` wrapper**: Encapsulates split/rotate/concat. Cleaner abstraction but adds a class nothing else uses. `encoding_dim` property ambiguity (report 256 or 64?).
2. **Modify `ReferenceRotaryEncoder`**: Invasive to shared component used by Llama/OLMo/Mistral. High regression risk.

### Safety verification

`pos_encoder` is called **twice** per forward (once for q_rot, once for k_rot). **Verified safe:** `ReferenceRotaryEncoder.forward()` reads `state_bag.step_nr` but does NOT modify it. Step counter is advanced by the caller (decoder loop), not the encoder.

---

## Decision 4: Native Torch for Conv1d and Parameter

### The situation

`GatedDeltaNet` uses `nn.Conv1d` and `nn.Parameter` — no fairseq2 wrappers exist.

### FSDP impact: None

fairseq2 FSDP uses `use_orig_params=True`, discovering ALL parameters via `named_parameters()` recursively:

```python
# fairseq2/nn/fsdp/fsdp1.py
FSDP1Module(module, use_orig_params=True, ...)
```

When `apply_fsdp_to_transformer_lm` wraps each decoder layer, FSDP finds `conv1d.weight`, `dt_bias`, `A_log` and includes them in the shard — same as `Linear.weight`.

**Evidence:** Llama4 MoE uses bare `nn.Parameter` for router (`self.router = nn.Parameter(...)`) and works with FSDP.

### Tensor Parallelism impact: Limited but acceptable

| Module | TP Sharder | Behavior |
|--------|-----------|----------|
| `fairseq2.nn.Linear` | `LinearSharder` | Sharded across TP ranks |
| `nn.Conv1d` | ❌ None | **Replicated** across TP ranks |
| `nn.Parameter` (dt_bias, A_log) | ❌ None | **Replicated** across TP ranks |

Acceptable because:
- **Conv1d weight**: `(8192, 1, 4)` = 32K params = 128KB per rank. Negligible.
- **dt_bias, A_log**: 32 floats each = 128 bytes. Negligible.
- **Linear projections** (in_proj_qkv, etc.) ARE `fairseq2.nn.Linear` and CAN be TP-sharded.

### Future `GatedDeltaNetSharder`

```python
class GatedDeltaNetSharder(ModuleSharder):
    def shard(self, module, gangs, spec):
        module.in_proj_qkv = ColumnShardedLinear.from_linear(module.in_proj_qkv, gangs.tp)
        module.in_proj_z   = ColumnShardedLinear.from_linear(module.in_proj_z, gangs.tp)
        module.out_proj    = RowShardedLinear.from_linear(module.out_proj, gangs.tp)
        # Conv1d and dt_bias/A_log remain replicated (tiny)
        return module
```

Tracked for Phase 5.

---

## Decision 5: Attribute Names Match HuggingFace

### What we decided

`Qwen35DecoderLayer` uses attribute names that exactly match HF:

```python
if layer_type == "full_attention":
    self.register_module("self_attn", self_attn)      # matches HF
    self.register_module("linear_attn", None)
elif layer_type == "linear_attention":
    self.register_module("self_attn", None)
    self.register_module("linear_attn", linear_attn)  # matches HF
```

### Why this matters: stateless interop

The regex key map works **without knowing which layer is which type**:

```python
# Full attention keys — only exist in HF state dict for full-attention layers
r"^model\.layers\.([0-9]+)\.self_attn\.q_proj\."  → r"decoder.layers.\1.self_attn.q_proj."
# Linear attention keys — only exist for linear-attention layers
r"^model\.layers\.([0-9]+)\.linear_attn\.in_proj_qkv\." → r"decoder.layers.\1.linear_attn.in_proj_qkv."
```

The HF prefix (`self_attn` vs `linear_attn`) naturally encodes the layer type. No config lookup needed during conversion.

### What would happen with a generic name

If we'd used `token_mixer` for both layer types, the converter would need:
1. Config access during conversion — to know which layer indices are full vs linear
2. Index-dependent key mapping — different regex per layer index
3. A custom converter function that iterates layers instead of simple regex

### Tradeoff: None module slots

Each layer has two optional attributes, one always `None`:
```python
# Full attention layer state_dict: has self_attn.*, no linear_attn.*
# Linear attention layer state_dict: has linear_attn.*, no self_attn.*
```

`register_module("linear_attn", None)` ensures `None` slots don't appear in `state_dict()` — PyTorch skips `None` submodules. The state dict naturally contains only the active module's parameters.

---

## Decision 6: PyTorch Fallback Kernels First

### What we decided

All three GatedDeltaNet kernels are pure PyTorch, ported from HF's fallbacks:

| Function | HF Source | Purpose |
|----------|-----------|---------|
| `torch_causal_conv1d_update` | lines 299-314 | Single-step conv for decode |
| `torch_chunk_gated_delta_rule` | lines 323-400 | Chunked prefill |
| `torch_recurrent_gated_delta_rule` | lines 403-442 | Step-by-step decode |

No dependency on `causal_conv1d` or `fla` packages.

### Why fallbacks first

1. **Tests run everywhere** — CPU-only CI, dev laptops, clean environments.
2. **Correctness verifiable** — direct port of HF's reference implementation.
3. **No dependency risk** — `causal_conv1d` and `fla` are external packages.

### Performance characteristics

| Kernel | Complexity | When used | Production-viable? |
|--------|-----------|-----------|-------------------|
| `torch_causal_conv1d_update` | O(1) per step | Decode (S=1) | ✅ Yes |
| `torch_recurrent_gated_delta_rule` | O(S) per head | Decode (S=1) | ✅ Yes |
| `torch_chunk_gated_delta_rule` | O(S/C × C²) per head | Prefill | ❌ No |

The chunked prefill has a Python for-loop iterating `chunk_size` times per chunk. For `seq_len=8192, chunk_size=64`: 128 chunks × 64 iterations = 8,192 loop steps with matrix ops each. This is 100-1000× slower than `fla`'s Triton kernel which fuses everything into one GPU launch.

### Future fast path pattern (following HF)

```python
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    _has_causal_conv1d = True
except ImportError:
    _has_causal_conv1d = False

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    _has_fla = True
except ImportError:
    _has_fla = False

# In GatedDeltaNet.__init__:
self._chunk_fn = chunk_gated_delta_rule if _has_fla else torch_chunk_gated_delta_rule
```

Tracked for Phase 5.
