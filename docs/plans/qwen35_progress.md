# Qwen 3.5 Implementation Progress

> Tracks what has been implemented, what's pending, and key decisions.

## Phase 1: Core Modules ✅ COMPLETE

### 1.1 `gated_delta_net.py` ✅

**File:** `src/fairseq2/models/qwen/gated_delta_net.py`

**Implemented:**
- `l2norm()` — L2 normalization helper
- `torch_causal_conv1d_update()` — PyTorch fallback for decode-step conv
- `torch_chunk_gated_delta_rule()` — PyTorch fallback for chunked prefill
- `torch_recurrent_gated_delta_rule()` — PyTorch fallback for step-by-step decode
- `GatedDeltaNetState(IncrementalState)` — conv_state + recurrent_state, with `reorder()` for beam search
- `RMSNormGated(nn.Module)` — wraps `fairseq2.nn.RMSNorm` + silu gate
- `GatedDeltaNet(nn.Module)` — full module: 4 `Linear` projections, `nn.Conv1d`, gating params, forward with state_bag

**fairseq2 APIs used:**
- `fairseq2.nn.Linear` — all 5 projections (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj)
- `fairseq2.nn.RMSNorm` — inside RMSNormGated
- `fairseq2.nn.IncrementalState` — base class for GatedDeltaNetState
- `fairseq2.nn.IncrementalStateBag` — state_bag.maybe_get_state / set_state

**Native torch (no fairseq2 wrapper exists):**
- `nn.Conv1d` — depthwise causal convolution
- `nn.Parameter` — dt_bias, A_log

### 1.2 `attention.py` ✅

**File:** `src/fairseq2/models/qwen/attention.py`

**Implemented:**
- `Qwen35Attention(MultiheadAttention)` — doubled Q proj, output gating, partial RoPE, QK-norm

**fairseq2 APIs used:**
- `fairseq2.nn.Linear` — q_proj (2x), k_proj, v_proj, output_proj
- `fairseq2.nn.LayerNorm` — q_norm, k_norm (created by factory as RMSNorm)
- `fairseq2.nn.PositionEncoder` — partial RoPE (splits at encoding_dim)
- `fairseq2.nn.BatchLayout` — sequence layout tracking
- `fairseq2.nn.IncrementalStateBag` — KV cache management
- `fairseq2.nn.functional.repeat_interleave` — GQA expansion
- `fairseq2.models.transformer.AttentionState` / `FullAttentionState` — KV cache
- `fairseq2.models.transformer.SDPA` — scaled dot-product attention
- `fairseq2.models.transformer.MultiheadAttention` — base class

### 1.3 `decoder_layer.py` ✅

**File:** `src/fairseq2/models/qwen/decoder_layer.py`

**Implemented:**
- `Qwen35DecoderLayer(TransformerLMDecoderLayer)` — hybrid dispatch: full attention OR linear attention per layer

**fairseq2 APIs used:**
- `fairseq2.models.transformer_lm.TransformerLMDecoderLayer` — base class
- `fairseq2.models.transformer.AttentionBiasCache` — attention bias
- `fairseq2.models.transformer.FeedForwardNetwork` — FFN interface
- `fairseq2.nn.LayerNorm` — pre-norm layer norms
- `fairseq2.nn.AdditiveResidualConnect` — residual connections
- `fairseq2.nn.ResidualConnect` — residual interface
- `fairseq2.nn.BatchLayout` — layout
- `fairseq2.nn.IncrementalStateBag` — state bag

**Key design:**
- Attribute names `self_attn` / `linear_attn` match HF for interop
- Pre-norm order: norm → attn/gdn → residual → norm → ffn → residual
- Registers None for unused token mixer slot (clean state_dict)

---

## Phase 2: Dense Model Integration ✅ COMPLETE

### 2.1 `config.py` ✅ — `Qwen35Config` added

- `Qwen35Config` dataclass with all 20 fields matching HF defaults
- `__post_init__` auto-generates `layer_types` from `full_attention_interval`
- `register_qwen35_configs()` with `qwen35_27b` arch
- `QWEN35_FAMILY = "qwen3_5"` constant

### 2.2 `factory.py` ✅ — `Qwen35Factory` added

- `create_qwen35_model()` top-level function
- `Qwen35Factory` with all `create_*` methods:
  - `create_position_encoder()` — `ReferenceRotaryEncoder(encoding_dim=64)` for partial RoPE
  - `create_decoder_layer()` — dispatches to `Qwen35DecoderLayer` with hybrid layer type
  - `create_gated_attention()` — `Qwen35Attention` with QK-norm, partial RoPE
  - `create_gated_delta_net()` — `GatedDeltaNet` with all config params
  - `create_ffn()` — `GLUFeedForwardNetwork`
  - `create_layer_norm()` — `RMSNorm(bias=False, eps=1e-6)`

### 2.3 `interop.py` ✅ — Qwen 3.5 key maps + weight conversion

- `_QWEN35_HG_KEY_MAP` — 25 regex rules covering full attn, linear attn, FFN, norms, embeddings
- GDN norm maps to `linear_attn.norm.inner_norm` (wrapping fairseq2 RMSNorm)
- `convert_qwen35_state_dict()` — applies key map + `weight += 1.0` for `(1+w)` RMSNorm conversion
- `_QWEN35_RMSNORM_KEYS` tuple identifies which weights to convert

### 2.4 `hub.py` ✅ — Hub accessors added

- `get_qwen35_model_hub` — `ModelHubAccessor(QWEN35_FAMILY, ...)`
- `get_qwen35_tokenizer_hub` — `TokenizerHubAccessor(QWEN35_FAMILY, ...)` (reuses QwenTokenizer)

### 2.5 `__init__.py` ✅ — Exports extended

- All Qwen 3.5 symbols exported: `QWEN35_FAMILY`, `Qwen35Config`, `Qwen35Factory`, `create_qwen35_model`, `convert_qwen35_state_dict`, hub accessors, `register_qwen35_configs`

### 2.6 `composition/models.py` ✅ — Family registered

- `register_model_family(QWEN35_FAMILY, ...)` with factory, state_dict_converter, compiler, FSDP, AC
- `register_qwen35_configs(container)` called

## Phase 3: Component Tests ✅ COMPLETE

Written by 4 parallel subagents, reviewed by Opus 4.5 code_search reviewer.

### 3.1 `test_gated_delta_net.py` ✅ (6 tests)

**File:** `tests/unit/models/qwen/test_gated_delta_net.py`

| Test | What it validates |
|------|-------------------|
| `test_forward_produces_correct_shape` | Output shape `(B, S, D)` matches input |
| `test_incremental_decode_matches_full_forward` | Prefill with state_bag matches full forward |
| `test_step_by_step_decode_matches_prefill` | **NEW** Prefill 8 tokens → decode 9th token matches full 9-token forward |
| `test_chunked_vs_recurrent_consistency` | `torch_chunk_gated_delta_rule` ≈ `torch_recurrent_gated_delta_rule` |
| `test_gated_delta_net_state_reorder` | `GatedDeltaNetState.reorder()` for beam search |
| `test_rmsnorm_gated_output` | `RMSNormGated` = `norm(x) * silu(gate)` |

### 3.2 `test_qwen35_attention.py` ✅ (6 tests)

**File:** `tests/unit/models/qwen/test_qwen35_attention.py`

| Test | What it validates |
|------|-------------------|
| `test_forward_produces_correct_shape` | Output `(B, S, model_dim)` |
| `test_output_gating_effect` | Gate produces non-trivial output |
| `test_partial_rope_applies_to_subset_of_dims` | `encoding_dim < head_dim` partial rotation |
| `test_gqa_with_fewer_kv_heads` | GQA with `num_kv_heads=2, num_heads=4` |
| `test_qk_norm_applied` | QK-norm changes output vs no-norm |
| `test_incremental_kv_cache_matches_full_forward` | **NEW** Token-by-token KV cache matches full forward |

### 3.3 `test_qwen35_interop.py` ✅ (4 tests)

**File:** `tests/unit/models/qwen/test_qwen35_interop.py`

| Test | What it validates |
|------|-------------------|
| `test_state_dict_key_round_trip` | fs2 → HF → fs2 keys identity |
| `test_rmsnorm_weight_conversion` | `weight += 1.0` for `(1+w)` norms |
| `test_gdn_norm_weight_not_converted` | GDN internal norm NOT shifted |
| `test_layer_types_are_correct` | 3 linear + 1 full pattern |

### 3.4 `test_qwen35_decoder_layer.py` ✅ (5 tests)

**File:** `tests/unit/models/qwen/test_qwen35_decoder_layer.py`

| Test | What it validates |
|------|-------------------|
| `test_full_attention_layer_forward` | Full attention decoder layer shape |
| `test_linear_attention_layer_forward` | GDN decoder layer shape |
| `test_invalid_layer_type_raises` | ValueError on bad layer_type |
| `test_create_small_model` | Factory → model → forward pass → logits shape |
| `test_model_has_hybrid_layers` | Layer types = [linear, linear, linear, full] |

**Review result:** All 4 files PASS. No blocking issues. Suggested improvements (non-blocking): add step-by-step decode test, tied_embeddings interop test, padding_mask test.

## Phase 4: MoE Support ✅ COMPLETE

Written by 2 parallel subagents (moe.py module + config/factory/interop/registration).

### 4.1 `moe.py` ✅ — MoE module

**File:** `src/fairseq2/models/qwen/moe.py`

| Class | Base | Purpose |
|-------|------|---------|
| `Qwen35TopKRouter` | `Module` | softmax → top-k → renormalize. `nn.Parameter(zeros(E, D))` |
| `Qwen35Experts` | `Module` | 3D params `gate_up_proj(E, 2I, D)` + `down_proj(E, D, I)`. Per-expert SiLU-gated MLP. |
| `Qwen35MoeBlock` | `FeedForwardNetwork` | Drop-in FFN replacement. Router + experts + `GLUFeedForwardNetwork` shared expert + sigmoid gate. |

**fairseq2 APIs:** `Linear` (shared_expert_gate), `GLUFeedForwardNetwork` (shared expert), `FeedForwardNetwork` (base class).

### 4.2 `config.py` ✅ — `Qwen35MoeConfig` added

- `QWEN35_MOE_FAMILY = "qwen3_5_moe"`
- `Qwen35MoeConfig(Qwen35Config)` with MoE fields: `num_experts=256`, `num_experts_per_tok=8`, `moe_intermediate_size=512`, etc.
- `register_qwen35_moe_configs()` with `qwen35_moe_35b_a3b` arch

### 4.3 `factory.py` ✅ — `Qwen35MoeFactory` added

- `create_qwen35_moe_model()` top-level function
- `Qwen35MoeFactory(Qwen35Factory)` overrides `create_ffn()` to return `Qwen35MoeBlock`

### 4.4 `interop.py` ✅ — MoE key maps added

- `_QWEN35_MOE_HG_KEY_MAP` extends dense key map with MoE-specific rules (router, experts, shared_expert, shared_expert_gate)
- `convert_qwen35_moe_state_dict()` with same RMSNorm conversion

### 4.5 `composition/models.py` ✅ — `"qwen3_5_moe"` registered

- Full model family registration with factory, state_dict_converter, compiler, FSDP, AC

### 4.6 `__init__.py` ✅ — MoE exports added

### 4.7 `test_qwen35_moe.py` ✅ — MoE tests (8 tests)

**File:** `tests/unit/models/qwen/test_qwen35_moe.py`

| Test | What it validates |
|------|-------------------|
| `test_forward_output_shapes` | Router returns `(T,E)`, `(T,K)`, `(T,K)` |
| `test_weights_sum_to_one` | Renormalized top-k weights sum to 1 |
| `test_logits_are_softmax` | Router logits are valid probability distribution |
| `test_forward_output_shape` (Experts) | Experts output `(T, D)` matches input |
| `test_weighted_output` | Zero weights → zero output |
| `test_forward_output_shape` (MoeBlock) | MoeBlock output `(B, S, D)` matches input |
| `test_shared_expert_contributes` | Shared expert output is non-zero |
| `test_drop_in_ffn_replacement` | MoeBlock is `isinstance(FeedForwardNetwork)` |

## Phase 5: Integration — PENDING

- [ ] Asset cards
- [ ] Sharder/FSDP specs
- [ ] Documentation

---

## Key Decisions Made

### Why `QWEN_FAMILY` and `QWEN35_FAMILY` are separate

Qwen 2.5 and Qwen 3.0 share `QWEN_FAMILY = "qwen"` because they differ only in **config values** (e.g., `qkv_proj_bias=True` vs `False`, `q_norm=False` vs `True`). The same `QwenConfig`, `QwenFactory`, `StandardMultiheadAttention`, and `StandardTransformerLMDecoderLayer` handle both — the factory just reads the config flags.

Qwen 3.5 requires a **separate family** because it introduces entirely new module types:

| Aspect | Qwen 2.5/3.0 (`"qwen"`) | Qwen 3.5 (`"qwen3_5"`) |
|--------|--------------------------|-------------------------|
| Token mixer | `StandardMultiheadAttention` (all layers) | `Qwen35Attention` OR `GatedDeltaNet` (per-layer hybrid) |
| Decoder layer | `StandardTransformerLMDecoderLayer` | `Qwen35DecoderLayer` (hybrid dispatch) |
| RoPE | Full rotation | Partial (25% of dims) |
| RMSNorm | Standard `weight * norm(x)` | `(1+weight) * norm(x)` — needs interop weight conversion |
| State dict keys | Only `self_attn.*` | Both `self_attn.*` AND `linear_attn.*` |

The **real separator** is: can one factory + one config + one converter handle both? If adding a new model variant only needs different config values → same family. If it needs different module classes, different factory logic, and different state dict conversion → separate family.

Could we merge them into one mega-family? Technically yes, but `QwenConfig` would become a 35+ field grab-bag where half the fields are irrelevant for any given model, and the factory would be a tangled mess of conditionals. Separate families keep the code clean while sharing the same `qwen/` directory for tokenizer, hub, and other common infrastructure.

1. **Same `qwen/` directory** — Qwen 3.5 code lives alongside Qwen 2.5/3.0 but registers as separate families (`"qwen3_5"`, `"qwen3_5_moe"`).

   **Alternatives considered:**
   - *3 separate directories* (`qwen/`, `qwen3_5/`, `qwen3_5_moe/`): Would duplicate `tokenizer.py` (same HF tokenizer with `<think>` tags), `hub.py` patterns, and parts of `interop.py`. Llama4 uses this approach (`llama/` vs `llama4/`), but Llama4 shares far less with Llama than Qwen 3.5 shares with Qwen 3.0.
   - *Single directory with everything in one family*: Discussed above — rejected because of config/factory incompatibility.

   **What's shared:** tokenizer, hub accessor pattern, directory. **What's separate:** config, factory, interop converter, model family registration.

2. **RMSNorm `(1+w)` handled in interop** — Weight conversion `+= 1.0` on load, reusing standard `RMSNorm`.

   **Why not create a `Qwen35RMSNorm` class?** A custom norm class would need to live in the module graph, which means every `create_layer_norm()` call in the factory would need to know whether it's Qwen 3.5 or not. The interop approach is invisible to the rest of the codebase — the factory creates standard `RMSNorm`, and the weight conversion happens once at checkpoint load time.

   **Caveat:** One-way conversion. If we need HF export (`to_hg_state_dict`) for Qwen 3.5, we'll need `weight -= 1.0` in the reverse path. Not yet implemented — track for Phase 5.

   **Which weights are converted?** Only `input_layernorm`, `post_attention_layernorm`, and final `model.norm`. The `RMSNormGated` inside GatedDeltaNet uses standard `weight=ones` — it is NOT converted. This distinction is enforced by `_QWEN35_RMSNORM_KEYS` tuple matching only the layer norm suffixes.

3. **Partial RoPE handled in attention** — `Qwen35Attention` splits q/k at `encoding_dim`, applies `pos_encoder` only to the first 64 dims, concatenates back.

   **Alternatives considered:**
   - *`PartialRotaryEncoder` wrapper class*: Would split/rotate/concat inside the encoder's `forward()`. Cleaner abstraction, but adds a new class that nothing else in fairseq2 uses, and the split point (`encoding_dim`) is already known to the attention module via `pos_encoder.encoding_dim`.
   - *Modify `ReferenceRotaryEncoder` to support partial dims*: Invasive change to a shared component used by all models. Rejected.

   **Verified safe:** `ReferenceRotaryEncoder` reads `state_bag.step_nr` but doesn't modify it, so calling it twice per forward (once for `q_rot`, once for `k_rot`) doesn't double-advance the step counter. This was confirmed by reading the encoder source code.

4. **Native torch for Conv1d and Parameter** — No fairseq2 wrappers exist.

   **FSDP impact: None.** fairseq2's FSDP uses `use_orig_params=True` and discovers ALL parameters via `named_parameters()` recursively. `nn.Conv1d.weight`, `nn.Parameter` (dt_bias, A_log) are all discovered and sharded automatically when the parent `GatedDeltaNet` module is wrapped at the decoder-layer level.

   **Tensor Parallelism impact: Limited.** fairseq2's TP sharding uses `ModuleSharder` classes that convert `Linear` → `ColumnShardedLinear`/`RowShardedLinear`. The Conv1d and bare Parameters won't be TP-sharded — they'll be replicated across TP ranks. For GatedDeltaNet this is acceptable because: (a) Conv1d is depthwise with `groups=conv_dim`, so each channel is independent and could be sharded by splitting along the channel dim, but this would need a custom `Conv1dSharder`; (b) `dt_bias` and `A_log` are tiny (`num_v_heads` = 32 floats) — replication cost is negligible. If TP sharding becomes important, a `GatedDeltaNetSharder` can be added in Phase 5.

5. **Attribute names match HF** — `self_attn` / `linear_attn` for clean interop key mapping.

   **Why this matters concretely:** The interop regex maps like `r"^model\.layers\.([0-9]+)\.linear_attn\."` → `r"decoder.layers.\1.linear_attn."` work because the fairseq2 attribute name IS `linear_attn`. If we'd used a generic name like `token_mixer` for both layer types, the converter would need to:
   1. Know which layer indices are full vs linear (requires config access during conversion).
   2. Apply different key maps per layer index.
   3. Handle the asymmetry that HF uses `self_attn` for full layers but `linear_attn` for linear layers.

   By matching HF names, the regex map is stateless — no config needed, no index-dependent logic.

   **Tradeoff:** The `Qwen35DecoderLayer` has two optional attributes (`self_attn` and `linear_attn`), one of which is always `None`. This means the state dict has entries for only one of them per layer, which is correct but slightly unusual. The `register_module("linear_attn", None)` call ensures the unused slot doesn't appear in `state_dict()`.

6. **PyTorch fallbacks first** — All three delta rule kernels ported as pure PyTorch. Fast paths (`causal_conv1d` / `fla`) can be added later.

   **Why fallbacks first:**
   - Tests run on CPU without GPU or custom CUDA kernels.
   - Correctness is verifiable against HF's own fallback implementations (our code is a direct port).
   - CI environments may not have `causal_conv1d` or `fla` installed.

   **Performance characteristics:**
   - `torch_chunk_gated_delta_rule`: O(S/C × C²) per head — the inner loop at lines 146-149 iterates `chunk_size` times per chunk. For `seq_len=8192, chunk_size=64`, that's 128 chunks × 64 iterations = 8192 steps, each doing matrix ops. Viable for testing, not for production.
   - `torch_recurrent_gated_delta_rule`: O(S) per head — one step per token. Used only during decode (S=1), so performance is fine.
   - `torch_causal_conv1d_update`: O(1) per step — single conv operation. Fine for production.

   **Future fast path pattern** (following HF):
   ```python
   if is_fast_path_available:
       from causal_conv1d import causal_conv1d_fn
       from fla.ops.gated_delta_rule import chunk_gated_delta_rule
   else:
       causal_conv1d_fn = None  # use torch fallback
   ```
   Track for Phase 5.
