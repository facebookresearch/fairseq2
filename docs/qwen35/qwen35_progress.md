# Qwen 3.5 Implementation Progress

> Tracks what has been implemented, what's pending, and key decisions.

## Phase 1: Core Modules ✅ COMPLETE

### 1.1 `gated_delta_net.py` ✅

- `GatedDeltaNet(nn.Module)` — full linear attention module with 5 projections, Conv1d, gating params
- `GatedDeltaNetState(IncrementalState)` — conv_state + recurrent_state, `reorder()` for beam search
- `RMSNormGated(nn.Module)` — `RMSNorm(x) * silu(gate)`
- 3 PyTorch fallback kernels: `torch_causal_conv1d_update`, `torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`

### 1.2 `attention.py` ✅

- `Qwen35Attention(MultiheadAttention)` — doubled Q proj, output gating, partial RoPE, QK-norm

### 1.3 `decoder_layer.py` ✅

- `Qwen35DecoderLayer(TransformerLMDecoderLayer)` — hybrid dispatch: full attention OR linear attention per layer
- Attribute names `self_attn` / `linear_attn` match HF for stateless interop

---

## Phase 2: Dense Model Integration ✅ COMPLETE

### 2.1 `config.py` ✅

- `Qwen35Config` with all fields matching HF defaults
- `__post_init__` auto-generates `layer_types` from `full_attention_interval`
- `register_qwen35_configs()` with all dense archs (0.8B, 2B, 9B, 27B)

### 2.2 `factory.py` ✅

- `Qwen35Factory` with all `create_*` methods
- `create_position_encoder()` — `ReferenceRotaryEncoder(encoding_dim=64)` for partial RoPE

### 2.3 `interop.py` ✅

- `_QWEN35_HG_KEY_MAP` — 25 regex rules
- `convert_qwen35_state_dict()` — key map + `weight += 1.0` for `(1+w)` RMSNorm conversion
- `_QWEN35_RMSNORM_KEYS` — includes q_norm/k_norm (critical for parity)

### 2.4 `hub.py`, `__init__.py`, `composition/models.py` ✅

- Hub accessors, exports, family registration all complete

---

## Phase 3: Component Tests ✅ COMPLETE (29 tests)

| File | Tests | Coverage |
|------|-------|----------|
| `test_gated_delta_net.py` | 6 | Forward shape, incremental decode, chunked vs recurrent, state reorder, RMSNormGated |
| `test_qwen35_attention.py` | 6 | Forward shape, output gating, partial RoPE, GQA, QK-norm, incremental KV cache |
| `test_qwen35_decoder_layer.py` | 5 | Full/linear attention forward, invalid type, model creation, hybrid layers |
| `test_qwen35_interop.py` | 4 | Key round-trip, RMSNorm conversion, GDN norm exclusion, layer types |
| `test_qwen35_moe.py` | 8 | Router shapes/weights/logits, expert shape/weights, MoeBlock shape/shared/drop-in |

---

## Phase 4: MoE Support ✅ COMPLETE

### 4.1 `moe.py` ✅

- `Qwen35TopKRouter` — softmax → top-k → renormalize, returns raw pre-softmax logits
- `Qwen35Experts` — 3D parameter experts with per-expert SiLU-gated MLP
- `Qwen35MoeBlock(FeedForwardNetwork)` — drop-in FFN replacement with router + experts + shared expert

### 4.2 Config + Factory + Interop ✅

- `Qwen35MoeConfig(Qwen35Config)` with MoE fields (E=256, K=8, I=512)
- `Qwen35MoeFactory(Qwen35Factory)` overrides `create_ffn()` → `Qwen35MoeBlock`
- `_QWEN35_MOE_HG_KEY_MAP` with MoE-specific key maps

---

## Phase 4.5: Bug Fixes & HF Parity ✅ COMPLETE

### Bugs Fixed

| File | Bug | Fix |
|------|-----|-----|
| `attention.py` | SDPA call used wrong kwargs, missing layout args, no tuple unpack | `attn_output, _ = self.sdpa(q, seqs_layout, k, keys_layout, v, bias_cache)` |
| `attention.py` | `super().__init__(model_dim)` — base takes no args | `super().__init__()` |
| `attention.py` | `repeat_interleave` imported from wrong module | `from fairseq2.ops import repeat_interleave` |
| `moe.py` | Router returned post-softmax as "logits" | Separate `router_probs` variable; return raw `router_logits` |
| `interop.py` | **q_norm/k_norm weights NOT converted with `+= 1.0`** | Added to `_QWEN35_RMSNORM_KEYS` |

### Test Fixes

| File | Issue | Fix |
|------|-------|-----|
| `test_qwen35_attention.py` | Incremental test used `IdentityBias` (bidirectional vs causal mismatch) | Use `CausalAttentionBias` |
| `test_qwen35_moe.py` | Test asserted router logits sum to 1 (no longer true) | Assert logits do NOT sum to 1, weights DO |
| `test_qwen35_interop.py` | `layer_types` not regenerated after `num_layers` change | Reset `layer_types=None` before `__post_init__()` |

### HF Parity Test ✅ PASS

**File:** `tests/parity/test_qwen35_hf_parity.py` — Loads `Qwen/Qwen3.5-0.8B`, converts state dict, asserts logit closeness.

```
Full-seq logit max  abs diff: 7.63e-06 < 1e-04   ✅ PASS
Cosine similarity:            1.00000048
Top-1 token:                  ' Paris' == ' Paris'
Top-5 tokens:                 [Paris, the, located, :, \n] == [Paris, the, located, :, \n]
```

**Root cause:** `q_norm.weight` and `k_norm.weight` were missing from `_QWEN35_RMSNORM_KEYS`. HF's `Qwen3_5RMSNorm` uses the `(1+weight)` formula for ALL norms including QK-norm. Without conversion, `q_norm` computed `norm(x) * 0.43` instead of `norm(x) * 1.43` — a 3.3× error that compounded through all 6 full-attention layers.

**Investigation documented in:** `docs/qwen35/qwen35_parity_investigation.md`

---

## Phase 5: Model Configs & Asset Cards ✅ COMPLETE

### Registered Arch Configs

Configs extracted from local HF checkpoints at `/checkpoint/smallomnillm/shared/models/`.

#### Dense Models (`qwen3_5` family)

| Arch | hidden | layers | heads | kv | head_dim | ffn | tied | lkh | lvh |
|------|--------|--------|-------|----|----------|-----|------|-----|-----|
| `qwen35_0.8b` | 1024 | 24 | 8 | 2 | 256 | 3584 | ✓ | 16 | 16 |
| `qwen35_2b` | 2048 | 24 | 8 | 2 | 256 | 6144 | ✓ | 16 | 16 |
| `qwen35_9b` | 4096 | 32 | 16 | 4 | 256 | 12288 | ✗ | 16 | 32 |
| `qwen35_27b` | 5120 | 64 | 24 | 4 | 256 | 17408 | ✗ | 16 | 48 |

Common: `vocab=248320, max_pos=262144, head_dim=256, theta=10M, partial_rotary=0.25, fai=4, lkd=128, lvd=128`

Note: Base variants (2B-Base, 9B-Base) share the same arch config as their instruct counterparts.

#### MoE Models (`qwen3_5_moe` family)

| Arch | hidden | layers | heads | kv | head_dim | E | K | I | S |
|------|--------|--------|-------|----|----------|---|---|---|---|
| `qwen35_moe_35b_a3b` | 2048 | 40 | 16 | 2 | 256 | 256 | 8 | 512 | 512 |

Note: 35B-A3B-Base shares the same arch config.

### Asset Cards

**File:** `src/fairseq2/assets/cards/models/qwen35.yaml`

8 entries: `qwen35_0.8b`, `qwen35_2b`, `qwen35_2b_base`, `qwen35_9b`, `qwen35_9b_base`, `qwen35_27b`, `qwen35_moe_35b_a3b`, `qwen35_moe_35b_a3b_base`

---

## Phase 6: Integration Polish — PENDING

- [ ] HuggingFaceConverter for HF export (reverse `weight -= 1.0`)
- [ ] MoE hub accessors (`get_qwen35_moe_model_hub`, `get_qwen35_moe_tokenizer_hub`)
- [ ] Sharder/FSDP specs (`GatedDeltaNetSharder` for TP)
- [ ] Fast-path kernels (`causal_conv1d`, `fla`)
- [ ] Documentation updates

---

## Git Commit History

```
759a5c26 [qwen3.5] Add HF parity test and investigation docs
ae0bd858 [qwen3.5] Fix interop: add q_norm/k_norm to RMSNorm weight conversion
c5eb606a [qwen3.5] Add Phase 5 TODOs for HuggingFaceConverter registration
0eb14264 [qwen3.5] Add Qwen3.5-0.8B arch config and asset cards
3b00bae1 [qwen3.5] Fix MoE router to return raw pre-softmax logits
119bd212 [qwen3.5] Fix critical bugs in Qwen35Attention
```
