# Gemma3n to fairseq2 Port - Implementation Plan Overview

## Goal
Port Gemma3n E2B (2B effective parameters) from HuggingFace transformers to fairseq2 with complete inference and SFT training parity.

## Strategy
**Inference first, then training**
- Start with text-only (defer vision/audio to future)
- Full architecture including PLE, LAuReL, AltUp, MatFormer
- Test both fp32 and bf16 precision
- Use code-generated synthetic dataset (100 examples) for SFT

## Git Workflow and Code Quality

### Commit Guidelines
**CRITICAL**: Each feature must be a separate commit with tests.

- **Small commits**: Keep changes <1k LOC where possible
- **Atomic commits**: One logical change per commit
- **Test with code**: Each commit must include tests for the feature
- **Commit message format**:
  ```
  [gemma3n] Add DualRotaryEncoder for LAuReL

  - Implements dual-theta RoPE (10K + 100K frequencies)
  - Splits head dimension for standard + long-range encoding
  - Adds unit tests for frequency verification
  ```

### Code Quality Standards
**Run after each phase milestone:**

1. **`/unslop-code`** - Detect and remove AI code slop:
   - Redundant comments and tutorial narration
   - Vacuous/tautological tests
   - Over-abstraction and unnecessary complexity
   - Run before committing phase completion

2. **`/better-engineering`** - Enforce FAIR engineering standards:
   - Comprehensive type hints (Python 3.10+ syntax)
   - Sphinx/reST docstrings for all public APIs
   - Behavior-focused tests (not just coverage)
   - Clean, maintainable code hygiene
   - Run before phase review

### Commit Sequence Example (Phase 2)
```
1. [gemma3n] Add DualRotaryEncoder implementation (~200 LOC)
2. [gemma3n] Add SoftCappedSDPA wrapper (~150 LOC)
3. [gemma3n] Add AltUpFeedForwardNetwork (~200 LOC)
4. [gemma3n] Add Gemma3nDecoderLayer (~300 LOC)
5. [gemma3n] Add component tests for Phase 2 (~400 LOC)
6. [gemma3n] Run /unslop-code and clean up Phase 2
```

**Total Phase 2**: ~6 commits, each <500 LOC (except tests)

## Implementation Phases

### Phase 1: Core Architecture Foundation
**Duration**: 3-4 days
**Files**: `plan-phase-1.md`

Create basic fairseq2 module structure, config, and identify reusable components.

**Deliverables:**
- Module structure (`src/fairseq2/models/gemma3n/`)
- `Gemma3nConfig` dataclass
- List of reusable vs new components

---

### Phase 2: Component Implementation
**Duration**: 2-3 days
**Files**: `plan-phase-2.md`

Implement core architectural components unique to Gemma3n.

**Deliverables:**
- `DualRotaryEncoder` (LAuReL dual-theta RoPE)
- `SoftCappedSDPA` (attention logit soft-capping)
- `AltUpFeedForwardNetwork` (GELU gating)
- `Gemma3nDecoderLayer` (local vs global)
- `Gemma3nFactory` (model builder)
- Checkpoint conversion (`interop.py`)

---

### Phase 3: Inference Parity Testing
**Duration**: 2 days
**Files**: `plan-phase-3.md`

Verify fairseq2 implementation produces identical outputs to HuggingFace.

**Deliverables:**
- Test infrastructure (`tests/integration/models/gemma3n/`)
- Component tests (RoPE, attention, FFN)
- Full model parity test (fp32 + bf16)
- Incremental decoding test

---

### Phase 4: Training Integration
**Duration**: 2 days
**Files**: `plan-phase-4.md`

Enable SFT training with fairseq2 recipes.

**Deliverables:**
- Synthetic dataset generator (100 examples)
- SFT recipe (`recipes/gemma3n/sft/`)
- Training parity test (loss, gradients)

---

### Phase 5: Advanced Features
**Duration**: 1-2 days
**Files**: `plan-phase-5.md`

Implement PLE CPU offloading, MatFormer slicing, KV cache sharing, tensor parallelism.

**Deliverables:**
- `PLEModule` with CPU offloading
- MatFormer slicing logic
- KV cache sharing mechanism
- Tensor parallelism specs (`sharder.py`)

---

## Total Timeline
**10-13 days** for full implementation

## Key Architecture Features

### Gemma3n Innovations
1. **MatFormer** - Nested transformers (E4B contains E2B)
2. **PLE** - Per-Layer Embeddings (3B params CPU-cached)
3. **LAuReL** - Dual-theta RoPE (10K + 100K)
4. **AltUp** - Alternating up-projection with GELU
5. **Local/Global Attention** - 4:1 ratio (28 local + 7 global layers)
6. **KV Cache Sharing** - Global layers share cache
7. **Soft-Capping** - Attention logit capping

### Model Specs (E2B)
```
Hidden size: 2048
Layers: 35 (28 local + 7 global)
Attention heads: 16
KV heads: 8 (GQA)
FFN inner dim: 10752
Vocab size: 256128
Max seq len: 8192
```

## Critical Files

### Implementation
- `src/fairseq2/models/gemma3n/config.py`
- `src/fairseq2/models/gemma3n/factory.py`
- `src/fairseq2/models/gemma3n/interop.py`
- `src/fairseq2/nn/position_encoder.py` (extend)
- `src/fairseq2/models/transformer/ffn.py` (extend)

### Testing
- `tests/integration/models/gemma3n/test_inference_parity.py`
- `tests/integration/models/gemma3n/test_components.py`

### Training
- `recipes/gemma3n/sft/recipe.py`
- `recipes/gemma3n/sft/config.py`

## References
- Architecture doc: `tmp-docs/gemma-architecture-reference.md`
- HF implementation: `.venv/lib/python3.10/site-packages/transformers/models/gemma3n/`
- Project tracker: `tmp-docs/CLAUDE.md`

## Next Step
See `plan-phase-1.md` to begin implementation.
