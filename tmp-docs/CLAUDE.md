# Gemma3n to fairseq2 Port - Project Tracker

## Project Overview

**Goal**: Port Google DeepMind's Gemma3n multimodal language model from HuggingFace transformers to fairseq2 with complete parity on inference and SFT training.

**Model**: Gemma3n (E2B variant - 2B effective parameters)
**Source**: HuggingFace transformers 5.1.0 (`transformers.models.gemma3n`)
**Target**: fairseq2 v0.7.0

## Current Stage

**Phase**: ✅ **TEXT-ONLY COMPLETE** (2026-02-12)

**Status**:
- ✅ Text inference parity: 100% token agreement
- ✅ Asset system integration: All 4 variants (E2B/E4B base/instruct) loadable
- ✅ Tokenizer support: HF wrapper with chat templates

**Next Steps**:
- **Audio Tower Implementation** (Phase 2a)
  - USM-based audio encoder integration
  - Audio preprocessing pipeline
  - Audio+Text fusion mechanism
  - Multimodal inference parity
- Vision tower (Phase 2b)
- Training recipe (Phase 3) - deferred until multimodal complete

## Key Architecture Findings

### Gemma3n Overview
- **Modalities**: Text + Images (MobileNet-V5) + Audio (USM encoder, weights not released)
- **Architecture**: MatFormer (nested transformer) - E4B contains E2B as submodel
- **Key Innovations**:
  - Per-Layer Embeddings (PLE) - CPU/storage cached parameters
  - LAuReL (Learned Augmented Residual) - low-rank residual connections
  - AltUp (Alternating Updates) - selective position processing
  - 4:1 local:global attention ratio
  - Dual RoPE frequencies (10K local, 1M global)
  - Activation sparsity in FFN
  - KV cache sharing (last 15 layers)

### Model Variants
- **E2B**: 5B total params (2B on accelerator, 3B PLE in CPU)
- **E4B**: 8B total params (4B on accelerator, 4B PLE in CPU)

### HuggingFace Implementation (transformers 5.1.0)
- Location: `/home/aerben/repos/fairseq2/.venv/lib/python3.10/site-packages/transformers/models/gemma3n/`
- Key files:
  - `configuration_gemma3n.py` - Model configs
  - `modeling_gemma3n.py` - Main implementation
  - `modular_gemma3n.py` - Modular source
  - `feature_extraction_gemma3n.py` - Audio features
  - `processing_gemma3n.py` - Multimodal processing

### E4B Default Configuration
```python
hidden_size: 2048
num_hidden_layers: 35
num_attention_heads: 8
num_key_value_heads: 2  # GQA
head_dim: 256
intermediate_size: 16384
sliding_window: 512
laurel_rank: 64
num_kv_shared_layers: 15
rope_theta: {"global": 1_000_000, "local": 10_000}
vocab_size: 262_400
max_position_embeddings: 32_768
```

## Implementation Scope

### Phase 1: Text-Only Infrastructure ✅ COMPLETE
- ✅ Text-only inference with deterministic outputs
- ✅ Layer-by-layer parity validation
- ✅ 100% token prediction agreement with HuggingFace
- ✅ Both fp32 and bf16 precision support
- ✅ Checkpoint conversion from HuggingFace format
- ✅ Asset system integration (E2B/E4B variants)
- ✅ Tokenizer with chat template support

### Phase 2a: Audio Tower Integration 🚧 NEXT
- USM-based audio encoder
- Audio feature extraction pipeline
- Audio embedding integration with decoder
- Audio+Text multimodal inference
- Parity validation with HF audio inputs

### Phase 2b: Vision Tower Integration (FUTURE)
- MobileNet-V5 vision encoder
- Image feature extraction pipeline
- Vision embedding integration with decoder
- Image+Text multimodal inference
- Parity validation with HF vision inputs

### Phase 3: Training Recipe (DEFERRED)
- SFT training recipe
- Multimodal dataset loading
- Training parity validation
- Distributed training support (FSDP)

## Implemented Components

### Core Architecture ✅
1. **Gemma3nConfig** (`config.py`)
   - All config parameters including KV sharing, sparsity, AltUp
   - Dual RoPE theta configuration

2. **Gemma3nFrontend** (`frontend.py`)
   - Text embeddings with scaling
   - Per-Layer Embeddings (PLE) support
   - Embedding projection to model_dim

3. **Gemma3nDecoder** (`decoder.py`)
   - 4D tensor stacking/unstacking for AltUp
   - Per-layer embedding injection
   - KV projection sharing coordination

4. **Gemma3nDecoderLayer** (`decoder_layer.py`)
   - AltUp predict/correct integration
   - LAuReL (Learned Augmented Residual)
   - QK normalization
   - Local/Global attention switching
   - KV projection sharing (SOURCE/CONSUMER roles)

5. **Gemma3nAltUp** (`altup.py`)
   - Predict step: 4D→4D predictions
   - Correct step: error-based correction
   - Modality routing
   - Coefficient clipping

6. **KV Projection Sharing** (`kv_projection.py`)
   - Type-safe slot-based sharing mechanism
   - SOURCE layers (18=local, 19=global) store K/V
   - CONSUMER layers (20-29) reuse projections

### Modified Shared Components ✅
1. **MultiheadAttention** (`multihead_attention.py`)
   - `pre_computed_kv` parameter for KV sharing
   - Dual RoPE theta support
   - Sliding window attention

2. **Feed-Forward Networks** (`ffn.py`)
   - AltUpFeedForwardNetwork (for local layers)
   - GLUFeedForwardNetwork (for global layers)
   - Gaussian top-k activation sparsity

3. **SDPA Backends** (`sdpa/torch.py`, `sdpa/naive.py`)
   - Attention softcapping with tanh
   - Scale parameter override

### Checkpoint Conversion ✅
- **interop.py**: HuggingFace → fairseq2 checkpoint converter
  - Handles all weight mappings
  - PLE embeddings
  - AltUp projections
  - KV sharing layer configurations

### Testing ✅
- **Unit Tests**:
  - `test_altup.py` - AltUp predict/correct mechanism
  - `test_kv_sharing.py` - KV projection sharing registry
  - `test_activation_sparsity.py` - Gaussian top-k sparsification

- **Parity Validation**:
  - `scripts/gemma3n_validation/test_parity.py` - Full model parity test
  - `scripts/gemma3n_validation/PARITY_STATUS.md` - Detailed debugging notes

## Parity Testing Strategy

### ✅ Completed Approach

1. **Reference Implementation** (HuggingFace Transformers)
   - Used official `transformers.models.gemma3n` implementation
   - TorchSDPA backend for consistency
   - Fixed random seeds for deterministic outputs

2. **fairseq2 Implementation**
   - Reused existing primitives (RMSNorm, RoPE, MultiheadAttention, GLU FFN)
   - Implemented new components (LAuReL, AltUp, PLE, KV sharing)
   - Modified shared components (SDPA backends, attention, FFN)

3. **Validation**
   - Layer-by-layer hidden state comparison
   - Final logits comparison (token-by-token)
   - 100% token prediction agreement achieved
   - Max abs diff: 1.39e-04, Max rel diff: 2.67e-03

### Critical Configuration Details

**Layer Configuration** (30 layers total):
- Layers 0-3: Local (sliding window, AltUp FFN, sparsity=0.95)
- Layer 4: Global (full causal, GLU FFN, sparsity=0.95)
- Layers 5-8: Local (sliding window, AltUp FFN, sparsity=0.95)
- Layer 9: Global (full causal, GLU FFN, sparsity=0.95)
- Layers 10-17: Local/Global alternating (sparsity=0.0)
- **Layer 18**: Local SOURCE (stores K/V for layers 20,22,23,25-28)
- **Layer 19**: Global SOURCE (stores K/V for layers 21,24,29)
- Layers 20-29: CONSUMER layers (reuse K/V from 18 or 19)

**Key Parameters**:
- `num_kv_shared_layers = 10` (NOT 15 - critical!)
- `activation_sparsity = 0.95` for first 10 layers only
- `altup_num_inputs = 4`, `altup_coef_clip = 120.0`
- `laurel_rank = 64`
- `final_logit_soft_cap = 30.0`
- Dual RoPE: local=10,000, global=1,000,000

## Key Debugging Insights

### Critical Bugs Found
1. **KV Sharing Layer Count**: HF uses `num_kv_shared_layers=10`, NOT 15 as in paper
   - Impact: 675,000x divergence reduction (61.9 → 9.16e-05)
   - Lesson: Always verify config against actual checkpoint, not just papers

2. **Activation Sparsity Scope**: Required for BOTH local AND global layers
   - Impact: First 10 layers needed sparsity=0.95 regardless of type
   - Lesson: Debug configs may not match production configs

3. **HF Parameter Names**: `past_key_values` (plural) not `past_key_value` (singular)
   - Impact: KV cache wasn't being used at all in testing
   - Lesson: API surface matters - check exact parameter names

### Implementation Patterns That Worked
- **Slot-based KV sharing**: Type-safe dict slots better than registry pattern
- **4D stacking in decoder**: Cleaner than per-layer 3D→4D conversions
- **Enum-based layer roles**: SOURCE/CONSUMER/NONE prevents misconfiguration
- **Layer-by-layer validation**: Essential for catching divergence early

### What Didn't Match the Paper
- KV sharing: 10 layers not 15
- Sparsity: Applied to global layers too, not just local
- Source layers: 18/19 not arbitrary positions

**Lesson**: Papers describe idealized architectures; released models have pragmatic tweaks.

## Project Structure in fairseq2

```
/home/aerben/repos/fairseq2/
├── src/fairseq2/models/gemma3n/
│   ├── __init__.py
│   ├── config.py              ✅ Gemma3nConfig dataclass
│   ├── factory.py             ✅ Gemma3nFactory + builder
│   ├── decoder.py             ✅ Gemma3nDecoder with 4D AltUp
│   ├── decoder_layer.py       ✅ Gemma3nDecoderLayer with LAuReL
│   ├── frontend.py            ✅ Gemma3nFrontend with PLE
│   ├── model.py               ✅ Gemma3nModel with softcapping
│   ├── altup.py               ✅ AltUp predict/correct
│   ├── kv_projection.py       ✅ KV sharing enums
│   └── interop.py             ✅ HF checkpoint conversion
├── src/fairseq2/models/transformer/
│   ├── multihead_attention.py ✅ Modified for KV sharing
│   ├── ffn.py                 ✅ Added AltUpFeedForwardNetwork
│   └── sdpa/
│       ├── torch.py           ✅ Softcapping support
│       └── naive.py           ✅ Softcapping support
├── tests/unit/models/gemma3n/
│   ├── test_altup.py          ✅ AltUp unit tests
│   ├── test_kv_sharing.py     ✅ KV sharing tests
│   └── test_activation_sparsity.py ✅ Sparsity tests
├── scripts/gemma3n_validation/
│   ├── test_parity.py         ✅ Full parity validation
│   ├── PARITY_STATUS.md       ✅ Debugging documentation
│   └── [100+ debug scripts]   ✅ Layer-by-layer validation
├── recipes/gemma3n/           🚧 TODO: Training recipe
│   └── sft/
│       ├── __main__.py
│       ├── recipe.py
│       ├── config.py
│       └── dataset.py
└── tmp-docs/
    ├── CLAUDE.md              ✅ This file
    ├── plan-overview.md       ✅ Implementation plan
    ├── plan-phase-*.md        ✅ Phase details
    ├── gemma-architecture-reference.md
    └── gemma-paper.pdf
```

## Testing Datasets

**For SFT Training Parity:**
- **Option 1**: Code-generated synthetic dataset (100 examples, "A->B" format)
- **Option 2**: GSM8K tiny subset (fairseq2 already has `hg://facebook/fairseq2-lm-gsm8k`)
- **Option 3**: TinyStories subset (simple, widely used)

**Decision**: TBD after user input

## Open Questions

1. ✅ Is Gemma3n text-only or multimodal? → **Multimodal (Text+Image+Audio)**
2. ✅ Which model size for initial development? → **E2B (2B effective)**
3. ✅ Which precision for parity tests? → **Both fp32 and bf16**
4. ✅ Which SDPA backend? → **TorchSDPA**
5. ✅ KV sharing configuration? → **10 layers (NOT 15), sources at 18/19**
6. ✅ Activation sparsity? → **0.95 for first 10 layers only**
7. ✅ Audio/Vision weights released? → **YES, present in HF checkpoints**
8. 🚧 USM audio encoder architecture details? → **TBD (Phase 2a)**
9. 🚧 MobileNet-V5 vision encoder details? → **TBD (Phase 2b)**
10. 🚧 Training dataset for parity? → **Deferred to Phase 3**
11. ❓ PLE offloading to CPU/storage? → **Deferred (not needed for inference)**
12. ❓ MatFormer E4B nesting? → **Deferred (E2B only for now)**

## References

- Gemma 3 Technical Report: https://arxiv.org/abs/2503.19786
- MatFormer: https://arxiv.org/abs/2310.07707
- LAuReL: https://arxiv.org/abs/2411.07501
- AltUp: https://arxiv.org/abs/2301.13310
- HuggingFace Docs: https://huggingface.co/docs/transformers/main/en/model_doc/gemma3n
- Reverse Engineering: https://github.com/antimatter15/reverse-engineering-gemma-3n

## Progress Tracking

### Phase 1: Text-Only Infrastructure ✅ COMPLETE
- [x] Obtain Gemma3n architecture documentation
- [x] Verify transformers installation and Gemma3n access
- [x] Understand fairseq2 model architecture patterns
- [x] Understand fairseq2 testing infrastructure
- [x] Create comprehensive implementation plan
- [x] Implement Gemma3nConfig with all parameters
- [x] Implement Gemma3nFrontend with PLE support
- [x] Implement Gemma3nAltUp (predict/correct mechanism)
- [x] Implement Gemma3nDecoderLayer with LAuReL + QK norm
- [x] Implement Gemma3nDecoder with 4D stacking
- [x] Implement KV projection sharing mechanism
- [x] Add activation sparsity to FFN layers
- [x] Add attention softcapping to SDPA
- [x] Implement checkpoint conversion (HF → FS2)
- [x] Create unit tests (AltUp, KV sharing, sparsity)
- [x] Debug layer-by-layer parity
- [x] Achieve 100% token prediction agreement
- [x] Clean up code (remove narration comments)
- [x] Asset system integration
  - [x] Study asset registration pattern (LLaMA/Mistral)
  - [x] Identify all Gemma3n variants from HuggingFace
  - [x] Create variant discovery script
  - [x] Register E2B/E4B configs
  - [x] Create hub.py with model/tokenizer accessors
  - [x] Create asset cards (gemma3n.yaml)
  - [x] Add multimodal checkpoint filtering
  - [x] Test loading via load_model() API
  - [x] Implement custom Gemma3nTokenizer
  - [x] Register tokenizer family
  - [x] Add chat template support
  - [x] Verify tokenization parity with HuggingFace

### Phase 2a: Audio Tower Integration 🚧 IN PROGRESS
- [x] Study HuggingFace audio implementation
  - [x] Understand USM encoder architecture
  - [x] Identify audio preprocessing requirements
  - [x] Map audio state dict keys to fairseq2
  - [x] Document reusable primitives (conformer, FFN, attention, etc.)
  - [x] Create architecture reference doc
- [ ] Implement audio config and primitives
  - [x] `Gemma3nAudioConfig` dataclass
  - [ ] Audio feature extraction (mel-spectrogram preprocessing)
  - [ ] Subsample convolution projection
  - [ ] Conformer components (FFN, conv, attention)
  - [ ] Multimodal embedder
- [ ] Implement audio tower integration
  - [ ] Assemble conformer encoder
  - [ ] Load audio_tower weights from checkpoint
  - [ ] Integrate with embed_audio
  - [ ] Connect to decoder via AltUp modality routing
- [ ] Test audio+text inference
  - [ ] Create audio parity validation script
  - [ ] Verify audio features match HuggingFace
  - [ ] Achieve multimodal inference parity
- [ ] Update asset cards for audio support

### Phase 2b: Vision Tower Integration 🚧 TODO
- [ ] Study HuggingFace vision implementation
  - [ ] Understand MobileNet-V5 architecture
  - [ ] Identify image preprocessing requirements
  - [ ] Map vision state dict keys to fairseq2
- [ ] Implement vision feature extraction
  - [ ] Image preprocessing pipeline
  - [ ] Feature extraction to match HF
- [ ] Implement vision tower integration
  - [ ] Load vision_tower weights from checkpoint
  - [ ] Integrate with embed_vision
  - [ ] Connect to decoder via AltUp modality routing
- [ ] Test image+text inference
  - [ ] Create vision parity validation script
  - [ ] Verify vision features match HuggingFace
  - [ ] Achieve multimodal inference parity
- [ ] Update asset cards for vision support

### Phase 3: Training Recipe 🚧 DEFERRED
- [ ] Design SFT recipe structure
- [ ] Implement multimodal dataset loading/preprocessing
- [ ] Implement training loop
- [ ] Add distributed training support (FSDP)
- [ ] Create reference HF training baseline
- [ ] Validate training parity (loss curves)
- [ ] Test on multi-GPU setup

### Phase 4: Production Readiness 🚧 TODO
- [ ] Add comprehensive documentation
- [ ] Add integration tests
- [ ] Model hub registration
- [ ] Performance benchmarking
- [ ] Memory profiling
- [ ] PR preparation and review

---

**Last Updated**: 2026-02-12 (Text-only complete, audio tower next)
**Python Environment**: `/home/aerben/repos/fairseq2/.venv/bin/python3`
