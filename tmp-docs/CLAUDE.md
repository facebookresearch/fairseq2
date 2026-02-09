# Gemma3n to fairseq2 Port - Project Tracker

## Project Overview

**Goal**: Port Google DeepMind's Gemma3n multimodal language model from HuggingFace transformers to fairseq2 with complete parity on inference and SFT training.

**Model**: Gemma3n (E2B variant - 2B effective parameters)
**Source**: HuggingFace transformers 5.1.0 (`transformers.models.gemma3n`)
**Target**: fairseq2 v0.7.0

## Current Stage

**Phase**: 🔍 **EXPLORATION** - Understanding architecture and planning implementation

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

### Phase 1: Text-Only Parity (CURRENT FOCUS)
- Text-only inference with deterministic outputs
- Simple SFT training on tiny dataset
- Both fp32 and bf16 precision

### Phase 2: Vision Integration (FUTURE)
- MobileNet-V5 vision encoder
- Image+Text fusion
- Multimodal inference parity

### Phase 3: Audio Integration (BLOCKED - weights not released)
- USM-based audio encoder
- Audio+Text fusion
- Awaiting public audio weights release

## Parity Testing Strategy

### 1. Setup Reference Tests in Transformers
- Create deterministic inference test (fixed batch, multiple precisions)
- Create simple SFT training test (fixed synthetic dataset)
- Save outputs as tensors for comparison
- Use TorchSDPA backend (match fairseq2 default)

### 2. Port to fairseq2
- Reuse existing fairseq2 primitives where possible:
  - RMSNorm (already exists)
  - RoPE (modify for dual frequencies)
  - MultiheadAttention (add local window support)
  - GLU FFN (add sparsity support)
- Implement new components:
  - LAuReL residual
  - AltUp
  - PLE (Per-Layer Embeddings)
  - KV cache sharing
  - MatFormer slicing

### 3. Verify Parity
- Layer-by-layer hidden state comparison
- Final logits comparison (token-by-token)
- Training loss curve comparison
- Run on GPU via SLURM interactive node

## Project Structure in fairseq2

```
/home/aerben/repos/fairseq2/
├── src/fairseq2/models/gemma3n/
│   ├── __init__.py
│   ├── config.py              # Gemma3nConfig dataclass
│   ├── factory.py             # Gemma3nFactory
│   ├── hub.py                 # Model hub registration
│   ├── checkpoint.py          # HF checkpoint conversion
│   ├── tokenizer.py           # Tokenizer (if needed)
│   └── interop.py             # HF interoperability
├── recipes/gemma3n/
│   └── sft/                   # SFT recipe
│       ├── __main__.py
│       ├── recipe.py
│       ├── config.py
│       └── dataset.py
├── tests/integration/models/
│   └── test_gemma3n.py        # Parity tests
└── tmp-docs/                  # THIS DIRECTORY
    ├── CLAUDE.md              # This file
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
5. ⏳ Which dataset for trivial SFT test? → **User to decide**
6. ⏳ Should we implement PLE offloading in Phase 1 or defer? → **TBD**
7. ⏳ Should we implement MatFormer nesting in Phase 1 or defer? → **TBD**

## References

- Gemma 3 Technical Report: https://arxiv.org/abs/2503.19786
- MatFormer: https://arxiv.org/abs/2310.07707
- LAuReL: https://arxiv.org/abs/2411.07501
- AltUp: https://arxiv.org/abs/2301.13310
- HuggingFace Docs: https://huggingface.co/docs/transformers/main/en/model_doc/gemma3n
- Reverse Engineering: https://github.com/antimatter15/reverse-engineering-gemma-3n

## Progress Tracking

- [x] Obtain Gemma3n architecture documentation
- [x] Verify transformers installation and Gemma3n access
- [x] Understand fairseq2 model architecture patterns
- [x] Understand fairseq2 testing infrastructure
- [x] Understand fairseq2 training recipe system
- [ ] Create comprehensive implementation plan
- [ ] Setup reference inference test in transformers
- [ ] Setup reference training test in transformers
- [ ] Implement core primitives in fairseq2
- [ ] Implement attention layers in fairseq2
- [ ] Implement decoder layer in fairseq2
- [ ] Implement full model in fairseq2
- [ ] Verify inference parity
- [ ] Create SFT recipe
- [ ] Verify training parity
- [ ] (Future) Add vision encoder
- [ ] (Future) Add audio encoder

---

**Last Updated**: 2026-02-08 (Exploration phase)
**Python Environment**: `/home/aerben/repos/fairseq2/.venv/bin/python3`
