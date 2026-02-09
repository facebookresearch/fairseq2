# Phase 1: Core Architecture Foundation

**Duration**: 3-4 days
**Goal**: Set up fairseq2 module structure and identify reusable components

---

## Step 1.1: Create Module Structure

Create the following directory structure:

```
src/fairseq2/models/gemma3n/
├── __init__.py          # Public API exports
├── config.py            # Gemma3nConfig dataclass
├── factory.py           # Gemma3nFactory (main model builder)
├── checkpoint.py        # HuggingFace checkpoint loading
├── interop.py           # State dict conversion utilities
├── hub.py               # Model hub integration
├── tokenizer.py         # Tokenizer wrapper (reuse Gemma)
└── sharder.py           # Tensor parallelism specs (Phase 5)
```

---

## Step 1.2: Define Gemma3nConfig

**File**: `src/fairseq2/models/gemma3n/config.py`

```python
from dataclasses import dataclass
from typing import Literal, Optional

from fairseq2.models.transformer import TransformerConfig
from fairseq2.typing import DataType, Device

@dataclass(kw_only=True)
class Gemma3nConfig(TransformerConfig):
    """Configuration for Gemma3n models."""

    # Model architecture
    model_dim: int = 2048
    """The dimensionality of the model."""

    num_layers: int = 35
    """The number of Transformer decoder layers."""

    num_attn_heads: int = 16
    """The number of attention heads."""

    num_key_value_heads: int = 8
    """The number of key/value heads for GQA."""

    ffn_inner_dim: int = 10752
    """The dimensionality of inner projection layers in FFN."""

    # Local/Global attention
    num_local_layers: int = 28
    """Number of local sliding window attention layers."""

    num_global_layers: int = 7
    """Number of global full attention layers."""

    sliding_window: int = 512
    """Sliding window size for local attention."""

    # Position encoding (LAuReL - dual RoPE)
    rope_theta: float = 10_000.0
    """RoPE theta for local/standard frequencies."""

    dual_rope_theta: float = 100_000.0
    """RoPE theta for LAuReL long-range frequencies."""

    # Attention features
    head_dim: int = 128
    """Dimensionality of attention heads."""

    query_pre_attn_scalar: int = 128
    """Scalar for query pre-attention normalization."""

    final_logit_soft_cap: float = 30.0
    """Soft-capping value for attention logits."""

    # AltUp FFN
    altup_hidden_dim: int = 5376
    """Hidden dimension for AltUp feed-forward network."""

    altup_gate_activation: str = "gelu"
    """Activation function for AltUp gate (NOT silu)."""

    # PLE (Per-Layer Embeddings)
    ple_hidden_dim: int = 5120
    """Hidden dimension for PLE expansion."""

    ple_num_experts: int = 8
    """Number of experts in PLE module."""

    ple_top_k: int = 2
    """Number of experts to route to in PLE."""

    # LAuReL residual
    laurel_rank: int = 64
    """Rank for LAuReL low-rank residual connections."""

    # KV cache sharing
    num_kv_shared_layers: int = 15
    """Number of layers that share KV cache."""

    # Standard LM config
    vocab_size: int = 256_128
    """The size of the vocabulary."""

    max_seq_len: int = 8192
    """The maximum sequence length."""

    # Normalization
    norm_eps: float = 1e-6
    """The epsilon value for RMSNorm."""

    # MatFormer
    matformer_e2b_slice_dim: Optional[int] = None
    """Dimension slice for E2B mode (if using MatFormer)."""


def get_gemma3n_e2b_config() -> Gemma3nConfig:
    """Get configuration for Gemma3n E2B (2B effective)."""
    return Gemma3nConfig(
        model_dim=2048,
        num_layers=35,
        num_attn_heads=16,
        num_key_value_heads=8,
        ffn_inner_dim=10752,
        num_local_layers=28,
        num_global_layers=7,
        sliding_window=512,
        vocab_size=256_128,
        max_seq_len=8192,
    )
```

---

## Step 1.3: Identify Reusable vs New Components

### ✅ Reusable from fairseq2

**Can use as-is:**
- `RMSNorm` - Pre-attention and pre-FFN normalization
  - Location: `fairseq2.nn.normalization.RMSNorm`
  - Usage: Same as Gemma/Llama

**Can extend/modify:**
- `RotaryEncoder` - Extend for dual-theta RoPE
  - Location: `fairseq2.nn.position_encoder.RotaryEncoder`
  - Modification: Create `DualRotaryEncoder` subclass

- `StandardMultiheadAttention` - Modify for soft-capping
  - Location: `fairseq2.nn.transformer.multihead_attention`
  - Modification: Wrap SDPA with soft-capping layer

- `StandardTransformerLMDecoder` - Use as layer container
  - Location: `fairseq2.models.transformer.decoder`
  - Usage: Standard decoder structure

### ❌ Need New Implementation

**New components required:**

1. **DualRotaryEncoder** (`fairseq2.nn.position_encoder`)
   - Applies two RoPE frequencies (10K and 100K)
   - Concatenates results along head dimension
   - Implements LAuReL positional encoding

2. **SoftCappedSDPA** (`fairseq2.nn.transformer`)
   - Wraps TorchSDPA with logit soft-capping
   - Formula: `tanh(logits / soft_cap) * soft_cap`
   - Applied before softmax in attention

3. **AltUpFeedForwardNetwork** (`fairseq2.models.transformer.ffn`)
   - GELU gate activation (not SiLU!)
   - Different hidden dim (5376 vs 10752)
   - Used in local layers only

4. **Gemma3nDecoderLayer** (`fairseq2.models.gemma3n`)
   - Handles local vs global attention mode
   - Integrates PLE module (Phase 5)
   - Chooses AltUp vs standard FFN based on layer type

5. **PLEModule** (`fairseq2.models.gemma3n`) - Phase 5
   - Expert routing with top-k selection
   - CPU-cached parameters
   - Layer-specific embeddings

---

## Step 1.4: Component Mapping Table

| Component | HuggingFace | fairseq2 Equivalent | Status |
|-----------|-------------|---------------------|--------|
| Token embeddings | `Gemma3nTextScaledWordEmbedding` | `StandardEmbedding` | ✅ Reuse |
| RMSNorm | `Gemma3nRMSNorm` | `RMSNorm` | ✅ Reuse |
| RoPE (single) | `Gemma3nRotaryEmbedding` | `RotaryEncoder` | ✅ Reuse |
| RoPE (dual/LAuReL) | Custom logic | `DualRotaryEncoder` | ❌ New |
| Attention | `Gemma3nTextAttention` | `StandardMultiheadAttention` | 🔧 Extend |
| Soft-capping | Inline in attention | `SoftCappedSDPA` | ❌ New |
| Standard FFN | `Gemma3nTextMLP` | `GLUFeedForwardNetwork` | ✅ Reuse |
| AltUp FFN | `Gemma3nTextAltUp` | `AltUpFeedForwardNetwork` | ❌ New |
| LAuReL residual | `Gemma3nTextLaurelBlock` | Inline in layer | 🔧 Modify |
| PLE | HF custom module | `PLEModule` | ❌ New |
| Decoder layer | `Gemma3nTextDecoderLayer` | `Gemma3nDecoderLayer` | ❌ New |
| LM head | Output projection | `TiedProjection` | ✅ Reuse |

**Legend:**
- ✅ Reuse: Use existing fairseq2 component
- 🔧 Extend: Modify existing component
- ❌ New: Implement from scratch

---

## Step 1.5: Layer Structure

Gemma3n has 35 layers with a 4:1 local:global ratio.

**Layer type pattern:**
```python
def get_layer_type(layer_idx: int) -> str:
    """
    Determine if layer is 'local' or 'global'.

    Global layers: 3, 7, 11, 15, 19, 23, 27, 31, 34
    (every 4th layer starting at 3, plus final layer)
    """
    if layer_idx == 34:  # Final layer
        return "global"
    if (layer_idx + 1) % 4 == 0:  # Layers 3, 7, 11, ...
        return "global"
    return "local"
```

**Verification**: Check against HuggingFace config

---

## Step 1.6: Directory Setup Commands

```bash
# Create directory structure
mkdir -p src/fairseq2/models/gemma3n
touch src/fairseq2/models/gemma3n/__init__.py
touch src/fairseq2/models/gemma3n/config.py
touch src/fairseq2/models/gemma3n/factory.py
touch src/fairseq2/models/gemma3n/interop.py
touch src/fairseq2/models/gemma3n/checkpoint.py
touch src/fairseq2/models/gemma3n/hub.py
touch src/fairseq2/models/gemma3n/tokenizer.py

# Create test directory
mkdir -p tests/integration/models/gemma3n
touch tests/integration/models/gemma3n/__init__.py
touch tests/integration/models/gemma3n/test_components.py
touch tests/integration/models/gemma3n/test_inference_parity.py
```

---

## Commit Strategy for Phase 1

**Commit 1**: `[gemma3n] Add module structure and config`
- Create directory structure
- Add `__init__.py`, `config.py` with `Gemma3nConfig`
- Add layer type helper function
- ~150-200 LOC

**Commit 2**: `[gemma3n] Add checkpoint conversion stubs`
- Create `interop.py` with state dict conversion skeleton
- Create `checkpoint.py` with HF loader skeleton
- ~100-150 LOC

**Commit 3**: `[gemma3n] Add hub integration`
- Create `hub.py` for model registration
- Add tokenizer wrapper in `tokenizer.py`
- ~100 LOC

**Code Quality Check**:
- Run `/better-engineering` on all new files
- Ensure type hints and docstrings complete
- Commit cleanup: `[gemma3n] Phase 1 code quality improvements`

**Total**: 3-4 commits, ~400-500 LOC

---

## Deliverables for Phase 1

- [ ] Module structure created
- [ ] `Gemma3nConfig` dataclass defined
- [ ] Component reuse analysis complete
- [ ] Layer type logic implemented
- [ ] Test directory structure created
- [ ] All commits have tests
- [ ] `/better-engineering` passed

---

## Next Step
Proceed to `plan-phase-2.md` for component implementation.
