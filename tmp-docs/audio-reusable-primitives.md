# Fairseq2 Reusable Primitives for Audio Tower

**DO NOT REIMPLEMENT THESE - They already exist in fairseq2!**

## Core Building Blocks

### Conformer Components
```python
# File: src/fairseq2/models/conformer/block.py
from fairseq2.models.conformer import ConformerBlock

# Pattern: FFN(0.5x) → Self-Attn → Conv → FFN(0.5x) → Final Norm
# - Macaron-style FFN wrapping with 0.5 scaling (matches our residual_weight=0.5)
# - Pre-norm configuration
# - Clean integration with MultiheadAttention and FeedForwardNetwork
# - Per-component dropout
```

```python
# File: src/fairseq2/models/conformer/convolution.py
from fairseq2.models.conformer import ConformerConvolution

# Pattern: Pointwise → GLU → Depthwise → Norm → Activation → Pointwise
# - Depthwise via groups=model_dim
# - Supports batch norm or layer norm
# - Configurable kernel size (we need kernel=5)
# - SiLU/Swish activation default
# - Causal padding support
# - **USE THIS FOR lconv1d MODULE**
```

### Feed-Forward Networks
```python
# File: src/fairseq2/models/transformer/ffn.py
from fairseq2.models.transformer import (
    StandardFeedForwardNetwork,
    GLUFeedForwardNetwork,
    AltUpFeedForwardNetwork,  # **USE THIS - already in Gemma3n**
)

# AltUpFeedForwardNetwork:
# - GELU activation (tanh approximation, matches HF)
# - Gate + inner projection pattern
# - Gaussian top-k sparsification support
# - **PERFECT FOR AUDIO TOWER FFN LAYERS**
```

### Normalization
```python
# File: src/fairseq2/nn/normalization.py
from fairseq2.nn import RMSNorm, StandardLayerNorm

# RMSNorm:
# - Optional elementwise_affine (scale parameter)
# - Optional bias parameter
# - Configurable epsilon (we use 1e-6)
# - **USE EVERYWHERE IN AUDIO TOWER**
```

### Attention Mechanisms
```python
# File: src/fairseq2/models/transformer/multihead_attention.py
from fairseq2.models.transformer import StandardMultiheadAttention

# File: src/fairseq2/models/transformer/attention_bias.py
from fairseq2.models.transformer import (
    CausalAttentionBias,
    ChunkedAttentionBias,  # For local attention
    IdentityBias,
)

# StandardMultiheadAttention:
# - Flexible SDPA backend selection
# - Pre-computed KV support
# - Attention bias support
# - **BASE FOR AUDIO ATTENTION**
```

### Relative Position Embeddings
```python
# File: src/fairseq2/models/transformer/sdpa/shaw.py
from fairseq2.models.transformer.sdpa import ShawRelativePositionSDPA

# Shaw et al. relative position embeddings:
# - Left/right clipping bounds configurable
# - Optional relative position values for keys and values
# - Xavier uniform initialization
# - **USE FOR CONFORMER ATTENTION**
```

```python
# File: src/fairseq2/models/transformer/sdpa/relative.py
from fairseq2.models.transformer.sdpa import RelativePositionSDPA

# Transformer XL style (alternative):
# - U and V bias parameters
# - Shift trick for efficiency
# - **ALTERNATIVE IF SHAW DOESN'T MATCH**
```

### Feature Extraction / Downsampling
```python
# File: src/fairseq2/models/s2t_transformer/feature_extractor.py
from fairseq2.models.s2t_transformer import Conv1dFbankSubsampler

# Multi-layer 1D conv for mel-spectrogram downsampling:
# - GLU activation between layers
# - Configurable kernel sizes and strides
# - Sequence length contraction calculation
# - **ADAPT FOR subsample_conv_projection**
```

```python
# File: src/fairseq2/models/wav2vec2/feature_extractor.py
from fairseq2.models.wav2vec2 import Wav2Vec2FeatureExtractor

# Raw waveform feature extraction:
# - Group norm or layer norm support
# - Flexible layer descriptors
# - **ALTERNATIVE FOR RAW AUDIO**
```

### Position Encoders
```python
# File: src/fairseq2/nn/position_encoder.py
from fairseq2.nn import (
    SinusoidalPositionEncoder,
    LearnedPositionEncoder,
    RotaryEncoder,  # RoPE - used in Gemma3n
    DualRotaryEncoder,
)

# RotaryEncoder:
# - Configurable theta (frequency base)
# - Efficient relative positional encoding
# - **USED IN GEMMA3N TEXT DECODER**
```

### Foundation Layers
```python
# File: src/fairseq2/nn/projection.py
from fairseq2.nn import Linear

# Linear projection:
# - Configurable bias, init functions
# - Tensor parallelism support (ColumnShardedLinear, RowShardedLinear)
```

```python
# File: src/fairseq2/nn/embedding.py
from fairseq2.nn import StandardEmbedding

# Embedding layer:
# - Configurable pad_idx
# - Custom initialization functions
```

## Custom Implementations Needed

### 1. Audio Config
**File:** `src/fairseq2/models/gemma3n/audio_config.py` (NEW)
```python
@dataclass
class Gemma3nAudioConfig:
    """Audio tower configuration for Gemma3n."""
    hidden_size: int = 1536
    vocab_size: int = 128
    vocab_offset: int = 262272
    input_feat_size: int = 128
    rms_norm_eps: float = 1e-6

    # Conformer config
    conf_num_hidden_layers: int = 12
    conf_num_attention_heads: int = 8
    conf_conv_kernel_size: int = 5
    conf_attention_chunk_size: int = 12
    conf_attention_context_left: int = 13
    conf_attention_context_right: int = 0
    conf_attention_logit_cap: float = 50.0
    conf_reduction_factor: int = 4
    conf_residual_weight: float = 0.5
    # ... etc
```

### 2. Chunked Local Attention
**Need to verify:** Does `ChunkedAttentionBias` match our needs?
- chunk_size: 12
- left_context: 13
- right_context: 0

**If not, create:** Custom SDPA variant with chunking logic

### 3. Per-Dimension Attention Scaling
**File:** Custom modification to attention
```python
# per_dim_scale: (192,) parameter
# Apply scaling per attention dimension, not global
# Need to add to StandardMultiheadAttention or create variant
```

### 4. Multimodal Embedder
**File:** `src/fairseq2/models/gemma3n/multimodal_embedder.py` (NEW)
```python
class Gemma3nMultimodalEmbedder(Module):
    """Projects audio features to text model space."""
    # - Embedding table for hard tokens (128 → 1536)
    # - Hard/soft embedding norms (RMSNorm)
    # - Projection to text space (1536 → 2048)
    # - Post-projection norm (no scale)

    # Use: StandardEmbedding, RMSNorm, Linear (all exist!)
```

### 5. Subsample Conv Projection
**File:** `src/fairseq2/models/gemma3n/subsample.py` (NEW)
```python
class Gemma3nSubsampleConvProjection(Module):
    """Downsample mel-spectrogram to audio tower hidden size."""
    # Conv block 0: 1→128 channels (3x3)
    # Conv block 1: 128→32 channels (3x3)
    # Linear projection: 1024→1536

    # Can adapt Conv1dFbankSubsampler or build custom
```

## Implementation Checklist

- [ ] Create `audio_config.py` with `Gemma3nAudioConfig`
- [ ] Verify `ChunkedAttentionBias` works for our chunked attention
- [ ] Implement per-dim scaling (if needed as custom attention)
- [ ] Implement `Gemma3nMultimodalEmbedder` using existing primitives
- [ ] Implement `Gemma3nSubsampleConvProjection` (adapt or custom)
- [ ] Assemble conformer layers using `ConformerBlock`, `ConformerConvolution`, `AltUpFeedForwardNetwork`
- [ ] Create `Gemma3nAudioTower` combining all components
- [ ] Update `Gemma3nModel` to integrate audio tower
- [ ] Update checkpoint conversion in `interop.py`

## Key Patterns to Follow

**Always use existing fairseq2 patterns:**
1. `Module` base class from `fairseq2.nn`
2. `dataclass` configs with explicit types
3. `device` and `dtype` parameters in constructors
4. `reset_parameters()` for initialization
5. Type hints everywhere
6. Docstrings for all public APIs

**Never:**
- Reimplement RMSNorm, Linear, Embedding, etc.
- Create custom attention without checking SDPA variants first
- Bypass fairseq2 abstractions (use Module, not nn.Module directly)

## References

- Conformer: `src/fairseq2/models/conformer/`
- Transformer primitives: `src/fairseq2/models/transformer/`
- NN primitives: `src/fairseq2/nn/`
- S2T models: `src/fairseq2/models/s2t_transformer/`
- Wav2Vec2: `src/fairseq2/models/wav2vec2/`
- Gemma3n text: `src/fairseq2/models/gemma3n/`
