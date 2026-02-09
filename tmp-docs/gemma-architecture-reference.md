# Gemma 3n Architecture: Complete Technical Reference for Reimplementation

## Document Purpose
This document serves as a ground truth reference for junior researchers implementing Gemma 3n
in a fresh training framework. It covers architecture, key innovations, multimodal processing,
and provides a step-by-step implementation plan.

---

## 1. Executive Summary

**Gemma 3n** is Google DeepMind's edge-optimized multimodal language model, distinct from Gemma 3.
Key distinguishing features:

| Feature | Gemma 3 | Gemma 3n |
|---------|---------|----------|
| Architecture | Standard Transformer | MatFormer (Nested Transformer) |
| Vision Encoder | SigLIP | MobileNet-V5 |
| Audio Support | ❌ | ✅ (USM-based encoder) |
| Memory Optimization | Standard | PLE Caching, AltUp, LAuReL |
| Parameter Sizes | 1B, 4B, 12B, 27B | E2B (5B total/2B effective), E4B (8B total/4B effective) |
| Context Window | Up to 128K | 32K tokens |
| Attention Pattern | 5:1 local:global | 4:1 local:global + KV cache sharing |

**Key Papers/References:**
- Gemma 3 Technical Report: arxiv.org/abs/2503.19786 [Og9c]
- MatFormer: arxiv.org/abs/2310.07707 [rWsk]
- LAuReL: arxiv.org/abs/2411.07501 [53vv]
- AltUp: arxiv.org/abs/2301.13310 [bBAD]
- HuggingFace Implementation: transformers/models/gemma3n [nqxZ]
- Reverse Engineering: github.com/antimatter15/reverse-engineering-gemma-3n [MALA]

---

## 2. Architecture Overview

### 2.1 Model Variants

```
E2B Model:
├── Core Transformer: ~2B parameters (loaded on accelerator)
└── Per-Layer Embeddings (PLE): ~3B parameters (CPU/storage cached)
    Total: 5B parameters, Effective: 2B

E4B Model:
├── Core Transformer: ~4B parameters (loaded on accelerator)
└── Per-Layer Embeddings (PLE): ~4B parameters (CPU/storage cached)
    Total: 8B parameters, Effective: 4B

Note: E4B contains E2B as a nested submodel (MatFormer property)
```

### 2.2 High-Level Forward Pass

```
Input Processing:
    Text → Tokenization → Token Embeddings
    Image → MobileNet-V5 → Visual Token Embeddings
    Audio → USM Encoder → Audio Token Embeddings
    ↓
    Concatenate: [Text tokens | Visual tokens | Audio tokens]
    ↓
    Add Positional Encodings (RoPE)

Transformer Processing:
    For each block of 5 layers:
        Layers 0-3: Local Sliding Window Attention (1024 tokens)
        Layer 4: Global Attention (full 32K context) + KV Cache Sharing

    Each layer:
        1. Load PLE_i from CPU cache
        2. Compute: enhanced_input = PLE_i(input)
        3. LAuReL residual: x' = (L*R)*x + attention(x)
        4. AltUp: Alternating subset updates
        5. FFN with activation sparsity (statistical top-k)
        6. Discard PLE_i from memory

Output Generation:
    Final hidden states → Output projection → Logits over vocabulary
```

---

## 3. Core Architectural Components

### 3.1 MatFormer (Matryoshka Transformer)

**Purpose:** Train a single model that can be sliced into multiple smaller, fully functional submodels.

**Key Insight:** Nested FFN structure where smaller model parameters are a strict subset of larger model.

**Implementation in Gemma 3n:**
```python
# Conceptual structure - E4B FFN contains E2B FFN
class MatFormerFFN:
    def __init__(self, config):
        # E4B uses intermediate_size = 16384
        # E2B uses intermediate_size = 8192 (first half of E4B weights)
        self.intermediate_size_e4b = 16384
        self.intermediate_size_e2b = 8192

        self.gate_proj = nn.Linear(hidden_size, intermediate_size_e4b, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size_e4b, bias=False)
        self.down_proj = nn.Linear(intermediate_size_e4b, hidden_size, bias=False)

    def forward(self, x, use_e2b=False):
        intermediate_size = self.intermediate_size_e2b if use_e2b else self.intermediate_size_e4b

        # Slice weights for E2B mode
        gate = self.gate_proj.weight[:intermediate_size]
        up = self.up_proj.weight[:intermediate_size]
        down = self.down_proj.weight[:, :intermediate_size]

        # GEGLU activation
        hidden = F.gelu(x @ gate.T) * (x @ up.T)
        return hidden @ down.T
```

**Reference:** arxiv.org/html/2310.07707v2 [rWsk]

**Training Consideration:**
- Joint optimization of multiple nested configurations
- Loss = L_E4B + α₁ × L_E2B + α₂ × L_intermediate_configs

**⚠️ ASSUMPTION TO VERIFY:** The exact loss weighting coefficients (α₁, α₂) are not public. Start with equal weighting and tune empirically.

---

### 3.2 Per-Layer Embeddings (PLE)

**Purpose:** Reduce accelerator memory by offloading layer-specific embedding enhancements to CPU/storage.

**Mechanism:**
```python
class PerLayerEmbedding:
    """
    PLE computes layer-specific input enhancements.
    Parameters are cached in CPU memory/SSD, loaded per-layer.

    Reference: [sBko], [Xlig]
    """
    def __init__(self, config, layer_idx):
        self.layer_idx = layer_idx
        # PLE parameters - stored separately from main model
        # These can be large (E2B: 3B total, E4B: 4B total across all layers)
        self.embedding_matrix = nn.Parameter(...)  # Shape TBD

    def forward(self, hidden_states):
        # Load from CPU/storage cache
        # Compute enhancement based on current residual stream
        # Uses attention-like mechanism to "select" relevant embeddings
        enhancement = self._compute_enhancement(hidden_states)
        return hidden_states + enhancement

    def _compute_enhancement(self, hidden_states):
        # "Decides which pieces of PLE are relevant"
        # Varies from not at all, somewhat, to a lot
        # Incorporates back into stream
        pass
```

**Processing Flow per layer:**
1. Load PLE_i from cache to CPU memory
2. Compute embedding enhancement: E_i = PLE_i(input_i)
3. Pass enhanced input to layer i in accelerator
4. Discard PLE_i from memory
5. Move to layer i+1

**⚠️ ASSUMPTION TO VERIFY:** The exact PLE architecture (attention mechanism, how selection works) is not fully documented. The reverse engineering [MALA] suggests it uses the residual stream to "decide" relevance. Test with cross-attention-like mechanism.

---

### 3.3 LAuReL (Learned Augmented Residual Layer)

**Purpose:** Enhance residual connections with minimal parameter overhead.

**Standard Residual:**
```python
x' = x + attention(x)
```

**LAuReL-LR (Low Rank) Residual - used in Gemma 3n:**
```python
# Instead of identity residual, use low-rank transformation
x' = (L @ R) @ x + attention(x)

# Where L: (hidden_size, laurel_rank), R: (laurel_rank, hidden_size)
# laurel_rank = 64 (from HuggingFace config) [nqxZ]
```

**Implementation:**
```python
class LAuReLResidual(nn.Module):
    """
    Learned Augmented Residual Layer (Low-Rank variant)
    Reference: arxiv.org/abs/2411.07501 [53vv]
    """
    def __init__(self, hidden_size, laurel_rank=64):
        super().__init__()
        self.L = nn.Linear(hidden_size, laurel_rank, bias=False)
        self.R = nn.Linear(laurel_rank, hidden_size, bias=False)

    def forward(self, x, sublayer_output):
        # Low-rank residual: (L*R)*x instead of identity
        residual = self.R(self.L(x))
        return residual + sublayer_output
```

**Key Advantage:** 0.012% parameter increase for consistent performance gains.

---

### 3.4 AltUp (Alternating Updates)

**Purpose:** Reduce computation by selectively processing only a subset of positions/dimensions.

**Two variants used:**
1. **Embedding-AltUp:** Applies to embedding dimension
2. **Sequence-AltUp:** Applies to sequence dimension (more impactful for long contexts)

**Conceptual Implementation:**
```python
class AltUp(nn.Module):
    """
    Alternating Updates for efficient inference.
    Reference: arxiv.org/abs/2301.13310 [bBAD]
    """
    def __init__(self, config):
        self.altup_factor = 2  # Process 1/2 at a time

    def forward(self, hidden_states, layer_idx):
        # Sequence-AltUp: alternate which positions are computed
        if layer_idx % 2 == 0:
            # Process even positions, predict odd
            active_positions = hidden_states[:, ::2, :]
            # ... process active positions
            # ... predict/correct inactive positions
        else:
            # Process odd positions, predict even
            active_positions = hidden_states[:, 1::2, :]
            # ...
```

**⚠️ ASSUMPTION TO VERIFY:** Exact AltUp configuration (factor, prediction mechanism) not fully documented. The HuggingFace implementation may reveal details.

---

### 3.5 Attention Mechanism

**Pattern:** 4:1 local:global ratio (differs from Gemma 3's 5:1)

```python
# Layer type pattern (repeating)
layer_types = ['local', 'local', 'local', 'local', 'global'] * num_blocks
```

**Local Attention (Sliding Window):**
```python
class LocalSlidingWindowAttention(nn.Module):
    """
    Sliding window attention with window size 1024.
    KV cache only stores last 1024 tokens.

    Reference: [W9o4], [gtNr]
    """
    def __init__(self, config):
        self.window_size = 1024  # From config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        # ... Q, K, V projections

    def forward(self, hidden_states, attention_mask, position_ids):
        # Create sliding window mask
        # Each position attends to [-window_size, 0] relative positions
        causal_mask = self._create_sliding_window_mask(
            seq_len=hidden_states.size(1),
            window_size=self.window_size
        )

        # Apply standard attention with window mask
        # Use RoPE with base frequency 10,000 for local layers
        ...
```

**Global Attention:**
```python
class GlobalAttention(nn.Module):
    """
    Full attention over entire context (up to 32K tokens).
    Uses different RoPE base frequency.

    Reference: [lUcc], [0dat]
    """
    def __init__(self, config):
        self.max_position = 32768
        # Use RoPE with base frequency 1,000,000 for global layers
        self.rope_base = 1_000_000  # vs 10,000 for local

    def forward(self, hidden_states, attention_mask, position_ids):
        # Full causal attention
        # KV cache stores full sequence
        ...
```

**KV Cache Sharing:**
```python
# Middle-layer global attention caches K, V
# Subsequent layers reuse this cache (2x prefill speedup)

class KVCacheManager:
    def __init__(self):
        self.shared_kv = None
        self.share_from_layer = None  # Middle global layer

    def cache_for_sharing(self, layer_idx, K, V):
        if self._is_sharing_layer(layer_idx):
            self.shared_kv = (K, V)

    def get_shared_kv(self, layer_idx):
        if self._should_use_shared(layer_idx):
            return self.shared_kv
        return None
```

**⚠️ ASSUMPTION TO VERIFY:** Exact layer for KV sharing not documented. Likely middle global layer.

---

### 3.6 Normalization

**Pre-norm and Post-norm with RMSNorm:**
```python
class Gemma3nRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Uses (1 + weight) scaling like Gemma family.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * (1.0 + self.weight.float())).type(input_dtype)
```

**QK-Norm (replaces soft-capping from Gemma 2):**
```python
# Applied to Q and K before attention computation
# Prevents logits from exploding without tanh overhead
q = F.normalize(q, dim=-1)
k = F.normalize(k, dim=-1)
# Then apply RoPE and compute attention
```

---

### 3.7 Activation Sparsity (Statistical Top-k)

**Purpose:** Reduce FFN computation by only computing top-k activations.

```python
class SparseMLP(nn.Module):
    """
    FFN with statistical top-k activation sparsity.

    Reference: HuggingFace config [9ftS]
    Config parameter: activation_sparsity_pattern (Sequence[float])
    """
    def __init__(self, config, layer_idx):
        self.sparsity_factor = config.activation_sparsity_pattern[layer_idx]
        # First 10 layers typically have different sparsity

    def forward(self, x):
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)

        # Apply GELU to gate
        gate_activated = F.gelu(gate_output)

        # Statistical top-k: keep only top (1-sparsity) fraction
        if self.sparsity_factor > 0:
            k = int(gate_activated.size(-1) * (1 - self.sparsity_factor))
            topk_vals, topk_idx = torch.topk(gate_activated.abs(), k, dim=-1)
            mask = torch.zeros_like(gate_activated)
            mask.scatter_(-1, topk_idx, 1.0)
            gate_activated = gate_activated * mask

        hidden = gate_activated * up_output
        return self.down_proj(hidden)
```

**⚠️ ASSUMPTION TO VERIFY:** Exact sparsity_pattern values per layer not documented publicly.

---

### 3.8 Positional Embeddings (RoPE)

**Dual RoPE frequencies:**
```python
def get_rope_config(layer_type):
    if layer_type == 'local':
        return {
            'base': 10_000,
            'scaling_factor': 1.0
        }
    else:  # global
        return {
            'base': 1_000_000,  # 100x larger
            'scaling_factor': 8.0  # For 32K context extension
        }
```

**Standard RoPE implementation:**
```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # Split into pairs
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    # Apply rotation
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    return q_rot, k_rot
```

---

## 4. Multimodal Processing

### 4.1 Vision Encoder (MobileNet-V5)

**Architecture:** MobileNet-V5-300M with Multi-Query Attention blocks and Multi-Scale Fusion Adapter.

```python
class Gemma3nVisionEncoder:
    """
    MobileNet-V5 vision encoder.
    Reference: [s9gN], [gvs6]

    Key features:
    - Resolution: up to 768x768 pixels
    - 60 FPS on Pixel device
    - 13x faster than SigLIP (with quantization)
    - 46% fewer parameters than SigLIP
    """
    def __init__(self, config):
        # MobileNetV5 backbone from timm
        self.backbone = timm.create_model('mobilenetv5_300m', pretrained=False)

        # Multi-Scale Fusion Adapter
        self.msfa = MultiScaleFusionAdapter(config)

        # Projection to LM hidden size
        self.projector = nn.Linear(vision_hidden_size, text_hidden_size)

    def forward(self, pixel_values):
        # Extract multi-scale features
        features = self.backbone(pixel_values)

        # Fuse across scales
        fused = self.msfa(features)

        # Project to token embeddings
        # Output shape: (batch, num_image_tokens, hidden_size)
        # Default: 256 tokens per image [nqxZ]
        image_tokens = self.projector(fused)

        return image_tokens
```

**Image token configuration:**
```python
# From HuggingFace Gemma3nConfig [nqxZ]
mm_tokens_per_image = 256
boi_token_index = 255999  # Begin of image
eoi_token_index = 256000  # End of image
image_token_index = 262144
```

**⚠️ ASSUMPTION TO VERIFY:** Exact MSFA architecture not fully documented. Check timm implementation.

---

### 4.2 Audio Encoder (USM-based)

**Architecture:** Based on Universal Speech Model architecture.

```python
class Gemma3nAudioEncoder:
    """
    USM-based audio encoder.
    Reference: [rLDP], [NVCd]

    Key features:
    - 160ms chunks → 1 token per chunk
    - Up to 30 second audio clips
    - Supports ASR and AST for 35 languages
    """
    def __init__(self, config):
        self.chunk_duration_ms = 160
        self.max_duration_s = 30
        self.sample_rate = 16000  # Typical for speech

        # Feature extraction (mel spectrograms)
        self.feature_extractor = Gemma3nAudioFeatureExtractor(config)

        # USM encoder layers
        self.encoder = USMEncoder(config)

        # Projection to LM hidden size
        self.projector = nn.Linear(audio_hidden_size, text_hidden_size)

    def forward(self, audio_features):
        # audio_features from Gemma3nAudioFeatureExtractor
        encoded = self.encoder(audio_features)
        audio_tokens = self.projector(encoded)
        # Shape: (batch, num_audio_tokens, hidden_size)
        # ~188 tokens for 30s audio (30000ms / 160ms)
        return audio_tokens
```

**⚠️ NOTE:** Audio weights have not been released publicly as of the documentation date [MALA].

---

### 4.3 Multimodal Input Fusion

```python
class Gemma3nProcessor:
    """
    Combines text, image, and audio into unified input sequence.
    Reference: [nqxZ]
    """
    def __init__(self, config):
        self.tokenizer = Gemma3nTokenizer(config)
        self.image_processor = Gemma3nImageProcessor(config)
        self.audio_processor = Gemma3nAudioFeatureExtractor(config)

        # Token sequence lengths
        self.audio_seq_length = 188  # ~30s audio
        self.image_seq_length = 256

    def __call__(self, text=None, images=None, audio=None):
        # Tokenize text
        text_tokens = self.tokenizer(text)

        # Process images if present
        image_embeds = None
        if images is not None:
            pixel_values = self.image_processor(images)
            # Will be processed by vision encoder in model

        # Process audio if present
        audio_features = None
        if audio is not None:
            audio_features = self.audio_processor(audio)
            # Will be processed by audio encoder in model

        return {
            'input_ids': text_tokens['input_ids'],
            'attention_mask': text_tokens['attention_mask'],
            'pixel_values': pixel_values,
            'audio_features': audio_features
        }
```

**Special tokens for modality markers:**
```python
# Replace in chat template:
# <image_soft_token> → <__media__>
# <audio_soft_token> → <__media__>
```

---

## 5. Greenfield Implementation Plan

### Phase 1: Core Primitives (Week 1-2)

**Step 1.1: Basic Building Blocks**
```
□ RMSNorm with (1 + weight) scaling
□ RoPE with configurable base frequency
□ QK-Norm (query/key normalization)
□ GEGLU/SwiGLU activation function
```

**Step 1.2: Attention Primitives**
```
□ Sliding window attention mask generation
□ Full causal attention mask
□ SDPA wrapper with mask support
  - Your SDPA API integration point
  - Support both local and global patterns
```

**Validation:** Unit tests for each primitive against PyTorch reference.

---

### Phase 2: Attention Layers (Week 2-3)

**Step 2.1: Local Sliding Window Attention**
```python
# Key parameters:
window_size = 1024
num_heads = config.num_attention_heads
head_dim = config.head_dim
rope_base = 10_000

# Your SDPA integration:
def local_attention(q, k, v, window_size):
    # Create sliding window mask
    mask = create_sliding_window_mask(seq_len, window_size)
    # Call your SDPA
    return your_sdpa(q, k, v, attn_mask=mask)
```

**Step 2.2: Global Attention**
```python
# Key parameters:
rope_base = 1_000_000
rope_scaling = 8.0  # For 32K context

def global_attention(q, k, v, kv_cache=None):
    # Standard causal mask
    mask = create_causal_mask(seq_len)
    return your_sdpa(q, k, v, attn_mask=mask)
```

**Step 2.3: KV Cache with Sharing**
```python
class HybridKVCache:
    # Minimal cache for local layers (1024 tokens)
    # Full cache for global layers
    # Sharing mechanism for subsequent layers
```

**Validation:** Compare attention outputs with HuggingFace implementation.

---

### Phase 3: Efficient Layers (Week 3-4)

**Step 3.1: LAuReL Residual**
```
□ Low-rank projection (L, R matrices)
□ Integration with attention output
□ laurel_rank = 64
```

**Step 3.2: AltUp**
```
□ Alternating position selection
□ Prediction mechanism for inactive positions
□ Integration with layer forward
```

**Step 3.3: Sparse MLP**
```
□ Statistical top-k implementation
□ Per-layer sparsity configuration
□ GEGLU with sparsity mask
```

**Validation:** Parameter count verification, output comparison.

---

### Phase 4: Decoder Layer Assembly (Week 4-5)

**Step 4.1: Single Decoder Layer**
```python
class Gemma3nDecoderLayer:
    def __init__(self, config, layer_idx):
        self.layer_idx = layer_idx
        self.layer_type = get_layer_type(layer_idx)  # 'local' or 'global'

        # Components
        self.pre_norm = Gemma3nRMSNorm(config.hidden_size)
        self.post_norm = Gemma3nRMSNorm(config.hidden_size)

        if self.layer_type == 'local':
            self.attention = LocalSlidingWindowAttention(config)
        else:
            self.attention = GlobalAttention(config)

        self.laurel = LAuReLResidual(config.hidden_size, config.laurel_rank)
        self.mlp = SparseMLP(config, layer_idx)

    def forward(self, hidden_states, position_ids, kv_cache=None, ple_input=None):
        # Apply PLE enhancement if provided
        if ple_input is not None:
            hidden_states = hidden_states + ple_input

        # Pre-norm + Attention + LAuReL residual
        residual = hidden_states
        hidden_states = self.pre_norm(hidden_states)
        attn_output = self.attention(hidden_states, position_ids, kv_cache)
        hidden_states = self.laurel(residual, attn_output)

        # Post-norm + MLP + residual
        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states
```

**Step 4.2: PLE Module**
```python
class PerLayerEmbedder:
    # Separate module for PLE computation
    # Can be loaded/unloaded per layer
```

---

### Phase 5: Full Model Assembly (Week 5-6)

**Step 5.1: Text Model**
```python
class Gemma3nTextModel:
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Gemma3nDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = Gemma3nRMSNorm(config.hidden_size)

        # PLE modules (can be offloaded)
        self.ple_modules = nn.ModuleList([
            PerLayerEmbedder(config, i) for i in range(config.num_hidden_layers)
        ])
```

**Step 5.2: MatFormer Slicing**
```python
def get_e2b_model(e4b_model):
    """Extract E2B submodel from E4B."""
    # Slice FFN weights to smaller size
    # All other weights shared
```

---

### Phase 6: Vision Integration (Week 6-7)

**Step 6.1: MobileNet-V5 Encoder**
```
□ Use timm for backbone
□ Implement MSFA
□ Projection to LM space
```

**Step 6.2: Image-Text Fusion**
```
□ Image token insertion at correct positions
□ BOI/EOI token handling
□ Attention mask for image tokens
```

---

### Phase 7: Audio Integration (Week 7-8)

**Step 7.1: USM Encoder**
```
□ Audio feature extraction
□ Encoder architecture
□ Projection to LM space
```

**Step 7.2: Audio-Text Fusion**
```
□ Audio token insertion
□ Temporal alignment
```

**⚠️ NOTE:** Audio weights not released. May need to wait or train from scratch.

---

### Phase 8: Inference Validation (Week 8-9)

**Validation Checkpoints:**
```
□ Load HuggingFace weights successfully
□ Text-only generation matches HF output (token-by-token)
□ Image+Text generation matches HF output
□ KV cache produces identical results to full recompute
□ E2B extraction from E4B produces correct outputs
```

---

## 6. Configuration Reference

### 6.1 Key Config Parameters (from HuggingFace [nqxZ])

```python
# Text Config
hidden_size: int  # E2B: TBD, E4B: TBD
intermediate_size: int  # FFN size
num_hidden_layers: int
num_attention_heads: int
head_dim: int
vocab_size: int  # ~256K

# Attention
sliding_window: int = 1024
layer_types: Sequence[str]  # ['local', 'local', 'local', 'local', 'global', ...]

# Efficiency
laurel_rank: int = 64
activation_sparsity_pattern: Sequence[float]  # Per-layer sparsity

# Multimodal
mm_tokens_per_image: int = 256
audio_seq_length: int = 188
```

### 6.2 Expected Tensor Shapes

```python
# Input
input_ids: (batch, seq_len)
attention_mask: (batch, seq_len)
pixel_values: (batch, num_images, 3, height, width)  # height=width=768
audio_features: (batch, audio_seq_len, audio_feature_dim)

# Intermediate
hidden_states: (batch, seq_len, hidden_size)
q, k, v: (batch, num_heads, seq_len, head_dim)

# Output
logits: (batch, seq_len, vocab_size)
```

---

## 7. Key Assumptions and Testing Notes

### 7.1 Documented vs Assumed

| Component | Status | Notes |
|-----------|--------|-------|
| MatFormer nested structure | ✅ Documented | arxiv.org/abs/2310.07707 |
| LAuReL-LR variant | ✅ Documented | laurel_rank=64 in config |
| AltUp mechanism | ⚠️ Partial | Exact config unclear |
| PLE architecture | ⚠️ Partial | Mechanism described, not implementation |
| Sparsity pattern | ⚠️ Partial | First 10 layers different |
| KV sharing layer | ⚠️ Assumed | Middle global layer |
| Audio encoder | ⚠️ Incomplete | Weights not released |

### 7.2 Tests to Run on Demand

1. **Weight Loading:** Verify all tensors load with correct shapes
2. **Forward Pass:** Compare hidden states at each layer with HF
3. **Attention Patterns:** Visualize attention to verify local/global
4. **KV Cache:** Compare with/without cache for identical outputs
5. **Sparsity:** Measure actual sparsity in MLP activations
6. **PLE Offload:** Verify memory usage with PLE on/off accelerator

### 7.3 Potential Pitfalls

1. **RoPE frequencies:** Double-check base (10K vs 1M) per layer type
2. **Normalization order:** Pre-norm AND post-norm in Gemma 3n
3. **Weight scaling:** (1 + weight) in RMSNorm, not just weight
4. **QK-Norm:** Applied before RoPE, not after
5. **MatFormer slicing:** Ensure E2B uses first half of E4B FFN weights exactly

---

## 8. References

**Primary:**
- [Og9c] Gemma 3 Technical Report: https://arxiv.org/abs/2503.19786
- [rWsk] MatFormer: https://arxiv.org/abs/2310.07707
- [nqxZ] HuggingFace Gemma3n: https://huggingface.co/docs/transformers/main/en/model_doc/gemma3n
- [MALA] Reverse Engineering: https://github.com/antimatter15/reverse-engineering-gemma-3n

**Supporting:**
- [53vv] LAuReL: https://arxiv.org/abs/2411.07501
- [bBAD] AltUp: https://arxiv.org/abs/2301.13310
- [s9gN] llama.cpp MobileNetV5 PR: https://github.com/ggml-org/llama.cpp/pull/18256
- [gvs6] Gemma 3n Developer Guide: https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/
- [Xlig] Medium Deep Dive: https://nageswararaovutla7.medium.com/gemma-3n-model-architecture-a-complete-technical-deep-dive

---

*Document generated: 2026-02-08*
*For implementation questions, cross-reference HuggingFace source code.*
