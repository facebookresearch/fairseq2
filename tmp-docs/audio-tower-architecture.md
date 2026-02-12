# Gemma3n Audio Tower Architecture

## Summary

Based on checkpoint inspection of google/gemma-3n-E2B-it, the audio tower consists of:

1. **Subsample Convolution Projection** (5 params)
2. **Conformer Blocks** (264 params total = ~11 layers)
3. **Embed Audio** (multimodal embedder to project to text space)

## Subsample Convolution Projection

Downsamples mel-spectrogram and projects to hidden size.

### Components
```
subsample_conv_projection.conv_0.conv.weight: (128, 1, 3, 3)
subsample_conv_projection.conv_0.norm.weight: (128,)
subsample_conv_projection.conv_1.conv.weight: (32, 128, 3, 3)
subsample_conv_projection.conv_1.norm.weight: (32,)
subsample_conv_projection.input_proj_linear.weight: (1536, 1024)
```

### Pipeline
1. Input: mel-spectrogram (batch, time, 128 features)
2. Conv 0: 1 channel → 128 channels (3x3 kernel)
3. Norm 0: RMS norm over 128 channels
4. Conv 1: 128 channels → 32 channels (3x3 kernel)
5. Norm 1: RMS norm over 32 channels
6. Linear: Flatten + project 1024 → 1536 (hidden_size)

## Conformer Blocks

Universal Speech Model (USM) Conformer encoder with **12 layers**.

### Verified Layer Structure

All 12 layers have identical structure with **5 components**:

1. **ffw_layer_start**: First FFN (pre-attention)
2. **attention**: Multi-head self-attention with relative pos embeddings
3. **lconv1d**: Lightweight depthwise 1D convolution
4. **ffw_layer_end**: Second FFN (post-convolution)
5. **norm**: Final layer normalization

### Layer 0 Components (22 params total)

**attention** (8 params):
```
attention.attn.k_proj.weight: (1536, 1536)
attention.attn.per_dim_scale: (192,)
attention.attn.q_proj.weight: (1536, 1536)
attention.attn.relative_position_embedding.pos_proj.weight: (1536, 1536)
attention.attn.v_proj.weight: (1536, 1536)
attention.post.weight: (1536, 1536)
attention.post_norm.weight: (1536,)
attention.pre_attn_norm.weight: (1536,)
```

**ffw_layer_start** (4 params):
```
ffw_layer_start.ffw_layer_1.weight: (6144, 1536)
ffw_layer_start.ffw_layer_2.weight: (1536, 6144)
ffw_layer_start.post_layer_norm.weight: (1536,)
ffw_layer_start.pre_layer_norm.weight: (1536,)
```

**lconv1d** (5 params):
```
lconv1d.conv_norm.weight: (1536,)
lconv1d.depthwise_conv1d.weight: (1536, 1, 5)
lconv1d.linear_end.weight: (1536, 1536)
lconv1d.linear_start.weight: (3072, 1536)  # GLU: 2x expansion
lconv1d.pre_layer_norm.weight: (1536,)
```

**ffw_layer_end** (4 params):
```
ffw_layer_end.ffw_layer_1.weight: (6144, 1536)
ffw_layer_end.ffw_layer_2.weight: (1536, 6144)
ffw_layer_end.post_layer_norm.weight: (1536,)
ffw_layer_end.pre_layer_norm.weight: (1536,)
```

**norm** (1 param):
```
norm.weight: (1536,)
```

### Key Dimensions
- **hidden_size**: 1536
- **num_attention_heads**: 8
- **head_dim**: 192 (1536 / 8)
- **ffn_inner_dim**: 6144 (4x hidden_size)
- **conv_kernel_size**: 5 (depthwise conv)
- **num_layers**: 12

### Conformer Config
```python
conf_num_hidden_layers: 12
conf_num_attention_heads: 8
conf_conv_kernel_size: 5
conf_attention_chunk_size: 12
conf_attention_context_left: 13
conf_attention_context_right: 0
conf_attention_logit_cap: 50.0
conf_reduction_factor: 4
conf_residual_weight: 0.5
```

### Architecture Pattern (per layer)

```
Input (1536)
├─> ffw_layer_start
│   ├─> pre_layer_norm
│   ├─> linear_1: 1536 → 6144
│   ├─> activation (likely SiLU/Swish)
│   ├─> linear_2: 6144 → 1536
│   └─> post_layer_norm + residual (weight 0.5)
│
├─> attention
│   ├─> pre_attn_norm
│   ├─> Q/K/V projections (1536 → 1536)
│   ├─> Relative position embeddings
│   ├─> Multi-head attention (8 heads, per-dim scale)
│   ├─> Attention softcapping (logit_cap=50.0)
│   ├─> Chunked local attention (chunk=12, left=13, right=0)
│   ├─> Output projection
│   └─> post_norm + residual (weight 0.5)
│
├─> lconv1d
│   ├─> pre_layer_norm
│   ├─> linear_start: 1536 → 3072 (GLU)
│   ├─> depthwise_conv1d: kernel=5
│   ├─> conv_norm
│   ├─> activation (likely SiLU/Swish)
│   ├─> linear_end: 1536 → 1536
│   └─> residual (weight 0.5)
│
├─> ffw_layer_end
│   ├─> pre_layer_norm
│   ├─> linear_1: 1536 → 6144
│   ├─> activation
│   ├─> linear_2: 6144 → 1536
│   └─> post_layer_norm + residual (weight 0.5)
│
└─> norm (final layer norm)
```

## Embed Audio

Projects audio tower outputs to text model hidden size.

### Components
```
embed_audio.embedding.weight: (128, 1536)                    # Hard token embeddings
embed_audio.hard_embedding_norm.weight: (1536,)              # RMS norm
embed_audio.soft_embedding_norm.weight: (1536,)              # RMS norm
embed_audio.embedding_projection.weight: (2048, 1536)       # Project to text space
embed_audio.embedding_post_projection_norm.weight: (2048,)   # RMS norm (no scale)
```

### Pipeline
1. Input: Either token IDs or soft embeddings from audio_tower
2. If token IDs: Embedding lookup (128 vocab → 1536) + hard_embedding_norm
3. If soft embeddings: soft_embedding_norm
4. Linear projection: 1536 → 2048 (text hidden_size)
5. Post-projection norm (no scale parameter)

### Token Vocab
- **vocab_size**: 128 (special audio tokens)
- **vocab_offset**: 262272 (offset in main tokenizer)
- Tokens include: end-of-audio, soft token placeholders

## Audio Config Full Parameters

```python
hidden_size: 1536
vocab_size: 128
vocab_offset: 262272
input_feat_size: 128  # Mel-spectrogram features
rms_norm_eps: 1e-06
gradient_clipping: 10000000000.0

# Conformer config
conf_num_hidden_layers: 12
conf_num_attention_heads: 8
conf_conv_kernel_size: 5
conf_attention_chunk_size: 12
conf_attention_context_left: 13
conf_attention_context_right: 0
conf_attention_logit_cap: 50.0
conf_reduction_factor: 4  # Subsampling factor
conf_residual_weight: 0.5  # Residual connection weight

# Other
chunk_size_feed_forward: 0
default_theta: 10000.0
dtype: torch.bfloat16
```

## Integration with Main Model

### Data Flow
1. **Audio preprocessing**: Raw audio → mel-spectrogram (128 features)
   - Via `Gemma3nAudioFeatureExtractor`
   - 16kHz sample rate, 32ms frames, 10ms hop

2. **Subsampling**: Mel-spec → downsampled features (1536 dim)
   - Conv layers downsample time dimension
   - Linear projection to hidden_size

3. **Conformer encoding**: Features → contextual embeddings (1536 dim)
   - 11 conformer layers with local attention
   - Relative position embeddings

4. **Projection to text space**: Audio embeds → text embeds (2048 dim)
   - Via embed_audio module
   - Matches text decoder hidden_size

5. **Decoder integration**: Audio embeds injected via AltUp
   - `altup.modality_router` routes audio vs text
   - Audio features mixed with text embeddings

## Implementation TODO

### Phase 2a Checklist

**Core Components:**
- [ ] `Gemma3nAudioConfig` - Audio encoder configuration
  - [ ] All conformer parameters
  - [ ] Subsampling config
  - [ ] Integration with Gemma3nConfig

- [ ] `Gemma3nSubsampleConvProjection` - Downsample mel-spec
  - [ ] Conv block 0: 1→128 channels (3x3)
  - [ ] Conv block 1: 128→32 channels (3x3)
  - [ ] Linear projection: 1024→1536

- [ ] `Gemma3nConformerFFN` - Feed-forward network
  - [ ] Pre/post layer norms
  - [ ] Linear layers: 1536→6144→1536
  - [ ] Activation (SiLU/Swish)
  - [ ] Residual with weight 0.5

- [ ] `Gemma3nConformerLConv1d` - Lightweight 1D convolution
  - [ ] Pre-norm + GLU projection (1536→3072)
  - [ ] Depthwise conv1d (kernel=5)
  - [ ] Conv norm + activation
  - [ ] Output projection (1536→1536)
  - [ ] Residual with weight 0.5

- [ ] `Gemma3nConformerAttention` - Multi-head self-attention
  - [ ] Pre/post norms
  - [ ] Q/K/V projections (1536→1536)
  - [ ] Relative position embeddings (Shaw et al.)
  - [ ] Per-dimension scaling (192,)
  - [ ] Chunked local attention (chunk=12, left=13, right=0)
  - [ ] Attention softcapping (logit_cap=50.0)
  - [ ] Residual with weight 0.5

- [ ] `Gemma3nConformerLayer` - Complete conformer layer
  - [ ] FFN start
  - [ ] Attention
  - [ ] LConv1d
  - [ ] FFN end
  - [ ] Final norm

- [ ] `Gemma3nConformerEncoder` - Stack of 12 conformer layers
  - [ ] Layer stacking
  - [ ] Attention mask handling

- [ ] `Gemma3nMultimodalEmbedder` - Audio→text projection
  - [ ] Hard token embedding table (128→1536)
  - [ ] Hard/soft embedding norms
  - [ ] Projection to text space (1536→2048)
  - [ ] Post-projection norm (no scale)

- [ ] `Gemma3nAudioTower` - Complete audio encoder
  - [ ] Subsample + conformer pipeline
  - [ ] Config management
  - [ ] Forward pass handling

**Model Integration:**
- [ ] Update `Gemma3nConfig` to include audio_config
- [ ] Update `Gemma3nModel` to add audio_tower and embed_audio
- [ ] Modify forward() to handle audio inputs
- [ ] Update AltUp modality routing for audio

**Checkpoint Conversion:**
- [ ] Remove audio key filtering in `interop.py`
- [ ] Map subsample_conv_projection keys
- [ ] Map conformer.{0..11}.* keys
- [ ] Map embed_audio.* keys

**Testing:**
- [ ] Unit tests for conformer components
- [ ] Audio feature extraction parity test
- [ ] Audio tower output parity test
- [ ] End-to-end audio+text inference parity

### Implementation Order

1. **Audio config** (foundation for everything)
2. **Subsampling** (independent, can test alone)
3. **FFN** (reusable component)
4. **LConv1d** (moderate complexity)
5. **Attention** (most complex - relative pos, chunking, softcapping)
6. **Conformer layer** (assembles FFN, attention, conv)
7. **Conformer encoder** (stack layers)
8. **Multimodal embedder** (projects to text space)
9. **Audio tower** (subsample + conformer)
10. **Model integration** (connect to decoder)
11. **Checkpoint conversion** (enable loading)
12. **Parity validation** (verify correctness)
