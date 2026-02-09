# Phase 2: Component Implementation

**Duration**: 2-3 days
**Goal**: Implement core Gemma3n architectural components

---

## 2.1: DualRotaryEncoder (LAuReL)

**File**: `src/fairseq2/nn/position_encoder.py` (extend existing)

LAuReL uses two RoPE frequencies concatenated:
- Standard RoPE: theta=10,000
- Long-range RoPE: theta=100,000

### Implementation

```python
class DualRotaryEncoder(PositionEncoder):
    """
    Dual-frequency Rotary Position Encoder for LAuReL.
    Splits head dimension in half and applies different theta values.
    """

    def __init__(
        self,
        encoding_dim: int,
        max_seq_len: int,
        theta: float = 10_000.0,
        dual_theta: float = 100_000.0,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__(encoding_dim)

        # encoding_dim must be divisible by 4 (split for dual RoPE)
        if encoding_dim % 4 != 0:
            raise ValueError(f"encoding_dim must be divisible by 4, got {encoding_dim}")

        half_dim = encoding_dim // 2

        # Standard RoPE (first half)
        self.rope_std = RotaryEncoder(
            half_dim, max_seq_len, theta=theta, device=device, dtype=dtype
        )

        # Long-range RoPE (second half)
        self.rope_long = RotaryEncoder(
            half_dim, max_seq_len, theta=dual_theta, device=device, dtype=dtype
        )

    def forward(
        self,
        seqs: Tensor,
        seqs_layout: Optional[BatchLayout] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        # Split into two halves along head_dim
        # seqs shape: (batch, num_heads, seq_len, head_dim)
        half_dim = seqs.size(-1) // 2

        seqs_std = seqs[..., :half_dim]
        seqs_long = seqs[..., half_dim:]

        # Apply different RoPE frequencies
        seqs_std = self.rope_std(seqs_std, seqs_layout, state_bag)
        seqs_long = self.rope_long(seqs_long, seqs_layout, state_bag)

        # Concatenate back
        return torch.cat([seqs_std, seqs_long], dim=-1)
```

### Verification Test

```python
def test_dual_rope_frequencies():
    """Verify dual RoPE produces correct frequency values."""
    encoder = DualRotaryEncoder(
        encoding_dim=256,  # Will be split into 128 + 128
        max_seq_len=8192,
        theta=10_000.0,
        dual_theta=100_000.0,
    )

    # Test positions
    positions = torch.tensor([0, 100, 1000, 4000])
    seqs = torch.randn(1, 16, len(positions), 256)

    output = encoder(seqs)

    # Compare with HuggingFace implementation
    # (Load reference values from HF model)
```

---

## 2.2: SoftCappedSDPA

**File**: `src/fairseq2/nn/transformer/attention.py` (new class)

Gemma3n applies soft-capping to attention logits: `tanh(logits / cap) * cap`

### Implementation

```python
class SoftCappedSDPA(SDPA):
    """SDPA with attention logit soft-capping."""

    def __init__(
        self,
        base_sdpa: SDPA,
        soft_cap: float = 30.0,
    ):
        self._base_sdpa = base_sdpa
        self._soft_cap = soft_cap

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        *,
        attn_mask: Optional[AttentionMask] = None,
        needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Compute attention scores
        # Q @ K^T / sqrt(d_k)
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / (keys.size(-1) ** 0.5)

        # Apply soft-capping
        scores = torch.tanh(scores / self._soft_cap) * self._soft_cap

        # Apply mask if provided
        if attn_mask is not None:
            scores = scores + attn_mask.materialize()

        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, values)

        return output, attn_weights if needs_weights else None
```

### Verification Test

```python
def test_soft_capped_attention():
    """Verify soft-capping produces correct values."""
    base_sdpa = TorchSDPA(attn_dropout_p=0.0)
    soft_capped = SoftCappedSDPA(base_sdpa, soft_cap=30.0)

    # Test inputs
    q = torch.randn(2, 8, 128, 64)
    k = torch.randn(2, 8, 128, 64)
    v = torch.randn(2, 8, 128, 64)

    output, weights = soft_capped(q, k, v, needs_weights=True)

    # Verify output shape
    assert output.shape == (2, 8, 128, 64)

    # Compare with HuggingFace
```

---

## 2.3: AltUpFeedForwardNetwork

**File**: `src/fairseq2/models/transformer/ffn.py` (new class)

Key differences from standard GLU:
- Gate activation: GELU (not SiLU)
- Hidden dim: 5376 (not ffn_inner_dim)
- Used only in local layers

### Implementation

```python
class AltUpFeedForwardNetwork(FeedForwardNetwork):
    """
    Alternating Up-projection FFN with GELU gating.
    Used in Gemma3n local layers.
    """

    def __init__(
        self,
        model_dim: int,
        altup_hidden_dim: int,
        bias: bool = False,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()

        self.gate_proj = Linear(
            model_dim, altup_hidden_dim, bias=bias, device=device, dtype=dtype
        )
        self.inner_proj = Linear(
            model_dim, altup_hidden_dim, bias=bias, device=device, dtype=dtype
        )
        self.output_proj = Linear(
            altup_hidden_dim, model_dim, bias=bias, device=device, dtype=dtype
        )

        # GELU activation (NOT SiLU!)
        self.gate_activation = nn.GELU()

    def forward(self, seqs: Tensor) -> Tensor:
        # Gate path with GELU
        gate = self.gate_activation(self.gate_proj(seqs))

        # Inner path
        inner = self.inner_proj(seqs)

        # Element-wise multiplication
        hidden = gate * inner

        # Output projection
        return self.output_proj(hidden)
```

### Verification Test

```python
def test_altup_ffn():
    """Verify AltUp FFN matches HuggingFace."""
    ffn = AltUpFeedForwardNetwork(
        model_dim=2048,
        altup_hidden_dim=5376,
        bias=False,
    )

    seqs = torch.randn(2, 128, 2048)
    output = ffn(seqs)

    assert output.shape == (2, 128, 2048)

    # Compare activations with HF
```

---

## 2.4: Gemma3nDecoderLayer

**File**: `src/fairseq2/models/gemma3n/factory.py`

### Layer Type Logic

```python
def is_global_layer(layer_idx: int, num_layers: int = 35) -> bool:
    """
    Determine if layer uses global attention.

    Global layers: 3, 7, 11, 15, 19, 23, 27, 31, 34
    Pattern: Every 4th layer starting at 3, plus final layer.
    """
    if layer_idx == num_layers - 1:  # Final layer
        return True
    return (layer_idx + 1) % 4 == 0
```

### Layer Implementation

```python
class Gemma3nDecoderLayer(TransformerDecoderLayer):
    """Gemma3n decoder layer with local/global attention modes."""

    def __init__(
        self,
        layer_idx: int,
        config: Gemma3nConfig,
        self_attn: MultiheadAttention,
        ffn: FeedForwardNetwork,
        self_attn_norm: LayerNorm,
        ffn_norm: LayerNorm,
    ):
        super().__init__(self_attn, ffn, self_attn_norm, ffn_norm)

        self.layer_idx = layer_idx
        self.is_global = is_global_layer(layer_idx, config.num_layers)

    def forward(
        self,
        seqs: Tensor,
        seqs_layout: Optional[BatchLayout] = None,
        self_attn_mask: Optional[AttentionMask] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Pre-norm for attention
        residual = seqs
        seqs = self.self_attn_norm(seqs)

        # Self-attention
        seqs = self.self_attn(
            seqs,
            seqs_layout,
            keys=seqs,
            values=seqs,
            attn_mask=self_attn_mask,
            state_bag=state_bag,
        )

        # Residual connection
        seqs = residual + seqs

        # Pre-norm for FFN
        residual = seqs
        seqs = self.ffn_norm(seqs)

        # FFN
        seqs = self.ffn(seqs)

        # Residual connection
        seqs = residual + seqs

        return seqs, None
```

---

## Commit Strategy for Phase 2

**Commit 1**: `[gemma3n] Add DualRotaryEncoder for LAuReL`
- Implement `DualRotaryEncoder` in `src/fairseq2/nn/position_encoder.py`
- Add unit test `test_dual_rope_frequencies()`
- ~200 LOC + 50 LOC tests

**Commit 2**: `[gemma3n] Add SoftCappedSDPA wrapper`
- Implement `SoftCappedSDPA` in `src/fairseq2/nn/transformer/attention.py`
- Add unit test `test_soft_capped_attention()`
- ~150 LOC + 50 LOC tests

**Commit 3**: `[gemma3n] Add AltUpFeedForwardNetwork`
- Implement `AltUpFeedForwardNetwork` in `src/fairseq2/models/transformer/ffn.py`
- Add unit test `test_altup_ffn()`
- ~200 LOC + 50 LOC tests

**Commit 4**: `[gemma3n] Add Gemma3nDecoderLayer`
- Implement `Gemma3nDecoderLayer` in `src/fairseq2/models/gemma3n/factory.py`
- Add layer type logic
- Add unit test `test_decoder_layer()`
- ~300 LOC + 100 LOC tests

**Code Quality Check**:
- Run `/unslop-code` - remove redundant comments, tutorial narration
- Run `/better-engineering` - verify type hints and docstrings
- Commit: `[gemma3n] Phase 2 code quality cleanup`

**Total**: 4-5 commits, ~850 LOC implementation + 250 LOC tests

---

## Deliverables for Phase 2

- [ ] `DualRotaryEncoder` implemented
- [ ] `SoftCappedSDPA` implemented
- [ ] `AltUpFeedForwardNetwork` implemented
- [ ] `Gemma3nDecoderLayer` implemented
- [ ] Component tests written
- [ ] All tests pass against HF reference
- [ ] `/unslop-code` passed
- [ ] `/better-engineering` passed

---

## Next Step
Proceed to `plan-phase-3.md` for parity testing.
