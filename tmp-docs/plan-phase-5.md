# Phase 5: Advanced Features (PLE, MatFormer, KV Sharing, TP)

**Duration**: 1-2 days
**Goal**: Implement efficiency features and distributed training support

---

## 5.1: Per-Layer Embeddings (PLE)

**File**: `src/fairseq2/models/gemma3n/ple.py`

### PLE Module Implementation

```python
import torch
import torch.nn as nn
from typing import Optional
from fairseq2.typing import Device, DataType

class PLEModule(nn.Module):
    """
    Per-Layer Embeddings with expert routing.

    Adds 3B parameters (E2B) CPU-cached for layer-specific expansion.
    """

    def __init__(
        self,
        model_dim: int = 2048,
        ple_hidden_dim: int = 5120,
        num_experts: int = 8,
        top_k: int = 2,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()

        self.model_dim = model_dim
        self.ple_hidden_dim = ple_hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Gate for expert routing
        self.gate = nn.Linear(model_dim, num_experts, device=device, dtype=dtype)

        # Expert networks (up + down projections)
        self.experts_up = nn.ModuleList([
            nn.Linear(model_dim, ple_hidden_dim, bias=False, device=device, dtype=dtype)
            for _ in range(num_experts)
        ])

        self.experts_down = nn.ModuleList([
            nn.Linear(ple_hidden_dim, model_dim, bias=False, device=device, dtype=dtype)
            for _ in range(num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply PLE transformation.

        Args:
            hidden_states: (batch, seq_len, model_dim)

        Returns:
            Enhanced hidden states: (batch, seq_len, model_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute gating scores
        gate_logits = self.gate(hidden_states)  # (batch, seq_len, num_experts)

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = torch.softmax(top_k_logits, dim=-1)  # (batch, seq_len, top_k)

        # Process through selected experts
        output = torch.zeros_like(hidden_states)

        for k in range(self.top_k):
            # Get expert indices for this k
            expert_idx = top_k_indices[:, :, k]  # (batch, seq_len)
            weights = top_k_weights[:, :, k].unsqueeze(-1)  # (batch, seq_len, 1)

            # Process through each expert (batch processing)
            for expert_id in range(self.num_experts):
                # Mask for tokens routed to this expert
                mask = (expert_idx == expert_id)
                if not mask.any():
                    continue

                # Extract tokens for this expert
                expert_input = hidden_states[mask]  # (num_tokens, model_dim)

                # Expert computation: up -> activation -> down
                expert_output = self.experts_up[expert_id](expert_input)
                expert_output = torch.nn.functional.gelu(expert_output)
                expert_output = self.experts_down[expert_id](expert_output)

                # Add weighted expert output
                output[mask] += expert_output * weights[mask]

        return output

    def to_cpu(self):
        """Move PLE parameters to CPU for offloading."""
        for expert_up, expert_down in zip(self.experts_up, self.experts_down):
            expert_up.to("cpu")
            expert_down.to("cpu")

    def to_gpu(self, device: Device):
        """Move PLE parameters to GPU."""
        for expert_up, expert_down in zip(self.experts_up, self.experts_down):
            expert_up.to(device)
            expert_down.to(device)
```

### CPU Offloading Strategy

```python
class PLEOffloadManager:
    """Manage PLE parameter offloading to CPU."""

    def __init__(self, ple_modules: list[PLEModule]):
        self.ple_modules = ple_modules

        # Offload all to CPU initially
        for ple in ple_modules:
            ple.to_cpu()

    def prefetch(self, layer_idx: int, device: Device):
        """Prefetch PLE for next layer to GPU asynchronously."""
        if layer_idx < len(self.ple_modules):
            # Use CUDA stream for async transfer
            self.ple_modules[layer_idx].to_gpu(device)

    def offload(self, layer_idx: int):
        """Offload PLE for current layer back to CPU."""
        if layer_idx < len(self.ple_modules):
            self.ple_modules[layer_idx].to_cpu()
```

---

## 5.2: MatFormer Slicing

**File**: `src/fairseq2/models/gemma3n/factory.py` (extend)

### Slicing Logic

```python
def apply_matformer_slice(
    hidden_states: torch.Tensor,
    is_global_layer: bool,
    e2b_slice_dim: int = 1536,
) -> torch.Tensor:
    """
    Apply MatFormer slicing.

    Local layers (E2B): Process only first e2b_slice_dim dimensions.
    Global layers (E4B): Process full hidden states.
    """
    if is_global_layer:
        return hidden_states  # Full E4B processing

    # E2B slice
    return hidden_states[..., :e2b_slice_dim]


def expand_from_slice(
    sliced_output: torch.Tensor,
    original_dim: int,
    device: Device,
) -> torch.Tensor:
    """Expand sliced output back to original dimensionality."""
    batch_size, seq_len, slice_dim = sliced_output.shape

    # Pad with zeros
    padding = torch.zeros(
        batch_size, seq_len, original_dim - slice_dim,
        device=device, dtype=sliced_output.dtype
    )

    return torch.cat([sliced_output, padding], dim=-1)
```

### Modified Decoder Layer

```python
class Gemma3nDecoderLayer(TransformerDecoderLayer):
    """Extended with MatFormer slicing support."""

    def forward(self, seqs, use_matformer_slicing=False):
        # Apply slicing if needed
        if use_matformer_slicing and not self.is_global:
            seqs = apply_matformer_slice(seqs, self.is_global)

        # Standard layer processing
        seqs = self.self_attn_norm(seqs)
        seqs = self.self_attn(seqs, ...)
        # ... rest of layer

        # Expand back if sliced
        if use_matformer_slicing and not self.is_global:
            seqs = expand_from_slice(seqs, self.model_dim, seqs.device)

        return seqs
```

---

## 5.3: KV Cache Sharing

**File**: `src/fairseq2/nn/incremental_state.py` (extend)

### Shared KV Cache

```python
class SharedKVCache:
    """KV cache shared across multiple layers."""

    def __init__(self):
        self.shared_keys: Optional[torch.Tensor] = None
        self.shared_values: Optional[torch.Tensor] = None
        self.provider_layer_idx: Optional[int] = None

    def write(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor):
        """Write KV cache from a global layer."""
        self.shared_keys = keys
        self.shared_values = values
        self.provider_layer_idx = layer_idx

    def read(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read shared KV cache."""
        if self.shared_keys is None:
            raise RuntimeError("Shared KV cache not initialized")

        return self.shared_keys, self.shared_values
```

### Integration in Attention

```python
class Gemma3nAttention(MultiheadAttention):
    """Attention with KV cache sharing support."""

    def forward(
        self,
        queries,
        keys,
        values,
        shared_kv_cache: Optional[SharedKVCache] = None,
        is_provider_layer: bool = False,
    ):
        # Use shared KV if available
        if shared_kv_cache is not None and not is_provider_layer:
            keys, values = shared_kv_cache.read(self.layer_idx)

        # Standard attention
        output = self.sdpa(queries, keys, values, ...)

        # Write to shared cache if provider
        if shared_kv_cache is not None and is_provider_layer:
            shared_kv_cache.write(self.layer_idx, keys, values)

        return output
```

---

## 5.4: Tensor Parallelism

**File**: `src/fairseq2/models/gemma3n/sharder.py`

### Shard Specs

```python
from fairseq2.gang import Gang
from fairseq2.models.transformer.sharder import TransformerSharder

class Gemma3nSharder(TransformerSharder):
    """Tensor parallelism specs for Gemma3n."""

    def __init__(self, gang: Gang):
        super().__init__(gang)

    def get_shard_specs(self) -> dict:
        """Define sharding strategy for all parameters."""
        specs = {
            # Embeddings: replicated
            "decoder_frontend.embed.weight": None,

            # Attention projections
            "decoder.layers.*.self_attn.q_proj.weight": "column",
            "decoder.layers.*.self_attn.k_proj.weight": "column",
            "decoder.layers.*.self_attn.v_proj.weight": "column",
            "decoder.layers.*.self_attn.output_proj.weight": "row",

            # FFN projections
            "decoder.layers.*.ffn.gate_proj.weight": "column",
            "decoder.layers.*.ffn.inner_proj.weight": "column",
            "decoder.layers.*.ffn.output_proj.weight": "row",

            # AltUp FFN (local layers)
            "decoder.layers.*.altup_ffn.gate_proj.weight": "column",
            "decoder.layers.*.altup_ffn.inner_proj.weight": "column",
            "decoder.layers.*.altup_ffn.output_proj.weight": "row",

            # PLE experts: expert-parallel
            "ple_modules.*.experts_up.*.weight": "expert",
            "ple_modules.*.experts_down.*.weight": "expert",

            # Normalization: replicated
            "decoder.layers.*.self_attn_norm.weight": None,
            "decoder.layers.*.ffn_norm.weight": None,

            # Output projection
            "final_proj.weight": "column",
        }

        return specs
```

---

## 5.5: Activation Sparsity (Optional)

**File**: `src/fairseq2/models/gemma3n/ffn.py` (extend)

### Sparse Activation

```python
class SparseActivation(nn.Module):
    """Statistical top-k activation sparsity."""

    def __init__(self, top_k_ratio: float = 0.25):
        super().__init__()
        self.top_k_ratio = top_k_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply top-k sparsity."""
        if self.top_k_ratio >= 1.0:
            return x  # No sparsity

        # Compute k
        k = int(x.size(-1) * self.top_k_ratio)

        # Select top-k by magnitude
        topk_vals, topk_idx = torch.topk(x.abs(), k, dim=-1)

        # Create mask
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_idx, 1.0)

        return x * mask
```

---

## Commit Strategy for Phase 5

**Commit 1**: `[gemma3n] Add PLEModule implementation`
- Implement `PLEModule` in `src/fairseq2/models/gemma3n/ple.py`
- Expert routing with top-k selection
- ~250 LOC

**Commit 2**: `[gemma3n] Add PLE CPU offloading`
- Implement `PLEOffloadManager`
- Async GPU/CPU transfers
- ~150 LOC

**Commit 3**: `[gemma3n] Add MatFormer slicing`
- Implement `apply_matformer_slice()` in factory
- E2B/E4B dimension slicing logic
- ~200 LOC

**Commit 4**: `[gemma3n] Add KV cache sharing`
- Implement `SharedKVCache` in incremental_state
- Integrate with attention layers
- ~200 LOC

**Commit 5**: `[gemma3n] Add tensor parallelism specs`
- Implement `Gemma3nSharder` in `sharder.py`
- Column/row/expert sharding specs
- ~250 LOC

**Commit 6**: `[gemma3n] Add Phase 5 tests`
- Test PLE routing, MatFormer slicing, KV sharing
- Multi-GPU tests
- ~300 LOC

**Code Quality Check**:
- Run `/unslop-code` - remove over-engineering
- Run `/better-engineering` - final quality pass
- Commit: `[gemma3n] Phase 5 final cleanup and optimization`

**Total**: 6-7 commits, ~1350 LOC

---

## Deliverables for Phase 5

- [ ] PLEModule implemented
- [ ] CPU offloading working
- [ ] MatFormer slicing implemented
- [ ] KV cache sharing working
- [ ] Tensor parallelism specs defined
- [ ] Multi-GPU inference tested
- [ ] Multi-GPU training tested
- [ ] `/unslop-code` passed
- [ ] `/better-engineering` passed

---

## Final Verification

### Full Architecture Test

```python
def test_full_architecture():
    """Test all advanced features together."""

    config = Gemma3nConfig()
    config.use_ple = True
    config.use_matformer_slicing = True
    config.use_kv_sharing = True

    model = create_gemma3n_model(config)

    # Test forward pass
    input_ids = torch.randint(0, 256_128, (2, 128))
    output = model(input_ids)

    assert output.shape == (2, 128, 256_128)
```

---

## Next Steps After Phase 5

- [ ] Add vision encoder (MobileNet-V5)
- [ ] Add audio encoder (USM-based)
- [ ] Multimodal fusion
- [ ] Production optimizations
- [ ] Benchmarking and profiling
