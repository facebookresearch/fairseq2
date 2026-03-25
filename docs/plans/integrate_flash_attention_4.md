# Plan: Integrate FlashAttention4 into Fairseq2

**Author:** Yunchao Yang
**Date:** 2026-03-24
**Status:** Draft

---

## Background

FlashAttention4 (FA4) is the latest iteration from Dao-AILab, implemented in **Python CuTeDSL** with JIT compilation. Unlike FA2 (C++/CUDA) and FA3 (C++/CUDA for Hopper), FA4 supports multiple GPU architectures (SM80/SM90/SM100/SM110/SM120) and introduces new features like `mask_mod`, block sparsity, and `score_mod`.

### FA Version Comparison

| Feature | FA2 | FA3 | FA4 |
|---------|-----|-----|-----|
| GPU Support | SM80+ (Ampere) | SM90+ (Hopper) | SM80/90/100/110/120 |
| Implementation | C++/CUDA | C++/CUDA | Python CuTeDSL (JIT) |
| Package Name (pip) | `flash-attn` | N/A (source compile) | `flash-attn-4` |
| Import Path | `from flash_attn import ...` | `import flash_attn_3_cuda` | `from flash_attn.cute import ...` |
| Dropout | ✅ | ❌ | ❌ |
| ALiBi | ✅ | ❌ | ❌ |
| FP8 | ❌ | ✅ | ❌ (planned) |
| Paged KV Cache | ❌ | ✅ | ✅ (SM100) |
| Block Sparsity | ❌ | ❌ | ✅ |
| mask_mod / score_mod | ❌ | ❌ | ✅ |
| torch.ops registration | ✅ | ✅ | ❌ (JIT compiled) |

### FA4 Actual API (from `/home/yunchaoyang1/flash-attention/flash_attn/cute/interface.py`)

```python
def flash_attn_func(
    q,                          # (batch, seqlen, nheads, headdim)
    k,                          # (batch, seqlen, nheads_kv, headdim)
    v,                          # (batch, seqlen, nheads_kv, headdim_v)
    softmax_scale=None,
    causal=False,
    window_size=(None, None),   # Note: uses None, not -1
    learnable_sink=None,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    mask_mod=None,
    return_lse=False,
) -> Tuple[Tensor, Tensor]:

def flash_attn_varlen_func(
    q,                          # (total_q, nheads, headdim)
    k,                          # (total_k, nheads_kv, headdim)
    v,                          # (total_k, nheads_kv, headdim_v)
    cu_seqlens_q=None,          # (batch+1,) int32
    cu_seqlens_k=None,          # (batch+1,) int32
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(None, None),
    ...
    return_lse=False,
) -> Tuple[Tensor, Tensor]:
```

---

## Fairseq2 SDPA Architecture

### Existing Class Hierarchy

```
SDPA (abstract base - base.py)
├── Flash2SDPA (flash2.py)     ← flash_attn package
├── Flash3SDPA (flash3.py)     ← flash_attn_3_cuda / flash_attn_3._C
├── TorchSDPA (torch.py)       ← torch.nn.functional.scaled_dot_product_attention
├── NaiveSDPA (naive.py)       ← pure Python reference
├── RelativePositionSDPA
└── ShawRelativePositionSDPA
```

### SDPA Base Interface (`base.py`)

```python
class SDPA(Module, ABC):
    @abstractmethod
    def forward(
        self,
        q: Tensor,              # (N, S, H, K)
        q_layout: BatchLayout,
        k: Tensor,              # (N, S_kv, H, K)
        k_layout: BatchLayout,
        v: Tensor,              # (N, S_kv, H, V)
        bias_cache: AttentionBiasCache,
        *,
        needs_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
```

### SDPAFactory Protocol (`default.py`)

```python
class SDPAFactory(Protocol):
    def __call__(self, bias: AttentionBias, *, dropout_p: float = 0.0) -> SDPA:
```

### SDPAVariant Configuration (`recipe/config.py`)

```python
SDPAVariant: TypeAlias = Literal[
    "torch", "torch_math", "torch_mem_efficient", "torch_flash",
    "flash2", "flash3", "naive",
]
```

---

## Implementation Plan

### Phase 1: Create `flash4.py`

**File:** `src/fairseq2/models/transformer/sdpa/flash4.py`

```python
from __future__ import annotations

from typing import final

import torch
from torch import Tensor
from typing_extensions import override

try:
    from flash_attn.cute import (  # type: ignore[import-not-found,import-untyped]
        flash_attn_func,
        flash_attn_varlen_func,
    )
except ImportError:
    _has_flash_attn_4 = False
else:
    _has_flash_attn_4 = True

from fairseq2.error import NotSupportedError, OperationalError
from fairseq2.models.transformer.attention_bias import (
    AttentionBias,
    AttentionBiasCache,
    CausalAttentionBias,
    IdentityBias,
)
from fairseq2.models.transformer.sdpa.base import SDPA
from fairseq2.nn import BatchLayout


@final
class Flash4SDPA(SDPA):
    """Computes scaled dot-product attention using FlashAttention4."""

    def __init__(self, bias: AttentionBias, *, dropout_p: float = 0.0) -> None:
        super().__init__()

        self.bias = bias
        self.dropout_p = dropout_p

    @override
    def forward(
        self,
        q: Tensor,
        q_layout: BatchLayout,
        k: Tensor,
        k_layout: BatchLayout,
        v: Tensor,
        bias_cache: AttentionBiasCache,
        *,
        needs_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if not _has_flash_attn_4:
            raise OperationalError(
                "FlashAttention4 is not found. Use `pip install flash-attn-4`."
            )

        if q_layout.padded or k_layout.padded:
            raise NotSupportedError(
                f"`{Flash4SDPA}` does not support padded batches."
            )

        # Determine causal mode and window size from bias
        if isinstance(self.bias, IdentityBias):
            causal = False
            window_size = (None, None)
        elif isinstance(self.bias, CausalAttentionBias):
            causal = True
            attn_window_len = self.bias.attn_window_len
            if attn_window_len is not None:
                window_size = (attn_window_len, None)
            else:
                window_size = (None, None)
        else:
            raise NotSupportedError(
                f"`{Flash4SDPA}` does not support `{self.bias}`."
            )

        # FA4 does not support dropout
        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout_p

        if dropout_p != 0.0:
            raise NotSupportedError(
                f"`{Flash4SDPA}` does not support dropout."
            )

        if q_layout.packed ^ k_layout.packed:
            raise ValueError("`q_layout` and `k_layout` must be both packed.")

        if q_layout.packed:
            attns, _ = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=q_layout.seq_begin_indices_pt,
                cu_seqlens_k=k_layout.seq_begin_indices_pt,
                max_seqlen_q=q_layout.max_seq_len,
                max_seqlen_k=k_layout.max_seq_len,
                causal=causal,
                window_size=window_size,
            )
        else:
            attns, _ = flash_attn_func(
                q,
                k,
                v,
                causal=causal,
                window_size=window_size,
            )

        return attns, None

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"bias={self.bias}, dropout_p={self.dropout_p:G}"
```

#### Key Design Decisions

1. **Import path**: `from flash_attn.cute import ...` — FA4 lives inside the `flash_attn` package as a submodule, not a separate package.

2. **No `torch.library.custom_op` wrappers needed**: Unlike FA3 which requires manual `register_fake` and `register_autograd` for `torch.compile` compatibility, FA4 uses CuTeDSL JIT and handles this internally. This makes `flash4.py` structurally closer to `flash2.py` than `flash3.py`.

3. **`window_size` uses `(None, None)`**: FA4 uses `None` to mean "no limit", different from FA2's `(-1, -1)` convention.

4. **Dropout rejected explicitly**: FA4 has no `dropout_p` parameter. We accept it in `__init__` (to satisfy `SDPAFactory` protocol) but raise `NotSupportedError` if non-zero at runtime.

5. **No autocast handling**: FA2's `flash2.py` manually handles `torch.is_autocast_enabled()`. FA4 with JIT compilation may handle dtype internally — needs verification during testing.

---

### Phase 2: Export from `__init__.py`

**File:** `src/fairseq2/models/transformer/sdpa/__init__.py`

Add:
```python
from fairseq2.models.transformer.sdpa.flash4 import Flash4SDPA as Flash4SDPA
```

**File:** `src/fairseq2/models/transformer/__init__.py`

Add `Flash4SDPA` to the exports alongside existing `Flash2SDPA`, `Flash3SDPA`.

---

### Phase 3: Add `"flash4"` to `SDPAVariant` config

**File:** `src/fairseq2/recipe/config.py`

```python
SDPAVariant: TypeAlias = Literal[
    "torch",
    "torch_math",
    "torch_mem_efficient",
    "torch_flash",
    "flash2",
    "flash3",
    "flash4",       # ← Add this
    "naive",
]
```

---

### Phase 4: Register in recipe dispatcher

**File:** `src/fairseq2/recipe/internal/torch.py`

Add import:
```python
from fairseq2.models.transformer import (
    Flash2SDPA,
    Flash3SDPA,
    Flash4SDPA,     # ← Add this
    NaiveSDPA,
    TorchSDPA,
    set_default_sdpa_factory,
)
```

Add match case:
```python
def _set_default_sdpa_variant(self) -> None:
    name = self._section.torch.default_sdpa

    match name:
        case "torch":
            set_default_sdpa_factory(TorchSDPA)
        case "torch_math" | "torch_mem_efficient" | "torch_flash":
            set_default_sdpa_factory(TorchSDPA)
            backend = name[6:]
            try:
                self._set_torch_sdpa_backend(backend)
            except (ImportError, AttributeError):
                log.warning("PyTorch SDPA kernel cannot be set to {}. Falling back to auto mode.", backend)
        case "flash2":
            set_default_sdpa_factory(Flash2SDPA)
        case "flash3":
            set_default_sdpa_factory(Flash3SDPA)
        case "flash4":                                  # ← Add this
            set_default_sdpa_factory(Flash4SDPA)        # ← Add this
        case "naive":
            set_default_sdpa_factory(NaiveSDPA)
        case _:
            raise ValueError(
                f"`name` must be a known SDPA variant, but is '{name}' instead."
            )
```

---

### Phase 5: Verify with YAML config

**Example config:**
```yaml
common:
  torch:
    default_sdpa: flash4
```

---

## Files Changed Summary

| File | Change |
|------|--------|
| `src/fairseq2/models/transformer/sdpa/flash4.py` | **New file** — `Flash4SDPA` class |
| `src/fairseq2/models/transformer/sdpa/__init__.py` | Export `Flash4SDPA` |
| `src/fairseq2/models/transformer/__init__.py` | Export `Flash4SDPA` |
| `src/fairseq2/recipe/config.py` | Add `"flash4"` to `SDPAVariant` |
| `src/fairseq2/recipe/internal/torch.py` | Add `case "flash4"` + import |

---

## Installation Requirements

```bash
# FA4 requires these dependencies:
pip install flash-attn-4
# Which pulls in:
#   - nvidia-cutlass-dsl>=4.4.2
#   - torch
#   - einops
#   - apache-tvm-ffi>=0.1.5,<0.2
#   - quack-kernels>=0.3.3

# FA4 is NOT listed in fairseq2's dependencies (optional, like FA2/FA3)
```

---

## Coexistence with FA2 and FA3

All three can coexist in the same environment:

| Package | Python Import | Conflict? |
|---------|--------------|-----------|
| `flash-attn` (FA2) | `flash_attn` | ❌ |
| `flash_attn_3_cuda` (FA3) | `flash_attn_3_cuda` | ❌ |
| `flash-attn-4` (FA4) | `flash_attn.cute` | ⚠️ See note below |

> **Note:** FA4 (`flash-attn-4`) installs into `flash_attn.cute`. If FA2 (`flash-attn`) is also installed, they share the `flash_attn` namespace. Verify that `flash-attn-4` properly extends the `flash_attn` package via namespace packages or that it depends on `flash-attn` as a base. If they conflict, FA4 may need to be installed **after** FA2.

---

## Open Questions

1. **Autocast handling**: FA2's `flash2.py` explicitly handles `torch.is_autocast_enabled()` to cast q/k/v to the autocast dtype. Does FA4's JIT handle this internally, or do we need the same logic?

2. **`torch.compile` compatibility**: FA4 uses CuTeDSL JIT, not `torch.library.custom_op`. Does `torch.compile` trace through FA4 correctly? The FA4 test suite uses `FLASH_ATTENTION_FAKE_TENSOR=1` env var, suggesting some special handling may be needed.

3. **FA2/FA4 namespace conflict**: Both `flash-attn` and `flash-attn-4` install into the `flash_attn` namespace. Need to verify they can coexist or if `flash-attn-4` supersedes `flash-attn`.

4. **First-call JIT latency**: FA4 JIT-compiles kernels on first call for each `(dtype, head_dim, arch)` combination. This could cause unexpected latency at training start. Should we add a warmup step or log a warning?

5. **SM architecture detection**: FA4 auto-selects kernel class (Sm80/Sm90/Sm100/Sm120) based on GPU. Do we need to expose architecture override in fairseq2 config?
