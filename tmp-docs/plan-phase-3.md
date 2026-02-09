# Phase 3: Inference Parity Testing

**Duration**: 2 days
**Goal**: Verify fairseq2 Gemma3n produces identical outputs to HuggingFace

---

## 3.1: Test Infrastructure Setup

### Directory Structure

```
tests/integration/models/gemma3n/
├── __init__.py
├── test_components.py        # Component-level tests
├── test_inference_parity.py  # Full model parity
└── fixtures/
    └── reference_outputs.pt   # HF reference outputs
```

### Test Utilities

**File**: `tests/integration/models/gemma3n/__init__.py`

```python
import torch
from typing import Tuple

def load_hf_gemma3n(model_name: str = "google/gemma-3n-2b"):
    """Load HuggingFace Gemma3n model."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    return model


def create_test_inputs(
    vocab_size: int = 256_128,
    batch_size: int = 2,
    seq_len: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic test inputs."""
    torch.manual_seed(42)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    return input_ids, attention_mask
```

---

## 3.2: Component Tests

**File**: `tests/integration/models/gemma3n/test_components.py`

### Test 1: Dual RoPE Frequencies

```python
import pytest
import torch
from fairseq2.nn.position_encoder import DualRotaryEncoder
from tests.common import assert_close, device

def test_dual_rope_encoding():
    """Verify DualRotaryEncoder produces correct frequency values."""
    encoder = DualRotaryEncoder(
        encoding_dim=256,
        max_seq_len=8192,
        theta=10_000.0,
        dual_theta=100_000.0,
        device=device,
    )

    # Test inputs
    batch_size, num_heads, seq_len, head_dim = 2, 16, 128, 256
    seqs = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # Apply encoding
    output = encoder(seqs)

    # Verify shape
    assert output.shape == seqs.shape

    # TODO: Load HF reference and compare
    # hf_output = load_hf_rope_output(...)
    # assert_close(output, hf_output, atol=1e-6, rtol=1e-6)
```

### Test 2: Soft-Capped Attention

```python
def test_soft_capped_attention():
    """Verify soft-capping produces correct attention values."""
    from fairseq2.nn.transformer import SoftCappedSDPA, TorchSDPA

    base_sdpa = TorchSDPA(attn_dropout_p=0.0)
    soft_capped = SoftCappedSDPA(base_sdpa, soft_cap=30.0)

    # Test inputs
    batch, heads, seq_len, head_dim = 2, 8, 128, 64
    q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device)

    # Forward pass
    output, weights = soft_capped(q, k, v, needs_weights=True)

    # Verify shapes
    assert output.shape == (batch, heads, seq_len, head_dim)

    # TODO: Compare with HF
```

### Test 3: AltUp FFN

```python
def test_altup_ffn():
    """Verify AltUp FFN matches HuggingFace output."""
    from fairseq2.models.transformer.ffn import AltUpFeedForwardNetwork

    ffn = AltUpFeedForwardNetwork(
        model_dim=2048,
        altup_hidden_dim=5376,
        bias=False,
        device=device,
    )

    # Test inputs
    seqs = torch.randn(2, 128, 2048, device=device)
    output = ffn(seqs)

    # Verify shape
    assert output.shape == seqs.shape

    # TODO: Compare with HF AltUp output
```

---

## 3.3: Full Model Parity Test

**File**: `tests/integration/models/gemma3n/test_inference_parity.py`

### Test 1: fp32 Parity

```python
import pytest
import torch
from fairseq2.models.gemma3n import create_gemma3n_model, Gemma3nConfig
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from tests.common import assert_close, device

@pytest.mark.integration
def test_full_model_parity_fp32():
    """Test full model parity in fp32."""

    # Load HuggingFace model
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3n-2b",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    hf_model.to(device)
    hf_model.eval()

    # Create fairseq2 model
    config = Gemma3nConfig()  # Default E2B config
    fs2_model = create_gemma3n_model(config, device=device, dtype=torch.float32)
    fs2_model.eval()

    # Convert and load HF checkpoint
    fs2_state_dict = convert_gemma3n_state_dict(
        hf_model.state_dict(), config
    )
    fs2_model.load_state_dict(fs2_state_dict)

    # Create test inputs
    torch.manual_seed(42)
    input_ids = torch.randint(0, 256_128, (2, 128), device=device)

    # HuggingFace forward pass
    with torch.no_grad():
        hf_outputs = hf_model(input_ids)
        hf_logits = hf_outputs.logits

    # fairseq2 forward pass
    with torch.no_grad():
        fs2_logits = fs2_model(input_ids)

    # Compare logits
    assert_close(hf_logits, fs2_logits, atol=1e-4, rtol=1e-5)
```

### Test 2: bf16 Parity

```python
@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_full_model_parity_bf16():
    """Test full model parity in bf16."""

    # Load HuggingFace model
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3n-2b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    hf_model.to(device)
    hf_model.eval()

    # Create fairseq2 model
    config = Gemma3nConfig()
    fs2_model = create_gemma3n_model(config, device=device, dtype=torch.bfloat16)
    fs2_model.eval()

    # Convert and load checkpoint
    fs2_state_dict = convert_gemma3n_state_dict(
        hf_model.state_dict(), config
    )
    fs2_model.load_state_dict(fs2_state_dict)

    # Test inputs
    torch.manual_seed(42)
    input_ids = torch.randint(0, 256_128, (2, 128), device=device)

    # Forward passes
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        fs2_logits = fs2_model(input_ids)

    # Compare with relaxed tolerance for bf16
    assert_close(hf_logits, fs2_logits, atol=1e-2, rtol=1e-3)
```

### Test 3: Incremental Decoding

```python
@pytest.mark.integration
def test_incremental_decoding_parity():
    """Test that KV cache produces consistent outputs."""

    config = Gemma3nConfig()
    model = create_gemma3n_model(config, device=device, dtype=torch.float32)
    model.eval()

    # Test inputs
    torch.manual_seed(42)
    input_ids = torch.randint(0, 256_128, (1, 100), device=device)

    # Full forward pass (no cache)
    with torch.no_grad():
        full_logits = model(input_ids)

    # Incremental forward pass (with cache)
    from fairseq2.nn.incremental_state import IncrementalStateBag

    state_bag = IncrementalStateBag(max_num_steps=100)

    with torch.no_grad():
        incremental_logits = []
        for i in range(input_ids.size(1)):
            seq = input_ids[:, i : i + 1]
            logits = model(seq, state_bag=state_bag)
            incremental_logits.append(logits)
            state_bag.increment_step_nr()

    incremental_logits = torch.cat(incremental_logits, dim=1)

    # Compare
    assert_close(full_logits, incremental_logits, atol=1e-5, rtol=1e-6)
```

---

## 3.4: Running Tests

### Run All Tests
```bash
pytest tests/integration/models/gemma3n/ -v
```

### Run Component Tests Only
```bash
pytest tests/integration/models/gemma3n/test_components.py -v
```

### Run Parity Tests on GPU
```bash
pytest tests/integration/models/gemma3n/test_inference_parity.py --device cuda:0 -v
```

### Run with Coverage
```bash
pytest tests/integration/models/gemma3n/ --cov=fairseq2.models.gemma3n
```

---

## 3.5: Debugging Parity Failures

### Layer-by-Layer Comparison

```python
def debug_layer_outputs(hf_model, fs2_model, input_ids):
    """Compare hidden states at each layer."""

    # HF forward with output_hidden_states
    hf_outputs = hf_model(input_ids, output_hidden_states=True)
    hf_hidden_states = hf_outputs.hidden_states

    # FS2 forward (need to hook layers)
    fs2_hidden_states = []

    def hook_fn(module, input, output):
        fs2_hidden_states.append(output[0].detach())

    hooks = []
    for layer in fs2_model.decoder.layers:
        hooks.append(layer.register_forward_hook(hook_fn))

    _ = fs2_model(input_ids)

    for hook in hooks:
        hook.remove()

    # Compare layer by layer
    for i, (hf_hidden, fs2_hidden) in enumerate(zip(hf_hidden_states, fs2_hidden_states)):
        try:
            assert_close(hf_hidden, fs2_hidden, atol=1e-4, rtol=1e-5)
            print(f"Layer {i}: ✓ Match")
        except AssertionError as e:
            print(f"Layer {i}: ✗ Mismatch - {e}")
            break
```

---

## Commit Strategy for Phase 3

**Commit 1**: `[gemma3n] Add test infrastructure`
- Create test directory structure
- Add test utilities (`load_hf_gemma3n`, `create_test_inputs`)
- ~100 LOC

**Commit 2**: `[gemma3n] Add component parity tests`
- Add `test_components.py` with all component tests
- Tests for DualRoPE, SoftCapping, AltUp
- ~300 LOC

**Commit 3**: `[gemma3n] Add fp32 full model parity test`
- Add `test_full_model_parity_fp32()`
- Implement HF checkpoint loading and conversion
- ~200 LOC

**Commit 4**: `[gemma3n] Add bf16 parity test`
- Add `test_full_model_parity_bf16()`
- Adjust tolerances for bf16
- ~100 LOC

**Commit 5**: `[gemma3n] Add incremental decoding test`
- Add `test_incremental_decoding_parity()`
- Verify KV cache consistency
- ~150 LOC

**Code Quality Check**:
- Run `/unslop-code` - remove vacuous tests, tutorial comments
- Run `/better-engineering` - ensure behavior-focused tests
- Commit: `[gemma3n] Phase 3 test quality improvements`

**Total**: 5-6 commits, ~850 LOC tests

---

## Deliverables for Phase 3

- [ ] Test infrastructure setup
- [ ] Component tests written and passing
- [ ] fp32 parity test passing
- [ ] bf16 parity test passing
- [ ] Incremental decoding test passing
- [ ] Debugging utilities implemented
- [ ] `/unslop-code` passed
- [ ] `/better-engineering` passed

---

## Next Step
Proceed to `plan-phase-4.md` for training integration.
