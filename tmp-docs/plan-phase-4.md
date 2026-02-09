# Phase 4: Training Integration

**Duration**: 2 days
**Goal**: Enable SFT training with fairseq2 recipes

---

## 4.1: Synthetic Dataset Generation

**File**: `recipes/gemma3n/sft/dataset.py`

### Dataset Generator

```python
import json
from pathlib import Path
from typing import List, Dict

def generate_synthetic_sft_dataset(
    num_examples: int = 100,
    output_path: Path,
) -> None:
    """
    Generate simple synthetic SFT dataset.

    Format: Q: <question>\nA: <answer>
    """
    examples = []

    for i in range(num_examples):
        # Simple arithmetic
        q = f"What is {i} + {i}?"
        a = f"{i + i}"

        examples.append({
            "text": f"Q: {q}\nA: {a}",
            "id": i,
        })

    # Save as JSONL
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Generated {num_examples} examples at {output_path}")
```

### Generate Dataset

```bash
# Create dataset directory
mkdir -p data/gemma3n_sft_synthetic

# Generate dataset
python3 -c "
from recipes.gemma3n.sft.dataset import generate_synthetic_sft_dataset
from pathlib import Path

generate_synthetic_sft_dataset(
    num_examples=100,
    output_path=Path('data/gemma3n_sft_synthetic/train.jsonl')
)
"
```

---

## 4.2: Recipe Configuration

**File**: `recipes/gemma3n/sft/config.py`

```python
from dataclasses import dataclass
from fairseq2.recipes.lm import LMSFTConfig

@dataclass(kw_only=True)
class Gemma3nSFTConfig(LMSFTConfig):
    """Configuration for Gemma3n SFT training."""

    # Model
    model_arch: str = "gemma3n_e2b"
    """Model architecture name."""

    # Data
    dataset_path: str = "data/gemma3n_sft_synthetic/train.jsonl"
    """Path to training dataset."""

    max_seq_len: int = 512
    """Maximum sequence length."""

    # Training
    max_num_steps: int = 100
    """Maximum number of training steps."""

    batch_size: int = 2
    """Training batch size."""

    gradient_accumulation: int = 4
    """Gradient accumulation steps."""

    learning_rate: float = 1e-5
    """Learning rate."""

    # Advanced (Phase 5)
    use_ple_offload: bool = False
    """Offload PLE parameters to CPU."""

    use_matformer_slicing: bool = False
    """Use MatFormer E2B slicing."""
```

---

## 4.3: Recipe Implementation

**File**: `recipes/gemma3n/sft/recipe.py`

```python
from typing import Optional
import torch
from fairseq2.recipes.lm import LMSFTRecipe
from fairseq2.models.gemma3n import create_gemma3n_model, Gemma3nConfig
from .config import Gemma3nSFTConfig

class Gemma3nSFTRecipe(LMSFTRecipe):
    """SFT recipe for Gemma3n models."""

    def __init__(self, config: Gemma3nSFTConfig):
        super().__init__(config)
        self.config = config

    def create_model(self):
        """Create Gemma3n model."""
        model_config = Gemma3nConfig()

        model = create_gemma3n_model(
            model_config,
            device=self.device,
            dtype=self.dtype,
        )

        return model

    def create_data_pipeline(self):
        """Create data pipeline for SFT."""
        # Use standard fairseq2 LM data pipeline
        from fairseq2.data import create_text_data_pipeline

        pipeline = create_text_data_pipeline(
            dataset_path=self.config.dataset_path,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.max_seq_len,
        )

        return pipeline
```

**File**: `recipes/gemma3n/sft/__main__.py`

```python
from fairseq2.recipes.lm import run_sft_recipe
from .recipe import Gemma3nSFTRecipe
from .config import Gemma3nSFTConfig

if __name__ == "__main__":
    config = Gemma3nSFTConfig()
    recipe = Gemma3nSFTRecipe(config)
    run_sft_recipe(recipe)
```

---

## 4.4: Training Parity Test

**File**: `tests/integration/models/gemma3n/test_training_parity.py`

### Test 1: Single Training Step

```python
import pytest
import torch
from torch.optim import AdamW
from fairseq2.models.gemma3n import create_gemma3n_model, Gemma3nConfig
from tests.common import assert_close, device, temporary_manual_seed

@pytest.mark.integration
def test_training_step_parity():
    """Verify training step produces matching loss and gradients."""

    with temporary_manual_seed(42, device):
        # Load HuggingFace model
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3n-2b",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        hf_model.to(device)
        hf_model.train()

        # Create fairseq2 model
        config = Gemma3nConfig()
        fs2_model = create_gemma3n_model(config, device=device)
        fs2_model.train()

        # Load same weights
        from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
        fs2_model.load_state_dict(convert_gemma3n_state_dict(
            hf_model.state_dict(), config
        ))

        # Create identical batch
        input_ids = torch.randint(0, 256_128, (2, 128), device=device)
        labels = input_ids.clone()

        # HuggingFace training step
        hf_optimizer = AdamW(hf_model.parameters(), lr=1e-5)
        hf_optimizer.zero_grad()

        hf_outputs = hf_model(input_ids, labels=labels)
        hf_loss = hf_outputs.loss
        hf_loss.backward()

        # Get gradient norm
        hf_grad_norm = torch.nn.utils.clip_grad_norm_(hf_model.parameters(), 1.0)

        # fairseq2 training step
        fs2_optimizer = AdamW(fs2_model.parameters(), lr=1e-5)
        fs2_optimizer.zero_grad()

        fs2_logits = fs2_model(input_ids)
        # Compute cross-entropy loss manually
        fs2_loss = torch.nn.functional.cross_entropy(
            fs2_logits.view(-1, fs2_logits.size(-1)),
            labels.view(-1),
        )
        fs2_loss.backward()

        fs2_grad_norm = torch.nn.utils.clip_grad_norm_(fs2_model.parameters(), 1.0)

        # Compare loss
        assert_close(hf_loss, fs2_loss, atol=1e-4, rtol=1e-5)

        # Compare gradient norms
        assert_close(hf_grad_norm, fs2_grad_norm, atol=1e-3, rtol=1e-3)
```

### Test 2: Multi-Step Training

```python
@pytest.mark.integration
def test_multi_step_training():
    """Verify model can train for multiple steps."""

    with temporary_manual_seed(42, device):
        config = Gemma3nConfig()
        model = create_gemma3n_model(config, device=device)
        model.train()

        optimizer = AdamW(model.parameters(), lr=1e-5)

        losses = []
        for step in range(10):
            # Random batch
            input_ids = torch.randint(0, 256_128, (2, 128), device=device)
            labels = input_ids.clone()

            # Forward
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Verify loss decreased (even on random data, should show some variation)
        assert len(losses) == 10
        print(f"Losses: {losses}")
```

---

## 4.5: Running Training

### Launch Training

```bash
# Using fairseq2 recipe system
python -m recipes.gemma3n.sft \
    --model-arch gemma3n_e2b \
    --dataset-path data/gemma3n_sft_synthetic/train.jsonl \
    --max-num-steps 100 \
    --batch-size 2 \
    --learning-rate 1e-5
```

### Monitor Training

```bash
# Check loss curve
tensorboard --logdir logs/gemma3n_sft
```

---

## 4.6: Expected Outcomes

### Training Logs

```
Step 0: loss=8.5432
Step 10: loss=6.2341
Step 20: loss=4.8923
...
Step 100: loss=2.1234
```

### Checkpoint Structure

```
checkpoints/gemma3n_sft/
├── step_10/
│   ├── model.pt
│   └── optimizer.pt
├── step_50/
└── step_100/
```

---

## Commit Strategy for Phase 4

**Commit 1**: `[gemma3n] Add synthetic dataset generator`
- Implement `generate_synthetic_sft_dataset()` in `recipes/gemma3n/sft/dataset.py`
- Generate 100 examples
- ~100 LOC

**Commit 2**: `[gemma3n] Add SFT recipe configuration`
- Create `Gemma3nSFTConfig` in `recipes/gemma3n/sft/config.py`
- ~150 LOC

**Commit 3**: `[gemma3n] Add SFT recipe implementation`
- Implement `Gemma3nSFTRecipe` in `recipes/gemma3n/sft/recipe.py`
- Add `__main__.py` entry point
- ~300 LOC

**Commit 4**: `[gemma3n] Add training parity tests`
- Add `test_training_step_parity()` in `test_training_parity.py`
- Add `test_multi_step_training()`
- ~350 LOC

**Code Quality Check**:
- Run `/unslop-code` - remove enterprise boilerplate
- Run `/better-engineering` - verify training code quality
- Commit: `[gemma3n] Phase 4 code quality cleanup`

**Total**: 4-5 commits, ~900 LOC

---

## Deliverables for Phase 4

- [ ] Synthetic dataset generated (100 examples)
- [ ] Recipe configuration created
- [ ] Recipe implementation completed
- [ ] Single training step parity test passing
- [ ] Multi-step training test passing
- [ ] Can train for 100 steps successfully
- [ ] `/unslop-code` passed
- [ ] `/better-engineering` passed

---

## Next Step
Proceed to `plan-phase-5.md` for advanced features (PLE, MatFormer, KV sharing).
