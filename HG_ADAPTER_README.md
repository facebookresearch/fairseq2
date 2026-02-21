# HuggingFace Model Adapter for fairseq2 Training

## What We Built

There is now an automatic adapter system that enables **any HuggingFace causal language model** (GPT-2, Llama, Gemma, etc.) to be trained using fairseq2's training recipes (like SFT), which previously only supported native fairseq2 models.

## Key Components

### 1. `HgCausalLMAdapter` (adapter.py)
A wrapper class that:
- Inherits from `fairseq2.models.clm.CausalLM`
- Wraps any HuggingFace `PreTrainedModel`
- Translates between the two different APIs:
  - **fairseq2 API**: `model(seqs, seqs_layout, targets=...)`
  - **HuggingFace API**: `model(input_ids=..., attention_mask=..., labels=...)`
- Handles attention mask creation from fairseq2's `BatchLayout`
- Supports FSDP by delegating module traversal to the wrapped HF model

### 2. Factory Integration (factory.py)
Modified `create_hg_model()` to automatically wrap causal LM models:
- After loading a HuggingFace model
- Checks if `model_type == "causal_lm"`
- Automatically wraps it in `HgCausalLMAdapter`
- Returns a fairseq2-compatible `CausalLM` instance

### 3. Model Family Registration (composition/models.py)
Updated the HG family registration to:
- Use `Module` as the base class (instead of `PreTrainedModel`)
- This allows the family to return either wrapped or unwrapped models

## How It Works

```
User Config (YAML)
    ↓
fairseq2 loads model with family="hg", model_type="causal_lm"
    ↓
create_hg_model() loads HuggingFace model
    ↓
wrap_hg_model_if_causal_lm() wraps it in HgCausalLMAdapter
    ↓
Returns fairseq2 CausalLM (compatible with SFT recipe)
    ↓
FSDP wrapping (if enabled) - delegates to HF model's layers
    ↓
Training with fairseq2 SFT recipe ✓
```

## Usage

### Basic Configuration

```yaml
model:
  name: null  # IMPORTANT: Disable default model
  family: "hg"
  arch: "causal_lm"
  config_overrides:
    hf_name: "google/gemma-3-2b-it"  # Any HF causal LM
    model_type: "causal_lm"
    trust_remote_code: true

tokenizer:
  family: "hg"
  config_overrides:
    hf_name: "google/gemma-3-2b-it"
```

### Running Training

```bash
python -m recipes.lm.sft \
  --config-file recipes/lm/sft/configs/gemma_3_2b_it_gsm8k.yaml \
  ./out
```

## Supported Models

Any HuggingFace model that:
- Is a causal language model (decoder-only)
- Can be loaded with `AutoModelForCausalLM.from_pretrained()`
- Examples:
  - Gemma (google/gemma-*)
  - Llama (meta-llama/*)
  - GPT-2 (gpt2, gpt2-large, etc.)
  - Qwen (Qwen/*)
  - Mistral (mistralai/*)
  - And many more...

## What Gets Fixed

### Before (Broken)
```
ERROR: Model must be of type `CausalLM`,
       but is of type `FSDPGemma3ForCausalLM` instead.
```

### After (Working)
```
✓ HuggingFace model loaded
✓ Wrapped in HgCausalLMAdapter (implements CausalLM)
✓ Compatible with fairseq2 SFT recipe
✓ FSDP wrapping works correctly
✓ Training proceeds normally
```

## Important Notes

1. **Set `name: null`**: Must disable the default model name to use family/arch approach
2. **Device placement**: HF models are loaded to `gangs.root.device` by default
3. **Meta device**: Not supported for HF models (`supports_meta=False`)
4. **FSDP**: Automatically wraps transformer layers using `_no_split_modules`

## API Compatibility

The adapter handles all the translation:

| fairseq2 Input | HuggingFace Input |
|----------------|-------------------|
| `seqs` | `input_ids` |
| `seqs_layout.padding_mask` | `attention_mask` |
| `targets` | `labels` |
| `target_mask` | Applied to labels (-100 for ignored) |

| fairseq2 Output | HuggingFace Output |
|-----------------|-------------------|
| `loss` (scalar) | `outputs.loss` (with reduction applied) |
| `logits` (optional) | `outputs.logits` |

## Testing

To verify the integration works:

```bash
# 1. Check the model loads and wraps correctly
python -c "
from fairseq2.models.hg_qwen_omni import create_hg_model
from fairseq2.models.hg_qwen_omni.config import HuggingFaceModelConfig
from fairseq2.models.clm import CausalLM

config = HuggingFaceModelConfig(
    hf_name='gpt2',
    model_type='causal_lm'
)
model = create_hg_model(config)
print(f'Model type: {type(model).__name__}')
print(f'Is CausalLM: {isinstance(model, CausalLM)}')
"

# 2. Run actual training
python -m recipes.lm.sft \
  --config-file recipes/lm/sft/configs/gemma_3_2b_it_gsm8k.yaml \
  ./out
```

## Files Modified

1. **New**: `src/fairseq2/models/hg_qwen_omni/adapter.py` - The adapter class
2. **Modified**: `src/fairseq2/models/hg_qwen_omni/factory.py` - Automatic wrapping
3. **Modified**: `src/fairseq2/models/hg_qwen_omni/__init__.py` - Exports
4. **Modified**: `src/fairseq2/composition/models.py` - Model family registration
5. **New**: `recipes/lm/sft/configs/gemma_3_2b_it_gsm8k.yaml` - Example config
