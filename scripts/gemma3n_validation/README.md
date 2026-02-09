# Gemma3n Validation Scripts

Test scripts used during Phase 2 & 3 development. Run these on a compute node with GPU access.

## Setup
All scripts use: `/home/aerben/repos/fairseq2/.venv/bin/python`

## Key Scripts

### Model Validation
- **test_parity.sh** - Quick 2-layer model test (validates model creation & forward pass)
- **test_e2b_config.sh** - Verify E2B config matches HuggingFace exactly

### Checkpoint Conversion
- **show_conversion.sh** - Demo HF→fairseq2 key conversion
- **show_checkpoint_keys.sh** - Inspect HF checkpoint structure
- **validate_lang_conversion.sh** - Validate language_model key conversion

### Config Discovery
- **check_gemma_models.sh** - List available Gemma3n models on HuggingFace
- **get_text_config.sh** - Extract text_config from HF Gemma3n config
- **discover_gemma3n_config.sh** - Show all HF config attributes

## Results

✓ E2B config matches HuggingFace (30 layers, 2.05B params)
✓ Model creates and runs on GPU (8x H100)
✓ Checkpoint conversion working for basic transformer
⚠ Advanced features (LAuReL, PLE, AltUp) deferred to Phase 4/5
