### wav2vec2 Recipes

This directory contains two distinct wav2vec2-based recipes:

`ssl/` - **Self-Supervised Pretraining**

Implements wav2vec2 self-supervised pretraining on unlabeled audio data using contrastive learning. The model learns audio representations by predicting masked portions of the input audio. Uses SequenceBatch for audio-only processing and targets the diversity loss and feature penalty from the original wav2vec2 paper.

`asr/` - **Automatic Speech Recognition Fine-tuning**

Implements wav2vec2 fine-tuning for automatic speech recognition using CTC loss on labeled audio-text pairs. Loads pretrained wav2vec2 weights, adds a linear projection layer for character/subword prediction, and trains with encoder freezing schedules. Uses Seq2SeqBatch for audio-text pair processing and requires both audio files (.flac/.wav) and transcription files (.wrd).

**Code Differences**

While both recipes share similar audio processing pipelines (normalization, batching), they are clearer delineated with separate implementations due to:
- Different data formats (audio-only TSV vs audio+text TSV+WRD files)
- Different length batching (min/max cropping vs filtering by max audio length)
- Different batch types (SequenceBatch vs Seq2SeqBatch)
- Different loss functions (contrastive vs CTC)
- Different model architectures (encoder-only vs encoder+projection)

**Code Organization & How To Run**

```bash
export OUTPUT_DIR=/my/path/to/artefact/dir

cd wav2vec2

python -m ssl --config-file ssl/configs/my_config.yaml $OUTPUT_DIR

python -m ssl.eval --config-file ssl/eval/configs/my_eval_config.yaml $OUTPUT_DIR

python -m ssl.eval --config dataset.valid_split="test_clean,test_other"


python -m asr --config-file asr/configs/my_config.yaml $OUTPUT_DIR

python -m asr.eval --config-file asr/eval/configs/my_eval_config.yaml $OUTPUT_DIR
```

The idea is to keep the main recipe coupled with the evaluation runner but share the dataset implementation.
