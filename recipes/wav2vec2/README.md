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

**Datasets**

Both recipes use a manifest-based dataset reader which expects to find a `{split}.tsv` file under their `data: ` field in their asset definition. `ssl` uses librispeech and `asr` uses the librilight (10h) dataset. Librilight expects to find a `{split}.wrd` file with transcription of each file and is positionally synchronized (text in line 21 in `train.wrd` corresponds to the audio file in the `train.tsv` manifest).

Check the dataset implementation for more details.

For minimal reproduction, add a path in the librilight.yaml and librispeech.yaml asset cards that point to their manifests, e.g.:

```yaml
name: librilight_asr_10h
dataset_family: wav2vec2_asr
dataset_config:
  data: "/my/path/to/my/librilight/manifest/train.tsv"
```

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
