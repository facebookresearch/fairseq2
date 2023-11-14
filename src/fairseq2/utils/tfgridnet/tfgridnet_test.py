import torch
import torchaudio

from fairseq2.utils.tfgridnet.enh_inference import SeparateSpeech

separate_speech = {}

#Replace the config-yaml file and model.pth file with your path in next two lines
enh_model_sc = SeparateSpeech(
  train_config="../config.yaml",
  model_file="../5epoch.pth",
  # for segment-wise process on long speech
  normalize_segment_scale=False,
  show_progressbar=True,
  ref_channel=4,
  normalize_output_wav=True,
)


input = "../input.wav"
output = "../enhanced_audio.wav"

def use_torch_audio():
# model = master64().to("cpu")
   waveform, sr = torchaudio.load(input)
 

   waveform = waveform.to("cpu")
   estimate = enh_model_sc(waveform)
   estimate = torch.tensor(estimate)
   torchaudio.save(output, estimate[0], 16_000, encoding="PCM_S", bits_per_sample=16)

use_torch_audio()
