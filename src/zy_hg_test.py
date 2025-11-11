import platform
import os
import subprocess

import torch

import soundfile as sf

from fairseq2.assets import get_asset_store
from fairseq2.models.hg import get_hg_model_hub, get_hg_tokenizer_hub

from qwen_omni_utils import process_mm_info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

card = get_asset_store().retrieve_card("hg_qwen25_omni_3b")
model = get_hg_model_hub().load_model(card)

model = model.to(device)
processor = model.processor

# Be sure to download draw.mp4 first! Will not download from the chat template as is
if not os.path.exists('./draw.mp4'):
    url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"
    output_filename = "draw.mp4"
    if platform.system() == "Linux":
        subprocess.run(["wget", "-O", output_filename, url])
    else:
        subprocess.run(["curl", "-o", output_filename, url])

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "./draw.mp4"},
        ],
    },
]

# set use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)
