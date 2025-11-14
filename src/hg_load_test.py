import os
import subprocess

import torch
import torch.nn as nn
import torch.distributed as dist

import time

from fairseq2.device import get_default_device
from fairseq2.assets import get_asset_store
from fairseq2.models.hg import get_hg_model_hub, get_hg_tokenizer_hub
from fairseq2.gang import Gang, Gangs, FakeGang, ProcessGroupGang, create_parallel_gangs, maybe_get_current_gangs, create_fsdp_gangs

import soundfile as sf

from qwen_omni_utils import process_mm_info

def main():

    world_size = torch.cuda.device_count()
    device = get_default_device()
    root_gang = ProcessGroupGang.create_default_process_group(device)
    gangs = create_parallel_gangs(root_gang, tp_size=world_size)

    card = get_asset_store().retrieve_card("hg_qwen25_omni_3b")
    model = get_hg_model_hub().load_model(card, gangs=gangs)
    dist.barrier()
    
    processor = model.processor

    if not os.path.exists("./draw.mp4"):
        url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"
        output_filename = "draw.mp4"
        subprocess.run(["wget", "-O", output_filename, url])
    
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

    if gangs.tp.rank == 0:
        print(gangs.tp.device)
        USE_AUDIO_IN_VIDEO = True
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(text=text, audio=audios, image=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO).to(gangs.tp.device).to(model.dtype)
        print(f"Tensor located on device {device}")
        t_start = time.perf_counter()
        text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        print("Done!")
        t_end = time.perf_counter()
        text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(text)
        sf.write(
            "output.wav",
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )
        print(f"Running on {gangs.tp.size} GPUs took: {t_end - t_start} seconds.")
    dist.barrier()

    print(f"Process: {gangs.tp.rank} is now exiting successfully (exit code 0)")

    gangs.close()
    
if __name__ == "__main__":
    main()

