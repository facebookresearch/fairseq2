# HG load test

import time

import subprocess

import torch

import torch.distributed as dist

from fairseq2.assets import get_asset_store
from fairseq2.device import get_default_device
from fairseq2.gang import (
    ProcessGroupGang,
    create_parallel_gangs,
)
from fairseq2.models.hg_qwen_omni import get_hg_model_hub

import soundfile as sf

from qwen_omni_utils import process_mm_info

def main():
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
        device = get_default_device()
        root_gang = ProcessGroupGang.create_default_process_group(device)
        gangs = create_parallel_gangs(root_gang, tp_size=world_size//2)

        card = get_asset_store().retrieve_card("hg_qwen25_omni_3b")
        root_gang.barrier()
        model = get_hg_model_hub().load_model(card, gangs=gangs)
        # Let model load
        root_gang.barrier()

        processor = model.processor

        output_directory = './'
        url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"
        subprocess.run(["wget", "-nc", "-P", output_directory, url])
        # Let file download
        root_gang.barrier()
        
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

        # if not gangs or gangs.tp.rank == 0:
        if True:
            device = gangs.tp.device if gangs else "cpu"
            USE_AUDIO_IN_VIDEO = True
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = processor(text=text, audio=audios, image=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO).to(model.device).to(model.dtype)
            print(f"Tensor located on device {device}")
            print(f"Model located on device {device}")
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
            if gangs:
                print(f"Running on {gangs.tp.size} GPUs took: {t_end - t_start} seconds.")
            else:
                print(f"Running on cpu took: {t_end - t_start} seconds.")

            gangs.close()
    else:
        print("Must run with torchrun and at least 2 GPUs. Exiting...")

if __name__ == "__main__":
    main()
