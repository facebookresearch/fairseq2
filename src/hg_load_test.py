import torch
import torch.distributed as dist

from fairseq2.gang import ProcessGroupGang, create_parallel_gangs
from fairseq2.device import get_default_device
from fairseq2.nn import ddp as DDP
from fairseq2.assets import get_asset_store
from fairseq2.models.hg import get_hg_model_hub, get_hg_tokenizer_hub
from fairseq2.gang import maybe_get_current_gangs

import soundfile as sf

from qwen_omni_utils import process_mm_info

def main():

    world_size = torch.cuda.device_count()
    
    device = get_default_device()

    root_gang = ProcessGroupGang.create_default_process_group(device)

    print(f"Root gang: {root_gang}")
    
    gangs = create_parallel_gangs(root_gang, tp_size=world_size)

    process_group_gangs = gangs.tp.as_process_group()
    
    print(f"Process of rank {gangs.tp.rank}/{gangs.tp.size} has spawned")    

    card = get_asset_store().retrieve_card("hg_qwen25_omni_3b")
    if gangs.tp.rank == 0:
        model = get_hg_model_hub().load_model(card)
        
    else:
        process_group_gangs.barrier()
        model = 

    processor = model.processor
    
    print("Done!")
    
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

    USE_AUDIO_IN_VIDEO = True

    print(1)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    print(2)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    print(3)
    inputs = processor(text=text, audio=audios, image=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    print(4)
    inputs = DDP.to_ddp(inputs.to(model.dtype))
    print(5)
    text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    print(6)
    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(text)
    sf.write(
        "output.wav",
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )

    print(f"Process: {gangs.tp.rank} is now exiting successfully (exit code 0)")

    process_group_gangs.close()
    
if __name__ == "__main__":
    main()

