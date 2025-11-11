import torch
import torch.distributed as dist

from fairseq2.device import get_default_device
from fairseq2.assets import get_asset_store
from fairseq2.models.hg import get_hg_model_hub, get_hg_tokenizer_hub
from fairseq2.gang import Gangs, FakeGang, ProcessGroupGang, create_parallel_gangs, maybe_get_current_gangs

import soundfile as sf

from qwen_omni_utils import process_mm_info

def main():

    world_size = torch.cuda.device_count()
    device = get_default_device()
    root_gang = ProcessGroupGang.create_default_process_group(device)

    main_gangs = root_gang.create_gang(ranks=list(range(world_size)))
    mismatch_gangs = root_gang.create_gang(ranks=[0, 1])
    
    gangs = Gangs(root=root_gang, tp=main_gangs, dp=mismatch_gangs, sdp=FakeGang(torch.device("cpu")), rdp=FakeGang(torch.device("cpu")), pp=FakeGang(torch.device("cpu")))

    card = get_asset_store().retrieve_card("hg_qwen25_omni_3b")
    model = get_hg_model_hub().load_model(card, gangs=gangs)

    """
    if torch.cuda.is_available():
    
        world_size = torch.cuda.device_count()
        print(f"World size: {world_size}")
        all_ranks = list(range(world_size))
    
        device = get_default_device()
        print(f"cuda:{device}")

        root_gang = ProcessGroupGang.create_default_process_group(device)
        
        gang = root_gang.create_gang(all_ranks)
        
        gangs = Gangs(root=root_gang, tp=gang, dp=gang, pp=gang, rdp=None, sdp=None)

        process_group_gangs = gangs.tp.as_process_group()
    
        print(f"Process of rank {gangs.tp.rank}/{gangs.tp.size} has spawned")

    else:

        gangs = FakeGang(torch.device("cpu"))

    """
        
    processor = model.processor
    
    # for name, param in model.named_parameters():
        # print(name, param.device)
        
    
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
    
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, image=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO, device=device).to(model.dtype)
    # inputs = inputs.to(device)(model.dtype)
    print(f"Tensor shard on device {device}")
    text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
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

