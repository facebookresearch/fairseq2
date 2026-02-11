#!/usr/bin/env python3
"""Check if frontend advances state_bag.step_nr."""

import torch
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.nn import BatchLayout, IncrementalStateBag

device = torch.device("cpu")

config = get_gemma3n_e2b_config()
model = create_gemma3n_model(config, device=device, dtype=torch.float32)

input_ids = torch.tensor([[1, 2, 3]], device=device)
seq_lens = [3]
batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
state_bag = IncrementalStateBag(max_num_steps=3)

print(f"Initial step_nr: {state_bag.step_nr}")

model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)

print(f"After frontend step_nr: {state_bag.step_nr}")
print(f"\nFrontend advanced step_nr by: {state_bag.step_nr}")

if state_bag.step_nr > 0:
    print("\n⚠️  Frontend advanced step_nr!")
    print("This will cause RoPE to use wrong position offsets in attention!")
