#!/usr/bin/env python3
"""
Ultra Simple Gemma SFT Training Example

This script demonstrates loading a Gemma model using fairseq2's HuggingFace
integration and training it on 3 simple supervised fine-tuning (SFT) pairs.

Supports single-GPU and multi-GPU training with FSDP.

Usage:
    # CPU mode (slow, for testing only)
    python simple_gemma_sft.py --device cpu

    # Single GPU mode
    python simple_gemma_sft.py --device cuda

    # Multi-GPU mode with FSDP (using torchrun)
    torchrun --nproc_per_node=2 simple_gemma_sft.py --device cuda
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from fairseq2.checkpoint import StandardCheckpointManager
from fairseq2.device import Device
from fairseq2.file_system import LocalFileSystem
from fairseq2.gang import ProcessGroupGang, create_fsdp_gangs, create_parallel_gangs
from fairseq2.io import _TorchTensorFileDumper, _TorchTensorFileLoader
from fairseq2.models.hg_qwen_omni import (
    apply_fsdp_to_hg_transformer_lm,
    load_causal_lm,
    load_hg_tokenizer_simple,
)
from fairseq2.nn.fsdp import to_fsdp2
from fairseq2.utils.threading import _StandardThreadPool


def create_sft_pairs() -> list[dict[str, str]]:
    """Create 20 simple SFT (instruction, response) pairs."""
    return [
        {
            "instruction": "What is the capital of France?",
            "response": "The capital of France is Paris.",
        },
        {
            "instruction": "What is 2 + 2?",
            "response": "2 + 2 equals 4.",
        },
        {
            "instruction": "What color is the sky?",
            "response": "The sky is blue.",
        },
        {
            "instruction": "How many continents are there?",
            "response": "There are 7 continents on Earth.",
        },
        {
            "instruction": "What is the largest ocean?",
            "response": "The Pacific Ocean is the largest ocean.",
        },
        {
            "instruction": "What is 10 minus 3?",
            "response": "10 minus 3 equals 7.",
        },
        {
            "instruction": "What is the capital of Italy?",
            "response": "The capital of Italy is Rome.",
        },
        {
            "instruction": "How many days in a week?",
            "response": "There are 7 days in a week.",
        },
        {
            "instruction": "What is 5 times 3?",
            "response": "5 times 3 equals 15.",
        },
        {
            "instruction": "What is the smallest prime number?",
            "response": "The smallest prime number is 2.",
        },
        {
            "instruction": "What color is grass?",
            "response": "Grass is green.",
        },
        {
            "instruction": "How many sides does a triangle have?",
            "response": "A triangle has 3 sides.",
        },
        {
            "instruction": "What is the capital of Japan?",
            "response": "The capital of Japan is Tokyo.",
        },
        {
            "instruction": "What is 100 divided by 10?",
            "response": "100 divided by 10 equals 10.",
        },
        {
            "instruction": "What planet do we live on?",
            "response": "We live on planet Earth.",
        },
        {
            "instruction": "How many hours in a day?",
            "response": "There are 24 hours in a day.",
        },
        {
            "instruction": "What is the capital of Spain?",
            "response": "The capital of Spain is Madrid.",
        },
        {
            "instruction": "What is 8 plus 7?",
            "response": "8 plus 7 equals 15.",
        },
        {
            "instruction": "What color is the sun?",
            "response": "The sun is yellow.",
        },
        {
            "instruction": "How many months in a year?",
            "response": "There are 12 months in a year.",
        },
    ]


class SFTDataset(Dataset):
    """Simple dataset for SFT pairs."""

    def __init__(self, pairs: list[dict[str, str]], tokenizer, device):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        text = f"Instruction: {pair['instruction']}\nResponse: {pair['response']}"
        return text


def collate_fn(batch, tokenizer, device):
    """Collate function to prepare batches."""
    # Tokenize all texts
    encoded = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # For causal LM, labels are the same as input_ids (next token prediction)
    # We set padding tokens to -100 so they're ignored in loss computation
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cpu or cuda",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for checkpoints",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-6, help="Learning rate"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--no-fsdp",
        action="store_true",
        help="Disable FSDP even in multi-GPU setup",
    )
    args = parser.parse_args()

    # Check if we're in distributed mode
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        use_fsdp = world_size > 1 and not args.no_fsdp
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        use_fsdp = False

    # Set device
    if args.device == "cuda":
        device = Device("cuda", index=local_rank)
        torch.cuda.set_device(local_rank)
    else:
        device = Device("cpu")

    # Only rank 0 prints
    def print_rank0(*msg):
        if rank == 0:
            print(*msg)

    print_rank0(f"World size: {world_size}, Rank: {rank}, Device: {device}")
    print_rank0(f"FSDP enabled: {use_fsdp}")

    # Create output directory
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize distributed training if needed
    gangs = None
    if is_distributed:
        print_rank0("\n0. Initializing distributed training...")
        root_gang = ProcessGroupGang.create_default_process_group(device)
        gangs = create_parallel_gangs(root_gang)
        # For FSDP, create FSDP-specific gangs
        if use_fsdp:
            gangs = create_fsdp_gangs(gangs)
        print_rank0("   ✓ Distributed gangs initialized")
    else:
        # Create fake gangs for single-GPU
        from fairseq2.gang import FakeGang

        fake_gang = FakeGang(device=device)
        from dataclasses import dataclass

        @dataclass
        class FakeGangs:
            root: FakeGang
            dp: FakeGang
            rdp: FakeGang
            sdp: FakeGang
            tp: FakeGang
            pp: FakeGang

            @property
            def device(self):
                return self.root.device

            def close(self):
                pass

        gangs = FakeGangs(
            root=fake_gang,
            dp=fake_gang,
            rdp=fake_gang,
            sdp=fake_gang,
            tp=fake_gang,
            pp=fake_gang,
        )

    # Create CheckpointManager
    print_rank0("\n0.5. Setting up checkpoint manager...")
    file_system = LocalFileSystem()
    tensor_loader = _TorchTensorFileLoader(file_system)
    tensor_dumper = _TorchTensorFileDumper(file_system)
    thread_pool = _StandardThreadPool.create_default(world_size)
    checkpoint_manager = StandardCheckpointManager(
        output_dir=args.output_dir,
        gangs=gangs,
        file_system=file_system,
        tensor_file_loader=tensor_loader,
        tensor_file_dumper=tensor_dumper,
        thread_pool=thread_pool,
    )
    print_rank0(f"   ✓ Checkpoint manager created (output_dir={args.output_dir})")

    # Check for existing checkpoint
    start_step = 0
    if checkpoint_manager.has_checkpoint():
        last_step = checkpoint_manager.maybe_get_last_step_number()
        if last_step is not None:
            print_rank0(
                f"   Found checkpoint at step {last_step}, will resume after loading model..."
            )
            start_step = last_step

    # Load model using fairseq2's HuggingFace integration
    print_rank0("\n1. Loading Gemma-3-1B-IT model...")
    # Use fp32 for stable training (fp16 requires loss scaling)
    model = load_causal_lm(
        "google/gemma-3-1b-it",
        dtype="float32",  # fp32 for stability
        trust_remote_code=True,
    )
    # Move model to the correct device
    model = model.to(device)

    # Apply FSDP if using multi-GPU
    if use_fsdp and gangs is not None:
        print_rank0("   Applying FSDP to model...")

        # Create an applier that matches the expected signature
        # apply_fsdp_to_hg_transformer_lm expects (model, granularity, wrapper)
        # but to_fsdp2 calls applier(model, wrapper), so we need to add granularity
        def fsdp_applier(model, wrapper):
            return apply_fsdp_to_hg_transformer_lm(model, "layer", wrapper)

        model = to_fsdp2(
            model,
            gangs=gangs,
            applier=fsdp_applier,
        )
        print_rank0("   ✓ FSDP applied - model sharded across all GPUs")

    model.train()
    print_rank0(f"   ✓ Model loaded: {model.__class__.__name__}")
    if use_fsdp:
        print_rank0(
            f"   Note: Each rank sees its local device (rank 0={device}), but model is sharded across all ranks"
        )

    # Load tokenizer
    print_rank0("\n2. Loading tokenizer...")
    tokenizer = load_hg_tokenizer_simple("google/gemma-3-1b-it")
    # Get the HuggingFace tokenizer from the wrapper using .raw property
    hf_tokenizer = tokenizer.raw
    print_rank0("   ✓ Tokenizer loaded")

    # Create SFT training data
    print_rank0("\n3. Creating SFT dataset...")
    sft_pairs = create_sft_pairs()
    print_rank0(f"   ✓ Created {len(sft_pairs)} training examples")

    # Create dataset and dataloader
    print_rank0(f"\n4. Creating DataLoader (batch_size={args.batch_size})...")
    dataset = SFTDataset(sft_pairs, hf_tokenizer, device)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, hf_tokenizer, device),
    )
    print_rank0(
        f"   ✓ DataLoader created ({len(dataset)} examples, {len(dataloader)} batches per epoch)"
    )

    # Setup optimizer
    print_rank0("\n5. Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Resume from checkpoint if available
    if start_step > 0:
        print_rank0(f"\n5.5. Resuming from checkpoint step {start_step}...")
        try:
            checkpoint_manager.load_model_state(start_step, model)
            checkpoint_manager.load_optimizer_state(start_step, optimizer)
            print_rank0("   ✓ Model and optimizer state loaded")
        except Exception as e:
            print_rank0(f"   ⚠ Warning: Could not load checkpoint: {e}")
            print_rank0("   Starting from scratch...")
            start_step = 0

    print_rank0(
        f"   ✓ Optimizer: AdamW (lr={args.learning_rate}, max_grad_norm={args.max_grad_norm})"
    )

    # Training loop
    print_rank0(f"\n6. Training for {args.num_epochs} epoch(s)...")
    print_rank0("-" * 60)

    global_step = start_step
    for epoch in range(args.num_epochs):
        print_rank0(f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(dataloader):
            global_step += 1
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.loss

            # Check for NaN loss
            if torch.isnan(loss):
                print_rank0(
                    f"⚠ Warning: NaN loss detected at step {global_step}. Stopping training."
                )
                print_rank0(
                    "   This usually means the learning rate is too high or training is unstable."
                )
                break

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()

            print_rank0(
                f"  Step {global_step} (batch {batch_idx + 1}/{len(dataloader)}) | Loss: {loss.item():.4f}"
            )

    print_rank0("-" * 60)
    print_rank0(f"\n✓ Training completed! Final step: {global_step}")

    # Save final checkpoint
    print_rank0(f"\n7. Saving checkpoint at step {global_step}...")
    try:
        # Create a simple wrapper for stateful objects
        class StatefulWrapper:
            def __init__(self, obj):
                self.obj = obj

            def state_dict(self):
                if hasattr(self.obj, "state_dict"):
                    return self.obj.state_dict()
                return {}

            def load_state_dict(self, state_dict):
                if hasattr(self.obj, "load_state_dict"):
                    self.obj.load_state_dict(state_dict)

        # Wrap objects for checkpoint manager
        trainer_state = StatefulWrapper({"step": global_step, "epoch": args.num_epochs})
        model_state = StatefulWrapper(model)
        optimizer_state = StatefulWrapper(optimizer)
        data_reader_state = StatefulWrapper({})

        checkpoint_manager.save_checkpoint(
            step_nr=global_step,
            trainer=trainer_state,
            model=model_state,
            optimizer=optimizer_state,
            data_reader=data_reader_state,
            blocking=True,
        )
        print_rank0(
            f"   ✓ Checkpoint saved to {args.output_dir}/checkpoints/step_{global_step}"
        )
    except Exception as e:
        print_rank0(f"   ⚠ Warning: Could not save checkpoint: {e}")

    # Only test generation on rank 0 in single-GPU mode
    # FSDP models require all ranks to participate in forward passes
    if rank == 0 and not use_fsdp:
        # Optional: Test generation on one of the instructions
        print("\n8. Testing model output (after training)...")
        model.eval()
        test_instruction = "Instruction: What is 2 + 2?\nResponse:"
        test_tokens = hf_tokenizer(test_instruction, return_tensors="pt")
        # Move all tensors to device
        test_input_ids = test_tokens["input_ids"].to(device)
        test_attention_mask = test_tokens["attention_mask"].to(device)

        print(f"   Input length: {test_input_ids.shape[1]} tokens")

        with torch.no_grad():
            # Use greedy decoding (more stable than sampling)
            generated = model.generate(
                test_input_ids,
                attention_mask=test_attention_mask,
                max_new_tokens=30,
                do_sample=False,  # Greedy decoding
                pad_token_id=hf_tokenizer.eos_token_id,
            )

        # Decode the full output
        generated_text = hf_tokenizer.decode(generated[0], skip_special_tokens=True)

        # Also decode just the new tokens (excluding the prompt)
        prompt_length = test_input_ids.shape[1]
        new_tokens = generated[0][prompt_length:]
        new_text = hf_tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"\n   Input prompt: {test_instruction}")
        print(f"   Generated token IDs: {new_tokens.tolist()[:10]}...")  # Show first 10
        print(f"   Full output: {generated_text}")
        print(f"   Generated (new tokens only): '{new_text}'")
        print(f"   Number of new tokens: {len(new_tokens)}")

        # Check if it's all EOS tokens
        if hf_tokenizer.eos_token_id is not None:
            eos_count = (new_tokens == hf_tokenizer.eos_token_id).sum().item()
            print(f"   EOS tokens in output: {eos_count}")

        print("\nDone!")
    elif rank == 0 and use_fsdp:
        print_rank0(
            "\n8. Skipping generation test (FSDP models require all ranks for inference)"
        )
        print_rank0("   For generation with FSDP, use a dedicated inference script.")
        print_rank0("\nDone!")

    # Cleanup
    checkpoint_manager.close()
    if is_distributed and gangs is not None:
        gangs.close()


if __name__ == "__main__":
    main()
