# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import torch

from fairseq2.assets import asset_store
from fairseq2.generation import (
    Chatbot,
    ChatMessage,
    SamplingSequenceGenerator,
    TopPSampler,
)
from fairseq2.generation.utils import StdOutPrintHook
from fairseq2.models.llama import (
    LLaMAChatbot,
    load_llama_config,
    load_llama_model,
    load_llama_tokenizer,
)


def run_llama_chatbot(checkpoint_dir: Optional[Path] = None) -> None:
    model_card = asset_store.retrieve_card("llama2_7b_chat")

    if checkpoint_dir is not None:
        model_card.field("checkpoint").set(checkpoint_dir / "consolidated.pth")
        model_card.field("tokenizer").set(checkpoint_dir / "tokenizer.model")

    config = load_llama_config(model_card)

    model = load_llama_model(
        model_card, dtype=torch.float16, device=torch.device("cuda")
    )

    tokenizer = load_llama_tokenizer(model_card)

    sampler = TopPSampler(p=0.8)

    generator = SamplingSequenceGenerator(
        model,
        sampler,
        temperature=0.6,
        max_gen_len=1024,
        max_seq_len=config.max_seq_len,
    )

    chatbot = LLaMAChatbot(generator, tokenizer)

    run_chatbot(chatbot)


def run_chatbot(chatbot: Chatbot) -> None:
    dialog = []

    system_prompt = input("System Prompt (press enter to skip): ")

    if system_prompt:
        dialog.append(ChatMessage(role="system", content=system_prompt))

    print("\nYou can end the chat by typing 'bye'.\n")

    while (prompt := input("You> ")) != "bye":
        message = ChatMessage(role="user", content=prompt)

        dialog.append(message)

        print("\nLLaMA> ", end="")

        hook = StdOutPrintHook(chatbot.text_decoder)

        with chatbot.generator.register_step_hook(hook):
            response, _ = chatbot(dialog)

        print("\n")

        dialog.append(response)

    print("\nLLaMA> Bye!")


def main() -> None:
    parser = ArgumentParser(prog="chatbot", description="A basic LLaMA chatbot")

    # checkpoint
    param = parser.add_argument(
        "-c", "--checkpoint-dir", metavar="DIR", dest="checkpoint_dir", type=Path
    )
    param.help = "path to the LLaMA checkpoint directory"

    args = parser.parse_args()

    run_llama_chatbot(args.checkpoint_dir)


if __name__ == "__main__":
    main()
