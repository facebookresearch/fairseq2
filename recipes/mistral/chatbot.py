# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import torch

from fairseq2.assets import default_asset_store
from fairseq2.data.text import load_text_tokenizer
from fairseq2.generation import (
    Chatbot,
    ChatMessage,
    SamplingSequenceGenerator,
    TopPSampler,
)
from fairseq2.models import create_chatbot, load_model
from fairseq2.models.decoder import DecoderModel
from fairseq2.typing import Device


def run_chatbot(checkpoint_dir: Optional[Path] = None) -> None:
    model_name = "mistral_7b_instruct"

    model_card = default_asset_store.retrieve_card(model_name)

    if checkpoint_dir is not None:
        model_card.field("checkpoint").set(checkpoint_dir / "consolidated.00.pth")
        model_card.field("tokenizer").set(checkpoint_dir / "tokenizer.model")

    model = load_model(model_card, dtype=torch.float16, device=Device("cuda:0"))

    if not isinstance(model, DecoderModel):
        raise ValueError("The model must be a decoder model.")

    tokenizer = load_text_tokenizer(model_card)

    sampler = TopPSampler(p=0.8)

    generator = SamplingSequenceGenerator(
        model, sampler, temperature=0.6, max_gen_len=1024
    )

    # compat
    chatbot = create_chatbot(generator, tokenizer)  # type: ignore[arg-type]

    do_run_chatbot(model_name, chatbot)


def do_run_chatbot(name: str, chatbot: Chatbot) -> None:
    dialog = []

    if chatbot.supports_system_prompt:
        system_prompt = input("System Prompt (press enter to skip): ")

        if system_prompt:
            dialog.append(ChatMessage(role="system", content=system_prompt))

        print()

    print("You can end the chat by typing 'bye'.\n")

    while (prompt := input("You> ")) != "bye":
        message = ChatMessage(role="user", content=prompt)

        dialog.append(message)

        print(f"\n{name}> ", end="")

        response, _ = chatbot(dialog, stdout=True)

        print("\n")

        dialog.append(response)

    print(f"\n{name}> Bye!")


def main() -> None:
    parser = ArgumentParser(prog="chatbot", description="A basic Mistral chatbot")

    # checkpoint
    param = parser.add_argument(
        "-c", "--checkpoint-dir", metavar="DIR", dest="checkpoint_dir", type=Path
    )
    param.help = "path to the model checkpoint directory"

    args = parser.parse_args()

    run_chatbot(args.checkpoint_dir)


if __name__ == "__main__":
    main()
