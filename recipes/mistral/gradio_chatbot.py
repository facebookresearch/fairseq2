# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

try:
    import gradio as gr
except ImportError:
    print("Install gradio: pip install gradio")

import torch

from fairseq2.assets import asset_store
from fairseq2.generation import (
    ChatMessage,
    SamplingSequenceGenerator,
    TopPSampler,
)
from fairseq2.models.mistral import (
    MistralChatbot,
    load_mistral_model,
    load_mistral_tokenizer,
)

model_card = asset_store.retrieve_card("mistral_7b_instruct")
model = load_mistral_model(
    model_card, dtype=torch.float16, device=torch.device("cuda:0")
)
tokenizer = load_mistral_tokenizer(model_card)


def interact_with_chatbot(input_text: str, history: list, top_p: float):
    sampler = TopPSampler(p=top_p)
    generator = SamplingSequenceGenerator(
        model, sampler, temperature=1.0, max_gen_len=1024
    )
    chatbot = MistralChatbot(generator, tokenizer)

    mistral_format_history = []
    for user, bot in history:
        user_msg = ChatMessage(role="user", content=user)
        bot_msg = ChatMessage(role="bot", content=bot)
        mistral_format_history.extend([user_msg, bot_msg])

    mistral_format_history.append(ChatMessage(role="user", content=input_text))
    response, _ = chatbot(mistral_format_history)

    return str(response.content)


demo = gr.ChatInterface(
    interact_with_chatbot,
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="Type your prompt here", container=False, scale=7),
    additional_inputs=[gr.Slider(0.1, 1.0, value=0.9, step=0.01, label="nucleus topp")],
    title="Mistral Instruct 7B",
    description="Mistral Instruct 7B served locally by fairseq2",
    theme="soft",
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
)


if __name__ == "__main__":
    demo.launch()
