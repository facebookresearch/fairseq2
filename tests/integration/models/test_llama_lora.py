# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq2.models.llama import create_llama_model, get_llama_lora_config, llama_archs
from fairseq2.nn.lora import (
    wrap_lora,
    unwrap_lora,
    merge_lora,
    unmerge_lora,
    freeze_non_lora
)


def test_lora_wrappers_llama_works() -> None:
    llama_config = llama_archs.get_config("7b")
    model = create_llama_model(llama_config, device="cpu")

    lora_config = get_llama_lora_config()

    inputs = torch.LongTensor([[1, 2], [1, 3]], device="cpu")

    model.eval()

    with torch.inference_mode():
        output_before_wrap, _ = model.decode(seqs=inputs, seq_lens=None)

    model = wrap_lora(model, lora_config)

    with torch.inference_mode():
        output_after_wrap, _ = model.decode(seqs=inputs, seq_lens=None)

    # Outputs should be the same as lora_B is initialized with zeros.
    torch.testing.assert_close(output_before_wrap, output_after_wrap)

    model = unwrap_lora(model, merge=False)

    with torch.inference_mode():
        output_after_unwrap, _ = model.decode(seqs=inputs, seq_lens=None)

    torch.testing.assert_close(output_after_wrap, output_after_unwrap)

    model = wrap_lora(model, lora_config)
    merge_lora(model)

    with torch.inference_mode():
        output_after_merge, _ = model.decode(seqs=inputs, seq_lens=None)

    torch.testing.assert_close(output_after_unwrap, output_after_merge)

    unmerge_lora(model)

    with torch.inference_mode():
        output_after_unmerge, _ = model.decode(seqs=inputs, seq_lens=None)

    torch.testing.assert_close(output_after_merge, output_after_unmerge)

    model.train()
    freeze_non_lora(model, unfreeze_bias="none")

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert "lora_" in name
