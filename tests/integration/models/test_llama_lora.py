# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq2.data import VocabularyInfo
from fairseq2.models.llama import LLaMAConfig, create_llama_model, get_llama_lora_config
from fairseq2.nn.lora import (
    freeze_non_lora,
    merge_lora,
    unmerge_lora,
    unwrap_lora,
    wrap_lora,
)


def test_lora_wrappers_llama_works() -> None:
    # Construct a smaller LLaMAModel to prevent CI from failing
    llama_config = LLaMAConfig(
        model_dim=1024,
        max_seq_len=2048,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        num_layers=16,
        num_attn_heads=8,
        num_key_value_heads=8,
        ffn_inner_dim=1024 * 4,
        ffn_inner_dim_to_multiple=1,
        dropout_p=0.1,
    )
    model = create_llama_model(llama_config, device=torch.device("cpu"))

    lora_config = get_llama_lora_config()

    inputs = torch.LongTensor([[1, 2], [1, 3]], device=torch.device("cpu"))

    model.eval()

    with torch.inference_mode():
        output_before_wrap, _ = model.decode(seqs=inputs, padding_mask=None)

    model = wrap_lora(model, lora_config)  # type: ignore[assignment]

    with torch.inference_mode():
        output_after_wrap, _ = model.decode(seqs=inputs, padding_mask=None)

    # Outputs should be the same as lora_B is initialized with zeros.
    torch.testing.assert_close(output_before_wrap, output_after_wrap)

    model = unwrap_lora(model, merge=False)  # type: ignore[assignment]

    with torch.inference_mode():
        output_after_unwrap, _ = model.decode(seqs=inputs, padding_mask=None)

    torch.testing.assert_close(output_after_wrap, output_after_unwrap)

    model = wrap_lora(model, lora_config)  # type: ignore[assignment]
    merge_lora(model)

    with torch.inference_mode():
        output_after_merge, _ = model.decode(seqs=inputs, padding_mask=None)

    torch.testing.assert_close(output_after_unwrap, output_after_merge)

    unmerge_lora(model)

    with torch.inference_mode():
        output_after_unmerge, _ = model.decode(seqs=inputs, padding_mask=None)

    torch.testing.assert_close(output_after_merge, output_after_unmerge)

    model.train()
    freeze_non_lora(model, unfreeze_bias="none")

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert "lora_" in name
