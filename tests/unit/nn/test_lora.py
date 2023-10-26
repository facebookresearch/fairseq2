# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch.nn.functional import embedding, linear

from fairseq2.nn import Linear
from fairseq2.nn.embedding import StandardEmbedding
from fairseq2.nn.lora import LoRAConfig, LoRAEmbedding, LoRALinear


def test_lora_linear_works() -> None:
    lora_config = LoRAConfig(r=4, alpha=1.0, dropout_p=0.0, keys=[])

    linear_layer = Linear(8, 8, bias=True)

    lora_linear = LoRALinear(
        linear_layer, lora_config, skip_init=False, device=torch.device("cpu")
    )

    torch.nn.init.kaiming_uniform_(lora_linear.lora_B, a=math.sqrt(5))

    assert lora_linear.lora_A.shape == (4, 8)

    assert lora_linear.lora_B.shape == (8, 4)

    seqs = torch.randn([2, 8])

    orig_out = linear_layer(seqs)

    lora_out = lora_linear(seqs)

    lora_partial_out = (
        linear(seqs, lora_linear.lora_B @ lora_linear.lora_A) * lora_linear.scaling
    )

    torch.testing.assert_close(lora_out - orig_out, lora_partial_out)


def test_lora_liner_merge_unmerge_work() -> None:
    lora_config = LoRAConfig(r=4, alpha=1.0, dropout_p=0.0, keys=[])

    linear_layer = Linear(8, 8, bias=True)

    lora_linear = LoRALinear(
        linear_layer, lora_config, skip_init=False, device=torch.device("cpu")
    )

    torch.nn.init.kaiming_uniform_(lora_linear.lora_B, a=math.sqrt(5))

    seqs = torch.randn([2, 8])

    orig_weight = lora_linear.weight.data.clone()

    wrapped_org_weight = lora_linear.wrapped.weight.data.clone()  # type: ignore[operator]

    torch.testing.assert_close(orig_weight, wrapped_org_weight)

    lora_linear.merge()

    assert lora_linear.merged

    merged_weight = lora_linear.weight.data.clone()

    wrapped_merged_weight = lora_linear.wrapped.weight.data.clone()  # type: ignore[operator]

    torch.testing.assert_close(merged_weight, wrapped_merged_weight)

    merged_out = lora_linear(seqs)

    lora_linear.unmerge()

    assert not lora_linear.merged

    un_merged_weight = lora_linear.weight.data.clone()

    wrapped_un_merged_weight = lora_linear.wrapped.weight.data.clone()

    torch.testing.assert_close(un_merged_weight, wrapped_un_merged_weight)

    un_merged_out = lora_linear(seqs)

    torch.testing.assert_close(orig_weight, un_merged_weight)

    torch.testing.assert_close(un_merged_out, merged_out)

    lora_AB = (lora_linear.lora_B @ lora_linear.lora_A * lora_linear.scaling).data

    torch.testing.assert_close(merged_weight - un_merged_weight, lora_AB)


def test_lora_embedding_works() -> None:
    lora_config = LoRAConfig(r=4, alpha=1.0, dropout_p=0.0, keys=[])

    pad_idx = 0

    embed_layer = StandardEmbedding(4, 8, pad_idx)

    lora_embed = LoRAEmbedding(embed_layer, lora_config, device=torch.device("cpu"))

    torch.nn.init.normal_(lora_embed.lora_A)

    assert lora_embed.lora_A.shape == (4, 4)

    assert lora_embed.lora_B.shape == (8, 4)

    seqs = torch.randint(0, 4, [2, 5])

    orig_out = embed_layer(seqs)

    lora_out = lora_embed(seqs)

    lora_partial_out = embedding(
        seqs, (lora_embed.lora_B @ lora_embed.lora_A).T * lora_embed.scaling, pad_idx
    )

    torch.testing.assert_close(lora_out - orig_out, lora_partial_out)


def test_lora_embedding_merge_unmerge_work() -> None:
    lora_config = LoRAConfig(r=4, alpha=1.0, dropout_p=0.0, keys=[])

    pad_idx = 0

    embed_layer = StandardEmbedding(4, 8, pad_idx)

    lora_embed = LoRAEmbedding(embed_layer, lora_config, device=torch.device("cpu"))

    torch.nn.init.normal_(lora_embed.lora_A)

    seqs = torch.randint(0, 4, [2, 5])

    orig_weight = lora_embed.weight.data.clone()

    wrapped_orig_weight = lora_embed.wrapped.weight.data.clone()  # type: ignore[operator]

    torch.testing.assert_close(orig_weight, wrapped_orig_weight)

    lora_embed.merge()

    assert lora_embed.merged

    merged_weight = lora_embed.weight.data.clone()

    wrapped_merged_weight = lora_embed.wrapped.weight.data.clone()  # type: ignore[operator]

    torch.testing.assert_close(merged_weight, wrapped_merged_weight)

    merged_out = lora_embed(seqs)

    lora_embed.unmerge()

    assert not lora_embed.merged

    un_merged_weight = lora_embed.weight.data.clone()

    wrapped_un_merged_weight = lora_embed.wrapped.weight.data.clone()

    torch.testing.assert_close(un_merged_weight, wrapped_un_merged_weight)

    un_merged_out = lora_embed(seqs)

    torch.testing.assert_close(orig_weight, un_merged_weight)

    torch.testing.assert_close(un_merged_out, merged_out)

    lora_AB = (lora_embed.lora_B @ lora_embed.lora_A * lora_embed.scaling).data.T

    torch.testing.assert_close(merged_weight - un_merged_weight, lora_AB)
