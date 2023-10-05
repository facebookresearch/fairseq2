import math
import pytest
import torch

from fairseq2.nn import Linear
from fairseq2.nn.embedding import StandardEmbedding
from fairseq2.models.llama import create_llama_model, get_llama_lora_config, llama_archs
from fairseq2.nn.lora import (
    LoRAConfig,
    LoRAEmbedding,
    LoRALinear,
    wrap_lora,
    unwrap_lora,
    merge_lora,
    unmerge_lora,
    freeze_non_lora
)

from torch.nn.functional import embedding, linear


def test_lora_liner_works() -> None:
    lora_config = LoRAConfig(
        r=4,
        alpha=1.,
        dropout_p=0.,
        keys=None
    )

    linear_layer = Linear(8, 8, bias=True)

    lora_linear = LoRALinear(linear_layer, lora_config, skip_init=False, device='cpu')

    torch.nn.init.kaiming_uniform_(lora_linear.lora_B, a=math.sqrt(5))

    assert lora_linear.lora_A.shape == (4, 8)

    assert lora_linear.lora_B.shape == (8, 4)

    seqs = torch.randn([2, 8])

    orig_out = linear_layer(seqs)

    lora_out = lora_linear(seqs)

    lora_partial_out = linear(seqs, lora_linear.lora_B @ lora_linear.lora_A) * lora_linear.scaling

    torch.testing.assert_close(lora_out - orig_out, lora_partial_out)


def test_lora_liner_merge_unmerge_work() -> None:
    lora_config = LoRAConfig(
        r=4,
        alpha=1.,
        dropout_p=0.,
        keys=None
    )

    linear_layer = Linear(8, 8, bias=True)

    lora_linear = LoRALinear(linear_layer, lora_config, skip_init=False, device='cpu')

    torch.nn.init.kaiming_uniform_(lora_linear.lora_B, a=math.sqrt(5))

    seqs = torch.randn([2, 8])

    orig_weight = lora_linear.weight.clone()

    lora_linear.merge()

    assert lora_linear.merged

    merged_weight = lora_linear.weight.clone()

    merged_out = lora_linear(seqs)

    lora_linear.unmerge()

    assert not lora_linear.merged

    un_merged_weight = lora_linear.weight.clone()

    un_merged_out = lora_linear(seqs)

    torch.testing.assert_close(orig_weight, un_merged_weight)

    torch.testing.assert_close(un_merged_out, merged_out)

    lora_AB = (lora_linear.lora_B @ lora_linear.lora_A * lora_linear.scaling).data

    torch.testing.assert_close(merged_weight - un_merged_weight, lora_AB)


def test_lora_embedding_works() -> None:
    lora_config = LoRAConfig(
        r=4,
        alpha=1.,
        dropout_p=0.,
        keys=None
    )

    pad_idx = 0

    embed_layer = StandardEmbedding(4, 8, pad_idx)

    lora_embed = LoRAEmbedding(embed_layer, lora_config, device='cpu')

    torch.nn.init.normal_(lora_embed.lora_A)

    assert lora_embed.lora_A.shape == (4, 4)

    assert lora_embed.lora_B.shape == (8, 4)

    seqs = torch.randint(0, 4, [2, 5])

    orig_out = embed_layer(seqs)

    lora_out = lora_embed(seqs)

    lora_partial_out = embedding(
        seqs, (lora_embed.lora_B @ lora_embed.lora_A).T * lora_embed.scaling,
        pad_idx)

    torch.testing.assert_close(lora_out - orig_out, lora_partial_out)


def test_lora_embedding_merge_unmerge_work() -> None:
    lora_config = LoRAConfig(
        r=4,
        alpha=1.,
        dropout_p=0.,
        keys=None
    )

    pad_idx = 0

    embed_layer = StandardEmbedding(4, 8, pad_idx)

    lora_embed = LoRAEmbedding(embed_layer, lora_config, device='cpu')

    torch.nn.init.normal_(lora_embed.lora_A)

    seqs = torch.randint(0, 4, [2, 5])

    orig_weight = lora_embed.weight.clone()

    lora_embed.merge()

    assert lora_embed.merged

    merged_weight = lora_embed.weight.clone()

    merged_out = lora_embed(seqs)

    lora_embed.unmerge()

    assert not lora_embed.merged

    un_merged_weight = lora_embed.weight.clone()

    un_merged_out = lora_embed(seqs)

    torch.testing.assert_close(orig_weight, un_merged_weight)

    torch.testing.assert_close(un_merged_out, merged_out)

    lora_AB = (lora_embed.lora_B @ lora_embed.lora_A * lora_embed.scaling).data.T

    torch.testing.assert_close(merged_weight - un_merged_weight, lora_AB)


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
