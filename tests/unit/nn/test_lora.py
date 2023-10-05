import pytest
import torch

from fairseq2.nn import Linear, Projection
from fairseq2.models.llama import create_llama_model, get_llama_lora_config, llama_archs
from fairseq2.nn.lora import (
    LoRAConfig,
    LoRALinear,
    wrap_lora,
    unwrap_lora,
    merge_lora,
    unmerge_lora,
    freeze_non_lora
)

from torch.nn.functional import linear


def test_lora_liner_works() -> None:
    lora_config = LoRAConfig(
        r=4,
        alpha=1.,
        dropout_p=0.,
        keys=None
    )

    linear_layer = Linear(8, 8, bias=True)

    lora_linear = LoRALinear(linear_layer, lora_config, skip_init=False, device='cpu')

    seqs = torch.randn([2, 8])

    orig_out = linear_layer(seqs)

    lora_out = lora_linear(seqs)

    assert lora_linear.lora_A.shape == (4, 8)

    assert lora_linear.lora_B.shape == (8, 4)

    lora_partial_out = linear(seqs, lora_linear.lora_B @ lora_linear.lora_A) * lora_linear.scaling

    torch.testing.assert_close(lora_out - orig_out, lora_partial_out)


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
