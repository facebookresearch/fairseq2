import pytest
import torch

from fairseq2.nn import Linear, Projection
from src.fairseq2.nn.lora import LoRAConfig, LoRALinear

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
