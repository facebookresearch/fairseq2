"""
This module provides the cif_function(...) method that implements parallel CIF,
including training (with target lengths) and inference.
- Author: Chih-Chiang Chang (github: George0828Zhang)
"""
import torch
from typing import Optional, Dict, List
from torch import Tensor


def cif_function(
    inputs: Tensor,
    alpha: Tensor,
    beta: float = 1.0,
    tail_thres: float = 0.5,
    padding_mask: Optional[Tensor] = None,
    target_lengths: Optional[Tensor] = None,
    eps: float = 1e-4,
    unbound_alpha: bool = False
) -> Dict[str, List[Tensor]]:
    r""" A fast parallel implementation of continuous integrate-and-fire (CIF)
    https://arxiv.org/abs/1905.11235

    Shapes:
        N: batch size
        S: source (encoder) sequence length
        C: source feature dimension
        T: target sequence length

    Args:
        inputs (Tensor): (N, S, C) Input features to be integrated.
        alpha (Tensor): (N, S) Weights corresponding to each elements in the
            inputs. It is expected to be after sigmoid function.
        beta (float): the threshold used for determine firing.
        tail_thres (float): the threshold for determine firing for tail handling.
        padding_mask (Tensor, optional): (N, S) A binary mask representing
            padded elements in the inputs. 1 is padding, 0 is not.
        target_lengths (Tensor, optional): (N,) Desired length of the targets
            for each sample in the minibatch.
        eps (float, optional): Epsilon to prevent underflow for divisions.
            Default: 1e-4
        unbound_alpha (bool, optional): Whether to check if 0 <= alpha <= 1.

    Returns -> Dict[str, List[Tensor]]: Key/values described below.
        cif_out: (N, T, C) The output integrated from the source.
        cif_lengths: (N,) The output length for each element in batch.
        alpha_sum: (N,) The sum of alpha for each element in batch.
            Can be used to compute the quantity loss.
        delays: (N, T) The expected delay (in terms of source tokens) for
            each target tokens in the batch.
        tail_weights: (N,) During inference, return the tail.
        scaled_alpha: (N, S) alpha after applying weight scaling.
        cumsum_alpha: (N, S) cumsum of alpha after scaling.
        right_indices: (N, S) right scatter indices, or floor(cumsum(alpha)).
        right_weights: (N, S) right scatter weights.
        left_indices: (N, S) left scatter indices.
        left_weights: (N, S) left scatter weights.
    """
    B, S, C = inputs.size()
    assert tuple(alpha.size()) == (B, S), f"{alpha.size()} != {(B, S)}"
    assert not torch.isnan(alpha).any(), "Nan in alpha tensor."
    assert unbound_alpha or (alpha.le(1.0 + eps).all() and alpha.ge(0.0 - eps).all()), (
        "Incorrect values in alpha tensor"
        ", 0.0 <= tensor <= 1.0"
    )

    dtype = alpha.dtype
    alpha = alpha.float()
    if padding_mask is not None:
        padding_mask = padding_mask.bool()
        assert not padding_mask[:, 0].any(), "Expected right-padded inputs."
        alpha = alpha.masked_fill(padding_mask, 0)

    if target_lengths is not None:
        assert target_lengths.size() == (B,)
        feat_lengths = target_lengths.long()
        desired_sum = beta * target_lengths.type_as(inputs) + eps
        alpha_sum = alpha.sum(1)
        alpha = alpha * (desired_sum / alpha_sum).unsqueeze(1)
        T = feat_lengths.max()
    else:
        alpha_sum = alpha.sum(1)
        feat_lengths = (alpha_sum / beta).floor().long()
        T = feat_lengths.max()

    # aggregate and integrate
    csum = alpha.cumsum(-1)
    with torch.no_grad():
        # indices used for scattering
        right_idx = (csum / beta).floor().long().clip(max=T)
        left_idx = right_idx.roll(1, dims=1)
        left_idx[:, 0] = 0

        # count # of fires from each source
        fire_num = right_idx - left_idx
        extra_weights = (fire_num - 1).clip(min=0)

    # The extra entry in last dim is for tail
    output = inputs.new_zeros((B, T + 1, C))
    delay = inputs.new_zeros((B, T + 1))
    source_range = torch.arange(1, 1 + S).unsqueeze(0).type_as(inputs)
    zero = alpha.new_zeros((1,))

    # right scatter
    fire_mask = fire_num > 0
    right_weight = torch.where(
        fire_mask,
        csum - right_idx.type_as(alpha) * beta,
        zero
    ).type_as(inputs)
    output.scatter_add_(
        1,
        right_idx.unsqueeze(-1).expand(-1, -1, C),
        right_weight.unsqueeze(-1) * inputs
    )
    delay.scatter_add_(
        1,
        right_idx,
        right_weight * source_range / beta
    )

    # left scatter
    left_weight = (
        alpha - right_weight - extra_weights.type_as(alpha) * beta
    ).type_as(inputs)
    output.scatter_add_(
        1,
        left_idx.unsqueeze(-1).expand(-1, -1, C),
        left_weight.unsqueeze(-1) * inputs
    )
    delay.scatter_add_(
        1,
        left_idx,
        left_weight * source_range / beta
    )

    # extra scatters
    if extra_weights.ge(0).any():
        extra_steps = extra_weights.max().item()
        tgt_idx = left_idx
        src_feats = inputs * beta
        for _ in range(extra_steps):
            tgt_idx = (tgt_idx + 1).clip(max=T)
            # (B, S, 1)
            src_mask = (extra_weights > 0)
            output.scatter_add_(
                1,
                tgt_idx.unsqueeze(-1).expand(-1, -1, C),
                src_feats * src_mask.unsqueeze(2)
            )
            delay.scatter_add_(
                1,
                tgt_idx,
                source_range * src_mask
            )
            extra_weights -= 1

    # tail handling
    if target_lengths is not None:
        # training time -> ignore tail
        output = output[:, :T, :]
        delay = delay[:, :T]
    else:
        # find out contribution to output tail
        # note: w/o scaling, extra weight is all 0
        zero = right_weight.new_zeros((1,))
        r_mask = right_idx == feat_lengths.unsqueeze(1)
        tail_weights = torch.where(r_mask, right_weight, zero).sum(-1)
        l_mask = left_idx == feat_lengths.unsqueeze(1)
        tail_weights += torch.where(l_mask, left_weight, zero).sum(-1)

        # a size (B,) mask that extends position that passed threshold.
        extend_mask = tail_weights >= tail_thres

        # extend 1 fire and upscale the weights
        if extend_mask.any():
            # (B, T, C), may have infs so need the mask
            upscale = (
                torch.ones_like(output)
                .scatter(
                    1,
                    feat_lengths.view(B, 1, 1).expand(-1, -1, C),
                    beta / (
                        tail_weights
                        .masked_fill(~extend_mask, beta)
                        .view(B, 1, 1)
                        .expand(-1, -1, C)),
                )
                .detach()
            )
            output *= upscale
            feat_lengths += extend_mask.long()
            T = feat_lengths.max()
        output = output[:, :T, :]
        delay = delay[:, :T]

        # a size (B, T) mask to erase weights
        tail_mask = torch.arange(T, device=output.device).unsqueeze(0) >= feat_lengths.unsqueeze(1)
        output[tail_mask] = 0

    return {
        "cif_out": [output],
        "cif_lengths": [feat_lengths],
        "alpha_sum": [alpha_sum.to(dtype)],
        "delays": [delay],
        "tail_weights": [tail_weights] if target_lengths is None else [],
        "scaled_alpha": [alpha],
        "cumsum_alpha": [csum],
        "right_indices": [right_idx],
        "right_weights": [right_weight],
        "left_indices": [left_idx],
        "left_weights": [left_weight],
    }