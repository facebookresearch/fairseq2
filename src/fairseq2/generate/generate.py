from typing import List, Optional, cast

import torch
from torch import Tensor

from fairseq2.generate.search import Search
from fairseq2.nn.incremental_state import IncrementalStateBag

from ..nn import Projection
from .tokenizer import Tokenizer


@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    search: Search,
    src_tokens: Tensor,
    prefix_tokens: Optional[Tensor] = None,
    top: int = 0,
    _disable_caching: bool = False,
) -> torch.Tensor:
    # TODO: Below, to shutup mypy, we use `cast(<expected type>, model.<field>)`
    #
    # `model` has the loose type `nn.Module`, which has type annotations on its
    # __getattr__ method to return `Union[Tensor, Module]`.
    #
    # Here, we're assuming that `model` has transformer structure,
    # and we should select a stricter type with annotations for the attached fields.

    # compute the encoder output for each beam
    with torch.autograd.profiler.record_function("forward_encoder"):
        enc_out, enc_attn_mask = cast(torch.nn.Module, model.encoder).forward(
            src_tokens
        )

    # TODO: a way to ask Search this
    multiple = search.beam_size  # type: ignore

    # broadcast over the beam search space.
    enc_out = (
        torch.broadcast_to(enc_out, (multiple,) + enc_out.shape)
        .transpose(0, 1)
        .contiguous()
        .view(-1, *enc_out.shape[1:])
    )
    if enc_attn_mask:
        enc_attn_mask = (
            torch.broadcast_to(enc_attn_mask, (multiple,) + enc_attn_mask.shape)
            .transpose(0, 1)
            .contiguous()
            .view(-1, *enc_attn_mask.shape[1:])
        )

    incremental_states = IncrementalStateBag()
    state = search.prepare_state(
        src_tokens,
        prefix_tokens=prefix_tokens,
    )
    max_len = state.tokens.size(1)
    for idx in range(max_len):
        if state.done:
            break

        with torch.autograd.profiler.record_function("forward_decoder"):
            search_tokens = search.next_query(state)
            if not cast(bool, model.batch_first):
                search_tokens = search_tokens.T

            # TODO: incremental state
            # dec_out = model.decoder.forward(state.tokens, enc_out, enc_attn_mask)
            dec_out = cast(torch.nn.Module, model.decoder).forward(
                search_tokens,
                enc_out,
                enc_attn_mask,
                incremental_states,
            )

            dec_out = cast(Projection, model.score_proj)(
                get_last_time_axis(
                    dec_out,
                    cast(bool, model.batch_first),
                ),
            )

            # TODO: remove this
            if idx < 5:
                torch.set_printoptions(precision=2, threshold=100, linewidth=100)
                print(idx, dec_out[0, :50])

        with torch.autograd.profiler.record_function("search_step"):
            # Select the last time step prediction
            state = search.step(dec_out, state)

    tokens = search.finalize(state, top=top).tokens
    return tokens.view(-1, tokens.shape[-1])


def get_last_time_axis(x: Tensor, batch_first: bool) -> Tensor:
    assert len(x.shape) == 3
    if batch_first:
        y = x[:, -1, :].squeeze(1)
    else:
        y = x[-1, :, :].squeeze(0)
    assert len(y.shape) == 2
    return y


def generate_str(
    model: torch.nn.Module, tokenizer: Tokenizer, search: Search, sentences: List[str]
) -> List[str]:
    src_tokens = tokenizer.encode_batch(sentences)
    tgt_tokens = generate(model, search, src_tokens, top=1)
    return tokenizer.decode_batch(tgt_tokens.squeeze(1))
