from typing import List, Optional

import torch
from torch import Tensor

from fairseq2.generate.search import Search
from fairseq2.nn.incremental_state import IncrementalStateBag

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
    # compute the encoder output for each beam
    with torch.autograd.profiler.record_function("forward_encoder"):
        enc_out, enc_attn_mask = model.encoder.forward(src_tokens)

    incremental_states = IncrementalStateBag()
    state = search.prepare_state(src_tokens, prefix_tokens)
    max_len = state.tokens.size(1)
    for _ in range(max_len):
        if state.done:
            break

        with torch.autograd.profiler.record_function("forward_decoder"):
            if model.batch_first:
                new_tokens = state.tokens[:, : state.step + 1]
            else:
                new_tokens = state.tokens[: state.step + 1, :]
            # TODO: incremental state
            # dec_out = model.decoder.forward(state.tokens, enc_out, enc_attn_mask)
            dec_out = model.decoder.forward(
                new_tokens,
                enc_out,
                enc_attn_mask,
                incremental_states,
            )

            dec_out = model.score_proj(get_last_time_axis(dec_out, model.batch_first))

        with torch.autograd.profiler.record_function("search_step"):
            # Select the last time step prediction
            state = search.step(dec_out, state)

    return search.finalize(state, top=top).tokens


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
