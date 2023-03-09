# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from fairseq2.data.text import VocabularyInfo
from fairseq2.generate import SpmTokenizer
from fairseq2.models.transformer import (
    TransformerConfig,
    TransformerModel,
    create_transformer_model,
)
from fairseq2.nn.transformer import TransformerNormOrder


def convert_fairseq1_config(cfg: Any) -> TransformerConfig:
    if cfg.encoder_learned_pos or cfg.decoder_learned_pos:
        raise RuntimeError(
            "Learned positional embeddings are not supported. File a bug report if you want them to be supported."
        )

    model_dim = cfg.decoder_input_dim

    if model_dim != cfg.encoder_embed_dim or model_dim != cfg.decoder_embed_dim:
        raise ValueError(
            f"The size of the model dimension does not match between the input ({model_dim}), encoder ({cfg.encoder_embed_dim}), and decoder ({cfg.decoder_embed_dim})."
        )

    if cfg.encoder_ffn_embed_dim != cfg.decoder_ffn_embed_dim:
        raise ValueError(
            f"The size of the FFN inner dimension does not match between the encoder ({cfg.encoder_ffn_embed_dim}) and decoder ({cfg.decoder_ffn_embed_dim})."
        )

    ffn_dim = cfg.encoder_ffn_embed_dim

    if cfg.encoder_normalize_before != cfg.decoder_normalize_before:
        raise ValueError(
            "The Layer Normalization order does not match between the encoder and decoder."
        )

    if not cfg.share_all_embeddings:
        raise ValueError("Non shared embeddings are not supported.")

    return TransformerConfig(
        max_seq_len=cfg.max_source_positions,
        model_dim=model_dim,
        num_enc_layers=cfg.encoder_layers,
        num_dec_layers=cfg.decoder_layers,
        num_enc_attn_heads=cfg.encoder_attention_heads,
        num_dec_attn_heads=cfg.decoder_attention_heads,
        ffn_inner_dim=ffn_dim,
        dropout_p=cfg.dropout,
        norm_order=TransformerNormOrder.PRE,
        legacy_pos_embed=True,
    )


def _upgrade_legacy_state_dict(cfg: Any, state_dict: Dict[str, Tensor]) -> None:
    old_new_key_map = {
        "encoder.embed_tokens.weight": "encoder_frontend.embed.weight",
        "encoder.embed_positions._float_tensor": "encoder_frontend.pos_embed.weight",
        "decoder.embed_tokens.weight": "decoder_frontend.embed.weight",
        "decoder.embed_positions._float_tensor": "decoder_frontend.pos_embed.weight",
        ".encoder_attn.": ".enc_dec_attn.",
        ".encoder_attn_layer_norm.": ".enc_dec_attn_layer_norm.",
        ".fc1.": ".ffn.inner_proj.",
        ".fc2.": ".ffn.out_proj.",
        ".final_layer_norm.": ".ffn_layer_norm.",
        "decoder.output_projection.weight": "score_proj.weight",
    }

    for key in sorted(state_dict.keys()):
        modified_key = key

        # Convert the legacy fairseq module key to its fairseq2 equivalent.
        for old, new in old_new_key_map.items():
            modified_key = modified_key.replace(old, new)

        if key != modified_key:
            if modified_key in state_dict:
                raise RuntimeError(
                    f"The state dictionary already contains a same-named key for '{modified_key}'. Please file a bug report."
                )

            state_dict[modified_key] = state_dict.pop(key)

    # Non-learned positional embeddings don't have to be stored in the state
    # dictionary since we can generate them on-the-fly.
    if not cfg.encoder_learned_pos:
        del state_dict["encoder_frontend.pos_embed.weight"]
    if not cfg.encoder_learned_pos:
        del state_dict["decoder_frontend.pos_embed.weight"]

    for emb in (
        "decoder_frontend.embed.weight",
        "encoder_frontend.embed.weight",
        "score_proj.weight",
    ):
        _swap_bos_pad_eos_unk(state_dict[emb])

    # fairseq2 uses the version attribute of PyTorch's state_dict.
    del state_dict["encoder.version"]
    del state_dict["decoder.version"]

    if cfg.share_all_embeddings:
        # fairseq checkpoints have duplicated but equal matrices.
        state_dict["encoder_frontend.embed.weight"] = state_dict["score_proj.weight"]
        state_dict["decoder_frontend.embed.weight"] = state_dict["score_proj.weight"]


def _swap_bos_pad_eos_unk(embeddings: Tensor) -> None:
    with torch.inference_mode():
        bos_pad_eos_unk = embeddings[:4, :].detach().clone()
        pad_unk_bos_eos = torch.stack([bos_pad_eos_unk[i] for i in [1, 3, 0, 2]])
        embeddings[:4, :] = pad_unk_bos_eos


def load_fairseq1_checkpoint(
    model_file: Path,
    spm_path: Path,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[TransformerModel, SpmTokenizer, TransformerConfig]:

    # TODO: this tuple is a bit weird, should we have a reference class for this tuple ?
    # I want the tokenizer and model to always go hand in hand.
    # The builder is also important to be hable to serialize the model and reload it later.

    tokenizer = SpmTokenizer.from_file(spm_path, _pad_shift_hack=True)

    # TODO: MoE models require a bit more complex loading
    state = torch.load(str(model_file), map_location=device)
    full_cfg = state["cfg"]
    cfg = full_cfg["model"]
    arch = cfg.arch
    assert arch == "transformer", "TODO"
    original1 = len(state["model"].keys())

    # HACKS: add special tokens to the model
    # those might only exist in some fairseq branches
    for lang in cfg.langs:
        tokenizer.add_special_token(f"__{lang}__")

    if getattr(cfg, "add_data_source_prefix_tags", False):
        for data_source in ["<MINED_DATA>", "<MMT_BT_DATA>", "<SMT_BT_DATA>"]:
            tokenizer.add_special_token(data_source)

    assert not getattr(cfg, "add_ssl_task_tokens", False), "TODO"

    new_cfg = convert_fairseq1_config(cfg)
    if dtype is not None:
        new_cfg.dtype = dtype

    vocab_info = VocabularyInfo(
        tokenizer.vocab_size(),
        tokenizer.UNK,
        tokenizer.BOS,
        tokenizer.EOS,
        tokenizer.PAD,
    )

    model = create_transformer_model(new_cfg, vocab_info, device)
    keys2 = set(model.state_dict().keys())

    _upgrade_legacy_state_dict(cfg, state["model"])
    keys1 = set(state["model"].keys())
    if keys2 != keys1:
        print(f"fairseq2 model: {len(keys2)} ({len(keys2 - keys1)} unique)")
        print("Key not found in checkpoint:", sorted(keys2 - keys1))
        print(
            f"fairseq1 model: {len(keys1)} ({len(keys1 - keys2)} unique, from {original1}"
        )
        print("Key in checkpoint, not used by model:", sorted(keys1 - keys2))
        raise Exception("Key mismatch, please open a bug")

    model.load_state_dict(state["model"])

    return (model, tokenizer, new_cfg)


if __name__ == "__main__":
    import func_argparse

    parser = func_argparse.func_argparser(load_fairseq1_checkpoint)
    fp_types = {"fp16": torch.float16, "bf16": torch.bfloat16}
    func_argparse.override(parser, "dtype", type=fp_types.__getitem__)

    model = func_argparse.parse_and_call(parser)
    print(model)
    print("Successfully loaded model !")
