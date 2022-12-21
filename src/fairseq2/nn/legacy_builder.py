import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

import fairseq2.nn
from fairseq2.generate import SpmTokenizer
from fairseq2.nn import transformer
from fairseq2.typing import DataType, Device

TODO = "TODO, please reachout to prioritize this"
FAIRSEQ1_PAD = 1

log = logging.getLogger(__name__)


def eq(*xs: int) -> int:
    x0 = xs[0]
    for i, x in enumerate(xs):
        assert x == x0, TODO
    return x0


class Fairseq1TransformerBuilder(transformer.TransformerBuilder):
    padding_token_idx: int

    def __init__(
        self,
        cfg: Any,
        num_tokens: int,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        self.cfg = cfg
        assert not cfg.encoder_learned_pos, TODO
        assert not cfg.decoder_learned_pos, TODO

        super().__init__(
            num_tokens=num_tokens,
            padding_token_idx=1,
            model_dim=eq(
                cfg.decoder_embed_dim, cfg.encoder_embed_dim, cfg.decoder_input_dim
            ),
            num_enc_layers=cfg.encoder_layers,
            num_dec_layers=cfg.decoder_layers,
            num_enc_attn_heads=cfg.encoder_attention_heads,
            num_dec_attn_heads=cfg.decoder_attention_heads,
            ffn_inner_dim=eq(cfg.encoder_ffn_embed_dim, cfg.decoder_ffn_embed_dim),
            norm_order=(
                transformer.TransformerNormOrder.PRE
                if eq(cfg.encoder_normalize_before, cfg.decoder_normalize_before)
                else transformer.TransformerNormOrder.POST
            ),
            dropout_p=cfg.dropout,
            # Fairseq1 was always batch_first
            batch_first=True,
            max_seq_len=eq(cfg.max_source_positions, cfg.max_target_positions),
            device=device,
            dtype=dtype,
        )

    def build_positional_embedding(self) -> Optional[fairseq2.nn.PositionalEmbedding]:
        return fairseq2.nn.HighPassSinusoidalPositionalEmbedding(
            max_seq_len=self.max_seq_len,
            embedding_dim=self.model_dim,
            padding_token_idx=1,
            batch_first=self.batch_first,
            **self._fct_kwargs,
        )


def _upgrade_state_dict(cfg: Any, state_dict: Dict[str, Tensor]) -> None:
    for k in sorted(state_dict.keys()):
        k2 = k
        k2 = k2.replace(".embed_tokens.weight", ".embed.weight")
        k2 = k2.replace(".embed_positions._float_tensor", ".pos_embed.weight")
        k2 = k2.replace(".encoder_attn.", ".enc_dec_attn.")
        k2 = k2.replace(".encoder_attn_layer_norm.", ".enc_dec_attn_layer_norm.")
        k2 = k2.replace(".fc1.", ".ffn.inner_proj.")
        k2 = k2.replace(".fc2.", ".ffn.out_proj.")
        k2 = k2.replace(".final_layer_norm.", ".ffn_layer_norm.")
        k2 = k2.replace("decoder.output_projection.weight", "score_proj.weight")
        if k2 != k:
            assert k2 not in state_dict
            state_dict[k2] = state_dict.pop(k)

    if not cfg.encoder_learned_pos:
        state_dict.pop("encoder.pos_embed.weight")
    if not cfg.encoder_learned_pos:
        state_dict.pop("decoder.pos_embed.weight")

    for emb in ("decoder.embed.weight", "encoder.embed.weight", "score_proj.weight"):
        _swap_bos_pad_eos_unk(state_dict[emb])

    # TODO: should we have something like this in fairseq2 ?
    state_dict.pop("encoder.version")
    state_dict.pop("decoder.version")
    if cfg.share_all_embeddings:
        # Fairseq1 checkpoints have duplicated but equal matrices.
        state_dict["encoder.embed.weight"] = state_dict["score_proj.weight"]
        state_dict["decoder.embed.weight"] = state_dict["score_proj.weight"]


def _swap_bos_pad_eos_unk(embeddings: Tensor) -> None:
    with torch.inference_mode():
        bos_pad_eos_unk = embeddings[:4, :].detach().clone()
        pad_unk_bos_eos = torch.stack([bos_pad_eos_unk[i] for i in [1, 3, 0, 2]])
        embeddings[:4, :] = pad_unk_bos_eos


def load_fairseq1_checkpoint(
    model_file: Path,
    spm_path: Path,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> Tuple[transformer.Transformer, SpmTokenizer, transformer.TransformerBuilder]:

    # TODO: should this return a builder ?

    # TODO: fix the tokenizer, by offsetting the pieces indexes by one, or fix the embeddings
    tokenizer = SpmTokenizer.from_file(spm_path, batch_first=True, _pad_shift_hack=True)

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

    builder = Fairseq1TransformerBuilder(
        cfg,
        num_tokens=tokenizer.vocab_size(),
        device=device,
        dtype=dtype,
    )
    model = builder.build()
    keys2 = set(model.state_dict().keys())

    _upgrade_state_dict(cfg, state["model"])
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

    return (model, tokenizer, builder)


if __name__ == "__main__":
    import func_argparse

    parser = func_argparse.func_argparser(load_fairseq1_checkpoint)
    fp_types = {"fp16": torch.float16, "bf16": torch.bfloat16}
    func_argparse.override(parser, "dtype", type=fp_types.__getitem__)

    model = func_argparse.parse_and_call(parser)
    print(model)
    print("Successfully loaded model !")
