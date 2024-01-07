# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Dict, final

import torch

from fairseq2.assets import AssetCard, asset_store, download_manager
from fairseq2.models.nllb.builder import NllbConfig, create_nllb_model, nllb_archs
from fairseq2.models.nllb.tokenizer import NllbTokenizer
from fairseq2.models.transformer import TransformerModel
from fairseq2.models.utils import ConfigLoader, ModelLoader, TokenizerLoaderBase
from fairseq2.models.utils.checkpoint import convert_fairseq_checkpoint
from fairseq2.typing import finaloverride


def convert_nllb_checkpoint(
    checkpoint: Dict[str, Any], config: NllbConfig
) -> Dict[str, Any]:
    """Convert a fairseq NLLB checkpoint to fairseq2."""
    state_dict = checkpoint["model"]

    # Check if we have a fairseq2 checkpoint.
    if "decoder_frontend.embed_weight" in state_dict:
        return checkpoint

    key_map = {
        # fmt: off
        r"^encoder\.embed_tokens\.":                              r"encoder_frontend.embed.",
        r"^decoder\.embed_tokens\.":                              r"decoder_frontend.embed.",
        r"^encoder\.layernorm_embedding\.":                       r"encoder_frontend.layer_norm.",
        r"^decoder\.layernorm_embedding\.":                       r"decoder_frontend.layer_norm.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"decoder.layers.\1.self_attn.output_proj.",
        r"^encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"encoder.layers.\1.self_attn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":  r"decoder.layers.\1.encoder_decoder_attn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.":            r"decoder.layers.\1.encoder_decoder_attn.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.": r"decoder.layers.\1.encoder_decoder_attn_layer_norm.",
        r"^encoder\.layers\.([0-9]+)\.fc1\.":                     r"encoder.layers.\1.ffn.inner_proj.",
        r"^decoder\.layers\.([0-9]+)\.fc1\.":                     r"decoder.layers.\1.ffn.inner_proj.",
        r"^encoder\.layers\.([0-9]+)\.fc2\.":                     r"encoder.layers.\1.ffn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.fc2\.":                     r"decoder.layers.\1.ffn.output_proj.",
        r"^encoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"encoder.layers.\1.ffn_layer_norm.",
        r"^decoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"decoder.layers.\1.ffn_layer_norm.",
        r"^decoder\.output_projection\.":                         r"final_proj.",
        # fmt: on
    }

    # Convert to fairseq2.
    checkpoint = convert_fairseq_checkpoint(checkpoint, key_map)

    state_dict = checkpoint["model"]

    embeds = state_dict["final_proj.weight"]

    # fairseq had a bug that accidentally introduced a dummy token in the
    # embedding table of NLLB-100. We just discard it.
    if embeds.size(0) == 256103:  # means NLLB-100
        embeds = embeds[:-1]

        state_dict["final_proj.weight"] = embeds

    # fairseq checkpoints have duplicate embedding weights. Ensure that we
    # use a single embedding table in fairseq2.
    state_dict["encoder_frontend.embed.weight"] = embeds
    state_dict["decoder_frontend.embed.weight"] = embeds

    # The embedding positions of the control symbols in fairseq's dict do
    # not match the SentencePiece model of the tokenizer.
    with torch.inference_mode():
        # (BOS, PAD, EOS, UNK) -> (PAD, UNK, BOS, EOS)
        embeds[[0, 1, 2, 3]] = embeds[[1, 3, 0, 2]]

    return checkpoint


@final
class NllbTokenizerLoader(TokenizerLoaderBase[NllbTokenizer]):
    """Loads tokenizers used by NLLB models."""

    @finaloverride
    def _load(self, pathname: Path, card: AssetCard) -> NllbTokenizer:
        langs = card.field("langs").as_list(str)

        default_lang = card.field("default_lang").as_(str)

        return NllbTokenizer(pathname, langs, default_lang)


load_nllb_config = ConfigLoader[NllbConfig](asset_store, nllb_archs)

load_nllb_model = ModelLoader[TransformerModel, NllbConfig](
    asset_store,
    download_manager,
    load_nllb_config,
    create_nllb_model,
    convert_nllb_checkpoint,
    restrict_checkpoints=False,
)

load_nllb_tokenizer = NllbTokenizerLoader(asset_store, download_manager)
