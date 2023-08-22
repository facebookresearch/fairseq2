# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping, Union, final

import torch

from fairseq2.assets import (
    AssetCard,
    AssetDownloadManager,
    AssetStore,
    asset_store,
    download_manager,
)
from fairseq2.models.nllb.builder import NllbConfig, create_nllb_model, nllb_archs
from fairseq2.models.nllb.tokenizer import NllbTokenizer
from fairseq2.models.transformer import TransformerModel
from fairseq2.models.utils.checkpoint_loader import upgrade_fairseq_checkpoint
from fairseq2.models.utils.model_loader import ModelConfigLoader, ModelLoader
from fairseq2.typing import finaloverride


@final
class NllbLoader(ModelLoader[TransformerModel, NllbConfig]):
    """Loads NLLB models."""

    @finaloverride
    def _upgrade_checkpoint(
        self, checkpoint: Mapping[str, Any], config: NllbConfig
    ) -> Mapping[str, Any]:
        state_dict = checkpoint["model"]

        # Check if we have a fairseq2 checkpoint.
        if "decoder_frontend.embed_weight" in state_dict:
            return checkpoint

        key_map = self._fairseq_key_map()

        # Convert to fairseq2.
        checkpoint = upgrade_fairseq_checkpoint(checkpoint, key_map)

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

    @staticmethod
    def _fairseq_key_map() -> Dict[str, str]:
        return {
            # fmt: off
            r"^encoder\.embed_tokens\.":                              r"encoder_frontend.embed.",
            r"^decoder\.embed_tokens\.":                              r"decoder_frontend.embed.",
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


load_nllb_model = NllbLoader(
    asset_store, download_manager, create_nllb_model, nllb_archs
)


load_nllb_config = ModelConfigLoader[NllbConfig](asset_store, nllb_archs)


class NllbTokenizerLoader:
    """Loads tokenizers of NLLB models."""

    def __init__(
        self, asset_store: AssetStore, download_manager: AssetDownloadManager
    ) -> None:
        """
        :param asset_store:
            The asset store to retrieve the model information.
        :param download_manager:
            The download manager to use.
        """
        self.asset_store = asset_store
        self.download_manager = download_manager

    def __call__(
        self,
        model_name_or_card: Union[str, AssetCard],
        force: bool = False,
        progress: bool = True,
    ) -> NllbTokenizer:
        """
        :param model_name_or_card:
            The name or asset card of the model whose tokenizer to load.
        :param force:
            If ``True``, downloads the tokenizer even if it is already in cache.
        :param progress:
            If ``True``, displays a progress bar to stderr.
        """
        if isinstance(model_name_or_card, AssetCard):
            card: AssetCard = model_name_or_card
        else:
            card = self.asset_store.retrieve_card(model_name_or_card)

        uri = card.field("tokenizer").as_uri()

        pathname = self.download_manager.download_tokenizer(
            uri, card.name, force=force, progress=progress
        )

        langs = card.field("langs").as_list(str)

        default_lang = card.field("default_lang").as_(str)

        return NllbTokenizer(pathname, langs, default_lang)


load_nllb_tokenizer = NllbTokenizerLoader(asset_store, download_manager)
