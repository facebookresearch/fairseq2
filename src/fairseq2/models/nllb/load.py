# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from fairseq2 import services
from fairseq2.assets import AssetCard, AssetDownloadManager, AssetStore
from fairseq2.models.nllb.build import (
    NllbConfig,
    create_nllb_model,
    get_nllb_archs,
    get_nllb_config,
)
from fairseq2.models.nllb.tokenizer import NllbTokenizer
from fairseq2.models.transformer import TransformerModel
from fairseq2.models.utils.checkpoint import (
    MapLocation,
    load_checkpoint,
    upgrade_fairseq_checkpoint,
)


def load_nllb_model(
    model_name: str, device: Optional[torch.device] = None, progress: bool = True
) -> Tuple[TransformerModel, NllbTokenizer]:
    """Load the specified NLLB model.

    :param model_name:
        The name of the model.
    :param device:
        The device on which to initialize the model.
    :param progress:
        If ``True``, displays a progress bar to stderr.

    :returns:
        The model and its associated tokenizer.
    """
    card = services.get(AssetStore).retrieve_card(model_name)

    return NllbLoader(card, progress=progress).load_model(device=device)


class NllbLoader:
    """Loads a specified NLLB model."""

    card: AssetCard
    download_manager: AssetDownloadManager
    force: bool
    progress: bool
    cfg: NllbConfig

    def __init__(
        self, card: AssetCard, force: bool = False, progress: bool = True
    ) -> None:
        """
        :param card:
            The asset card of the model.
        :param force:
            If ``True``, downloads the model assets even if they are already in
            cache.
        :param progress:
            If ``True``, displays a progress bar to stderr.
        """
        card.field("model_type").check_equals("nllb")

        self.card = card

        self.download_manager = services.get(AssetDownloadManager)

        self.force = force
        self.progress = progress

        supported_arch_names = get_nllb_archs()

        arch_name = self.card.field("model_arch").as_one_of(supported_arch_names)

        self.cfg = get_nllb_config(arch_name)

    def load_model(
        self, device: Optional[torch.device] = None
    ) -> Tuple[TransformerModel, NllbTokenizer]:
        """Load the NLLB model.

        :param device:
            The device on which to initialize the model.

        :returns:
            The model and its associated tokenizer.
        """
        tokenizer = self.load_tokenizer()

        # TODO: Initialize on Meta device!
        model = create_nllb_model(self.cfg, tokenizer.vocab_info, device)

        checkpoint = self.load_checkpoint(map_location="cpu")

        model.load_state_dict(checkpoint["model"])

        return model, tokenizer

    def load_checkpoint(self, map_location: MapLocation = None) -> Mapping[str, Any]:
        """Load the checkpoint of the NLLB model.

        :param map_location:
            Same as the ``map_location`` parameter of :meth:`torch.load`.
        """
        uri = self.card.field("checkpoint").as_uri()

        pathname = self.download_manager.download_checkpoint(
            uri, self.card.name, force=self.force, progress=self.progress
        )

        return load_checkpoint(
            pathname,
            self.card.name,
            map_location=map_location,
            upgrader=self._upgrade_checkpoint,
        )

    def load_tokenizer(self) -> NllbTokenizer:
        """Load the tokenizer of the NLLB model."""
        uri = self.card.field("tokenizer").as_uri()

        pathname = self.download_manager.download_tokenizer(
            uri, self.card.name, force=self.force, progress=self.progress
        )

        langs = self.card.field("langs").as_list(str)

        default_lang = self.card.field("default_lang").as_(str)

        return NllbTokenizer(pathname, langs, default_lang)

    @classmethod
    def _upgrade_checkpoint(cls, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        key_map = cls._fairseq_key_map()

        checkpoint = upgrade_fairseq_checkpoint(checkpoint, key_map)

        state_dict = checkpoint["model"]

        # fairseq checkpoints have duplicate embedding weights.
        embeds = state_dict["final_proj.weight"]

        state_dict["encoder_frontend.embed.weight"] = embeds
        state_dict["decoder_frontend.embed.weight"] = embeds

        # The embedding positions of the control tokens do not match the
        # SentencePiece model of the tokenizer.
        with torch.inference_mode():
            # (BOS, PAD, EOS, UNK) -> (PAD, UNK, BOS, EOS)
            embeds[[0, 1, 2, 3]] = embeds[[1, 3, 0, 2]]

        return checkpoint

    @staticmethod
    def _fairseq_key_map() -> Dict[str, str]:
        return {
            # fmt: off
            r"^decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"decoder.layers.\1.self_attn.output_proj.",
            r"^encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"encoder.layers.\1.self_attn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":  r"decoder.layers.\1.encoder_decoder_attn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":  r"decoder.layers.\1.encoder_decoder_attn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn\.":            r"decoder.layers.\1.encoder_decoder_attn.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.": r"decoder.layers.\1.encoder_decoder_attn_layer_norm.",
            r"^encoder\.layers\.([0-9]+)\.fc1\.":                     r"encoder.layers.\1.ffn.inner_proj.",
            r"^decoder\.layers\.([0-9]+)\.fc1\.":                     r"decoder.layers.\1.ffn.inner_proj.",
            r"^encoder\.layers\.([0-9]+)\.fc2\.":                     r"encoder.layers.\1.ffn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.fc2\.":                     r"decoder.layers.\1.ffn.output_proj.",
            r"^encoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"encoder.layers.\1.ffn_layer_norm.",
            r"^decoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"decoder.layers.\1.ffn_layer_norm.",
            r"^encoder\.embed_tokens\.":                              r"encoder_frontend.embed.",
            r"^decoder\.embed_tokens\.":                              r"decoder_frontend.embed.",
            r"^decoder\.output_projection\.":                         r"final_proj.",
            # fmt: on
        }
