# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch.serialization import MAP_LOCATION

from fairseq2.assets import (
    AssetCard,
    AssetDownloader,
    global_asset_downloader,
    global_asset_store,
)
from fairseq2.models.nllb.build import (
    create_nllb_model,
    get_nllb_archs,
    get_nllb_config,
)
from fairseq2.models.nllb.tokenizer import NllbTokenizer
from fairseq2.models.transformer import TransformerModel
from fairseq2.models.utils.load import load_checkpoint, upgrade_fairseq_checkpoint


def load_nllb_model(
    model_name: str, device: Optional[torch.device] = None, force: bool = False
) -> Tuple[TransformerModel, NllbTokenizer]:
    """Load the specified NLLB model.

    :param model_name:
        The name of the model.
    :param device:
        The device on which to initialize the model.
    :param force:
        If ``True``, downloads the model assets even if they are already in
        cache.

    :returns:
        The model and its associated tokenizer.
    """
    card = global_asset_store.retrieve_card(model_name)

    loader = NllbLoader(card, global_asset_downloader, force)

    return loader.load_model(device=device)


class NllbLoader:
    """Loads a specified NLLB model."""

    card: AssetCard
    downloader: AssetDownloader
    force: bool

    def __init__(
        self,
        card: AssetCard,
        downloader: Optional[AssetDownloader] = None,
        force: bool = False,
    ) -> None:
        """
        :param card:
            The asset card of the model.
        :param downloader:
            The asset downloader.
        :param force:
            If ``True``, downloads the model assets even if they are already in
            cache.
        """
        card.field("model_type").check_equals("nllb")

        self.card = card
        self.downloader = downloader or global_asset_downloader
        self.force = force

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

        supported_arch_names = get_nllb_archs()

        arch_name = self.card.field("model_arch").as_one_of(supported_arch_names)

        cfg = get_nllb_config(arch_name)

        # TODO: Initialize on Meta device!
        model = create_nllb_model(cfg, tokenizer.vocab_info, device)

        checkpoint = self.load_checkpoint(map_location="cpu")

        model.load_state_dict(checkpoint["model"])

        return model, tokenizer

    def load_checkpoint(self, map_location: MAP_LOCATION = None) -> Mapping[str, Any]:
        """Load the checkpoint of the NLLB model.

        :param map_location:
            Same as the ``map_location`` parameter of :meth:`torch.load`.
        """
        uri = self.card.field("checkpoint").as_uri()

        pathname = self.downloader.download_checkpoint(
            uri, self.card.name, force=self.force
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

        pathname = self.downloader.download_tokenizer(
            uri, self.card.name, force=self.force
        )

        langs = self.card.field("langs").as_list(str)

        default_lang = self.card.field("default_lang").as_(str)

        return NllbTokenizer(pathname, langs, default_lang)

    @classmethod
    def _upgrade_checkpoint(cls, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        checkpoint = upgrade_fairseq_checkpoint(checkpoint)

        state_dict = checkpoint["model"]

        # fairseq checkpoints have duplicate embedding weights.
        embeds = state_dict["score_proj.weight"]

        state_dict["encoder_frontend.embed.weight"] = embeds
        state_dict["decoder_frontend.embed.weight"] = embeds

        # The embedding positions of the control tokens do not match the
        # SentencePiece model of the tokenizer.
        with torch.inference_mode():
            # (BOS, PAD, EOS, UNK) -> (PAD, UNK, BOS, EOS)
            embeds[[0, 1, 2, 3]] = embeds[[1, 3, 0, 2]]

        return checkpoint
