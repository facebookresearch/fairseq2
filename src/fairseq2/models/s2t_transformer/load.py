# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Mapping, Optional, Tuple
from zipfile import ZipFile

import torch
from torch.serialization import MAP_LOCATION

from fairseq2 import services
from fairseq2.assets import AssetCard, AssetDownloadManager, AssetStore
from fairseq2.models.s2t_transformer.build import (
    create_s2t_transformer_model,
    get_s2t_transformer_archs,
    get_s2t_transformer_config,
)
from fairseq2.models.s2t_transformer.model import S2TTransformerModel
from fairseq2.models.s2t_transformer.tokenizer import S2TTransformerTokenizer
from fairseq2.models.utils.load import load_checkpoint, upgrade_fairseq_checkpoint


def load_s2t_transformer_model(
    model_name: str, device: Optional[torch.device] = None, progress: bool = True
) -> Tuple[S2TTransformerModel, S2TTransformerTokenizer]:
    """Load the specified S2T Transformer model.

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

    return S2TTransformerLoader(card, progress=progress).load_model(device=device)


class S2TTransformerLoader:
    """Loads a specified S2T Transformer model."""

    card: AssetCard
    download_manager: AssetDownloadManager
    force: bool
    progress: bool

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
        card.field("model_type").check_equals("s2t_transformer")

        self.card = card

        self.download_manager = services.get(AssetDownloadManager)

        self.force = force
        self.progress = progress

    def load_model(
        self, device: Optional[torch.device] = None
    ) -> Tuple[S2TTransformerModel, S2TTransformerTokenizer]:
        """Load the S2T Transformer model.

        :param device:
            The device on which to initialize the model.

        :returns:
            The model and its associated tokenizer.
        """
        tokenizer = self.load_tokenizer()

        supported_arch_names = get_s2t_transformer_archs()

        arch_name = self.card.field("model_arch").as_one_of(supported_arch_names)

        cfg = get_s2t_transformer_config(arch_name)

        # TODO: Initialize on Meta device!
        model = create_s2t_transformer_model(cfg, tokenizer.vocab_info, device)

        checkpoint = self.load_checkpoint(map_location="cpu")

        model.load_state_dict(checkpoint["model"])

        return model, tokenizer

    def load_checkpoint(self, map_location: MAP_LOCATION = None) -> Mapping[str, Any]:
        """Load the checkpoint of the S2T Transformer model.

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
            upgrader=upgrade_fairseq_checkpoint,
        )

    def load_tokenizer(self) -> S2TTransformerTokenizer:
        """Load the tokenizer of the S2T Transformer model."""
        uri = self.card.field("tokenizer").as_uri()

        zip_pathname = self.download_manager.download_tokenizer(
            uri, self.card.name, force=self.force, progress=self.progress
        )

        filename = self.card.field("tokenizer_file").as_filename()

        pathname = zip_pathname.with_name(filename)

        if self.force or not pathname.exists():
            try:
                with ZipFile(zip_pathname) as fp:
                    fp.extract(filename, path=zip_pathname.parent)
            except (KeyError, IOError) as ex:
                raise RuntimeError(
                    f"The load of the tokenizer of the model '{self.card.name}' has failed. Please file a bug report."
                ) from ex

        task = self.card.field("task").as_one_of({"transcription", "translation"})

        tgt_langs = self.card.field("tgt_langs").as_list(str)

        return S2TTransformerTokenizer(
            pathname, task, set(tgt_langs), default_tgt_lang=tgt_langs[0]
        )
