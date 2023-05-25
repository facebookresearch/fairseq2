# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping, Optional

import torch

from fairseq2 import services
from fairseq2.assets import AssetCard, AssetDownloadManager, AssetStore
from fairseq2.models.utils.checkpoint import (
    MapLocation,
    load_checkpoint,
    upgrade_fairseq_checkpoint,
)
from fairseq2.models.wav2vec2.build import (
    Wav2Vec2Config,
    create_wav2vec2_model,
    get_wav2vec2_archs,
    get_wav2vec2_config,
)
from fairseq2.models.wav2vec2.model import Wav2Vec2Model
from fairseq2.nn.transformer import TransformerNormOrder


def load_wav2vec2_model(
    model_name: str, device: Optional[torch.device] = None, progress: bool = True
) -> Wav2Vec2Model:
    """Load the specified wav2vec 2.0 model.

    :param model_name:
        The name of the model.
    :param device:
        The device on which to initialize the model.
    :param progress:
        If ``True``, displays a progress bar to stderr.

    :returns:
        The model.
    """
    card = services.get(AssetStore).retrieve_card(model_name)

    return Wav2Vec2Loader(card, progress=progress).load_model(device=device)


class Wav2Vec2Loader:
    """Loads a specified wav2vec 2.0 model."""

    card: AssetCard
    download_manager: AssetDownloadManager
    force: bool
    progress: bool
    cfg: Wav2Vec2Config

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
        card.field("model_type").check_equals("wav2vec2")

        self.card = card

        self.download_manager = services.get(AssetDownloadManager)

        self.force = force
        self.progress = progress

        supported_arch_names = get_wav2vec2_archs()

        arch_name = self.card.field("model_arch").as_one_of(supported_arch_names)

        self.cfg = get_wav2vec2_config(arch_name)

    def load_model(self, device: Optional[torch.device] = None) -> Wav2Vec2Model:
        """Load the wav2vec 2.0 model.

        :param device:
            The device on which to initialize the model.

        :returns:
            The model.
        """
        # TODO: Initialize on Meta device!
        model = create_wav2vec2_model(self.cfg, device)

        checkpoint = self.load_checkpoint(map_location="cpu")

        model.load_state_dict(checkpoint["model"])

        return model

    def load_checkpoint(self, map_location: MapLocation = None) -> Mapping[str, Any]:
        """Load the checkpoint of the wav2vec 2.0 model.

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

    def _upgrade_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        state_dict = checkpoint["model"]

        if self.cfg.norm_order == TransformerNormOrder.POST:
            # fmt: off
            state_dict["encoder_frontend.layer_norm.weight"] = state_dict["encoder.layer_norm.weight"]
            state_dict["encoder_frontend.layer_norm.bias"]   = state_dict["encoder.layer_norm.bias"]
            # fmt: on

            del state_dict["encoder.layer_norm.weight"]
            del state_dict["encoder.layer_norm.bias"]

        state_dict["quantizer.num_updates"] = torch.zeros((), device="cpu")

        key_map = self._fairseq_key_map()

        return upgrade_fairseq_checkpoint(checkpoint, key_map)

    @staticmethod
    def _fairseq_key_map() -> Dict[str, str]:
        return {
            # fmt: off
            r"^encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.": r"encoder.layers.\1.self_attn.output_proj.",
            r"^encoder\.layers\.([0-9]+)\.fc1\.":                 r"encoder.layers.\1.ffn.inner_proj.",
            r"^encoder\.layers\.([0-9]+)\.fc2\.":                 r"encoder.layers.\1.ffn.output_proj.",
            r"^encoder\.layers\.([0-9]+)\.final_layer_norm\.":    r"encoder.layers.\1.ffn_layer_norm.",
            r"^decoder\.layers\.([0-9]+)\.final_layer_norm\.":    r"decoder.layers.\1.ffn_layer_norm.",
            r"^encoder\.embed_tokens\.":                          r"encoder_frontend.embed.",
            r"^encoder\.pos_conv\.0\.":                           r"encoder_frontend.pos_encoder.conv.",
            r"^feature_extractor\.conv_layers\.([0-9]+)\.0.":     r"encoder_frontend.feature_extractor.layers.\1.conv.",
            r"^feature_extractor\.conv_layers\.0\.2.":            r"encoder_frontend.feature_extractor.layers.0.group_norm.",
            r"^layer_norm\.":                                     r"encoder_frontend.post_extract_layer_norm.",
            r"^post_extract_proj\.":                              r"encoder_frontend.post_extract_proj.",
            r"^mask_emb":                                         r"encoder_frontend.masker.temporal_mask_embed",
            r"^project_q\.":                                      r"final_target_proj.",
            # fmt: on
        }
