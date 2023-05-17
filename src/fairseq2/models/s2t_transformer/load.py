# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping, Optional, Tuple
from zipfile import ZipFile

import torch

from fairseq2 import services
from fairseq2.assets import AssetCard, AssetDownloadManager, AssetStore
from fairseq2.models.s2t_transformer.build import (
    S2TTransformerConfig,
    create_s2t_transformer_model,
    get_s2t_transformer_archs,
    get_s2t_transformer_config,
)
from fairseq2.models.s2t_transformer.tokenizer import S2TTransformerTokenizer
from fairseq2.models.transformer import TransformerModel
from fairseq2.models.utils.load import (
    MapLocation,
    load_checkpoint,
    upgrade_fairseq_checkpoint,
)


def load_s2t_transformer_model(
    model_name: str, device: Optional[torch.device] = None, progress: bool = True
) -> Tuple[TransformerModel, S2TTransformerTokenizer]:
    """Load the specified S2T Transformer model.

    :param model_name:
        The name of the model.
    :param device:
        The device on which to initialize the model.
    :param progress:
        If ``True``, displays a progress bar to stderr.

    :returns:
        The model and its associated target tokenizer.
    """
    card = services.get(AssetStore).retrieve_card(model_name)

    return S2TTransformerLoader(card, progress=progress).load_model(device=device)


class S2TTransformerLoader:
    """Loads a specified S2T Transformer model."""

    card: AssetCard
    download_manager: AssetDownloadManager
    force: bool
    progress: bool
    cfg: S2TTransformerConfig

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

        supported_arch_names = get_s2t_transformer_archs()

        arch_name = self.card.field("model_arch").as_one_of(supported_arch_names)

        self.cfg = get_s2t_transformer_config(arch_name)

    def load_model(
        self, device: Optional[torch.device] = None
    ) -> Tuple[TransformerModel, S2TTransformerTokenizer]:
        """Load the S2T Transformer model.

        :param device:
            The device on which to initialize the model.

        :returns:
            The model and its associated target tokenizer.
        """
        target_tokenizer = self.load_target_tokenizer()

        # TODO: Initialize on Meta device!
        model = create_s2t_transformer_model(
            self.cfg, target_tokenizer.vocab_info, device
        )

        checkpoint = self.load_checkpoint(map_location="cpu")

        model.load_state_dict(checkpoint["model"])

        return model, target_tokenizer

    def load_checkpoint(self, map_location: MapLocation = None) -> Mapping[str, Any]:
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
            upgrader=self._upgrade_checkpoint,
        )

    def load_target_tokenizer(self) -> S2TTransformerTokenizer:
        """Load the target tokenizer of the S2T Transformer model."""
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
                    f"The load of the target tokenizer of the model '{self.card.name}' has failed. Please file a bug report."
                ) from ex

        task = self.card.field("task").as_one_of({"transcription", "translation"})

        tgt_langs = self.card.field("tgt_langs").as_list(str)

        return S2TTransformerTokenizer(
            pathname, task, set(tgt_langs), default_tgt_lang=tgt_langs[0]
        )

    @classmethod
    def _upgrade_checkpoint(cls, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        key_map = cls._fairseq_key_map()

        return upgrade_fairseq_checkpoint(checkpoint, key_map)

    @staticmethod
    def _fairseq_key_map() -> Dict[str, str]:
        return {
            # fmt: off
            r"^encoder\.subsample\.conv_layers\.([0-9]+)\.":                    r"encoder_frontend.feature_extractor.layers.\1.conv.",
            r"^encoder\.transformer_layers\.([0-9]+)\.self_attn_layer_norm\.":  r"encoder.layers.\1.self_attn_layer_norm.",
            r"^encoder\.transformer_layers\.([0-9]+)\.self_attn\.out_proj\.":   r"encoder.layers.\1.self_attn.output_proj.",
            r"^encoder\.transformer_layers\.([0-9]+)\.self_attn\.":             r"encoder.layers.\1.self_attn.",
            r"^encoder\.transformer_layers\.([0-9]+)\.final_layer_norm\.":      r"encoder.layers.\1.ffn_layer_norm.",
            r"^encoder\.transformer_layers\.([0-9]+)\.fc1\.":                   r"encoder.layers.\1.ffn.inner_proj.",
            r"^encoder\.transformer_layers\.([0-9]+)\.fc2\.":                   r"encoder.layers.\1.ffn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":            r"decoder.layers.\1.encoder_decoder_attn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":            r"decoder.layers.\1.encoder_decoder_attn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":               r"decoder.layers.\1.self_attn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn\.":                      r"decoder.layers.\1.encoder_decoder_attn.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.":           r"decoder.layers.\1.encoder_decoder_attn_layer_norm.",
            r"^decoder\.layers\.([0-9]+)\.fc1\.":                               r"decoder.layers.\1.ffn.inner_proj.",
            r"^decoder\.layers\.([0-9]+)\.fc2\.":                               r"decoder.layers.\1.ffn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.final_layer_norm\.":                  r"decoder.layers.\1.ffn_layer_norm.",
            r"^decoder\.embed_tokens\.":                                        r"decoder_frontend.embed.",
            r"^decoder\.output_projection\.":                                   r"final_proj.",

            # S2T Conformer
            r"^encoder\.linear\.":                                                   r"encoder_frontend.proj.",
            r"^encoder\.conformer_layers\.([0-9]+)\.ffn(1|2)\.layer_norm\.":         r"encoder.layers.\1.ffn\2_layer_norm.",
            r"^encoder\.conformer_layers\.([0-9]+)\.ffn(1|2)\.w_1\.":                r"encoder.layers.\1.ffn\2.inner_proj.",
            r"^encoder\.conformer_layers\.([0-9]+)\.ffn(1|2)\.w_2\.":                r"encoder.layers.\1.ffn\2.output_proj.",
            r"^encoder\.conformer_layers\.([0-9]+)\.self_attn_layer_norm\.":         r"encoder.layers.\1.self_attn_layer_norm.",
            r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.linear_q\.":          r"encoder.layers.\1.self_attn.q_proj.",
            r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.linear_k\.":          r"encoder.layers.\1.self_attn.k_proj.",
            r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.linear_v\.":          r"encoder.layers.\1.self_attn.v_proj.",
            r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.linear_out\.":        r"encoder.layers.\1.self_attn.output_proj.",
            r"^encoder\.conformer_layers\.([0-9]+)\.conv_module\.layer_norm\.":      r"encoder.layers.\1.conv_layer_norm.",
            r"^encoder\.conformer_layers\.([0-9]+)\.conv_module\.pointwise_conv1\.": r"encoder.layers.\1.conv.pointwise_conv1.",
            r"^encoder\.conformer_layers\.([0-9]+)\.conv_module\.depthwise_conv\.":  r"encoder.layers.\1.conv.depthwise_conv.",
            r"^encoder\.conformer_layers\.([0-9]+)\.conv_module\.batch_norm\.":      r"encoder.layers.\1.conv.batch_norm.",
            r"^encoder\.conformer_layers\.([0-9]+)\.conv_module\.pointwise_conv2\.": r"encoder.layers.\1.conv.pointwise_conv2.",
            r"^encoder\.conformer_layers\.([0-9]+)\.final_layer_norm\.":             r"encoder.layers.\1.layer_norm.",

            # S2T Conformer - RelPos
            r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.pos_bias_u":   r"encoder.layers.\1.self_attn.sdpa.u_bias",
            r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.pos_bias_v":   r"encoder.layers.\1.self_attn.sdpa.v_bias",
            r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.linear_pos\.": r"encoder.layers.\1.self_attn.sdpa.r_proj.",
            # fmt: on
        }
