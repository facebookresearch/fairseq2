# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping, Union, final
from zipfile import ZipFile

from fairseq2.assets import (
    AssetCard,
    AssetDownloadManager,
    AssetStore,
    asset_store,
    download_manager,
)
from fairseq2.models.s2t_transformer.builder import (
    S2TTransformerConfig,
    create_s2t_transformer_model,
    s2t_transformer_archs,
)
from fairseq2.models.s2t_transformer.tokenizer import S2TTransformerTokenizer
from fairseq2.models.transformer import TransformerModel
from fairseq2.models.utils.checkpoint_loader import upgrade_fairseq_checkpoint
from fairseq2.models.utils.model_loader import ModelConfigLoader, ModelLoader
from fairseq2.typing import finaloverride


@final
class S2TTransformerLoader(ModelLoader[TransformerModel, S2TTransformerConfig]):
    """Loads S2T Transformer models."""

    @finaloverride
    def _upgrade_checkpoint(
        self, checkpoint: Mapping[str, Any], config: S2TTransformerConfig
    ) -> Mapping[str, Any]:
        key_map = self._fairseq_key_map()

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


load_s2t_transformer_model = S2TTransformerLoader(
    asset_store, download_manager, create_s2t_transformer_model, s2t_transformer_archs
)


load_s2t_transformer_config = ModelConfigLoader[S2TTransformerConfig](
    asset_store, s2t_transformer_archs
)


class S2TTransformerTokenizerLoader:
    """Loads target tokenizers of S2T Transformer models."""

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
    ) -> S2TTransformerTokenizer:
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

        zip_pathname = self.download_manager.download_tokenizer(
            uri, card.name, force=force, progress=progress
        )

        # The tokenizer is stored in a zip file, so we have to extract it first.
        filename = card.field("tokenizer_file").as_filename()

        pathname = zip_pathname.with_name(filename)

        if force or not pathname.exists():
            try:
                with ZipFile(zip_pathname) as fp:
                    fp.extract(filename, path=zip_pathname.parent)
            except (KeyError, IOError) as ex:
                raise RuntimeError(
                    f"The load of the target tokenizer of the model '{card.name}' has failed. Please file a bug report."
                ) from ex

        # The valid task names ares transcription and translation.
        task = card.field("task").as_one_of({"transcription", "translation"})

        target_langs = card.field("tgt_langs").as_list(str)

        return S2TTransformerTokenizer(
            pathname, task, set(target_langs), default_target_lang=target_langs[0]
        )


load_s2t_transformer_tokenizer = S2TTransformerTokenizerLoader(
    asset_store, download_manager
)
