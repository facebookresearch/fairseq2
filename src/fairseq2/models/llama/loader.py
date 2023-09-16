# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping, Union, final

from overrides import override as finaloverride

from fairseq2.assets import (
    AssetCard,
    AssetDownloadManager,
    AssetStore,
    asset_store,
    download_manager,
)
from fairseq2.models.llama.builder import LLaMAConfig, create_llama_model, llama_archs
from fairseq2.models.llama.tokenizer import LLaMATokenizer
from fairseq2.models.transformer import TransformerDecoderModel
from fairseq2.models.utils.checkpoint_loader import convert_model_state_dict
from fairseq2.models.utils.model_loader import ModelConfigLoader, ModelLoader


@final
class LLaMALoader(ModelLoader[TransformerDecoderModel, LLaMAConfig]):
    """Loads LLaMA models."""

    @finaloverride
    def _convert_checkpoint(
        self, checkpoint: Mapping[str, Any], config: LLaMAConfig
    ) -> Mapping[str, Any]:
        key_map = self._key_map()

        # We do not need the pre-computed 'rope.freqs' buffers.
        checkpoint = {k: v for (k, v) in checkpoint.items() if "rope.freqs" not in k}

        checkpoint = convert_model_state_dict(checkpoint, key_map)

        return {"model": checkpoint}

    @staticmethod
    def _key_map() -> Dict[str, str]:
        return {
            # fmt: off
            r"^layers\.([0-9]+)\.attention\.wq\.":    r"decoder.layers.\1.self_attn.q_proj.",
            r"^layers\.([0-9]+)\.attention\.wk\.":    r"decoder.layers.\1.self_attn.k_proj.",
            r"^layers\.([0-9]+)\.attention\.wv\.":    r"decoder.layers.\1.self_attn.v_proj.",
            r"^layers\.([0-9]+)\.attention\.wo\.":    r"decoder.layers.\1.self_attn.output_proj.",
            r"^layers\.([0-9]+)\.attention_norm\.":   r"decoder.layers.\1.self_attn_layer_norm.",
            r"^layers\.([0-9]+)\.feed_forward\.w1\.": r"decoder.layers.\1.ffn.gate_proj.",
            r"^layers\.([0-9]+)\.feed_forward\.w2\.": r"decoder.layers.\1.ffn.output_proj.",
            r"^layers\.([0-9]+)\.feed_forward\.w3\.": r"decoder.layers.\1.ffn.inner_proj.",
            r"^layers\.([0-9]+)\.ffn_norm\.":         r"decoder.layers.\1.ffn_layer_norm.",
            r"^norm\.":                               r"decoder.layer_norm.",
            r"^tok_embeddings\.":                     r"decoder_frontend.embed.",
            r"^output\.":                             r"final_proj.",
            # fmt: on
        }


load_llama_model = LLaMALoader(
    asset_store, download_manager, create_llama_model, llama_archs
)


load_llama_config = ModelConfigLoader[LLaMAConfig](asset_store, llama_archs)


class LLaMATokenizerLoader:
    """Loads tokenizers of LLaMA models."""

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
        *,
        force: bool = False,
        progress: bool = True,
    ) -> LLaMATokenizer:
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

        return LLaMATokenizer(pathname)


load_llama_tokenizer = LLaMATokenizerLoader(asset_store, download_manager)
