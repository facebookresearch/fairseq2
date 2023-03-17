# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import OrderedDict
from typing import Any, Dict, Final, Mapping, Optional, Tuple

import torch
from torch.serialization import MAP_LOCATION

from fairseq2.data.typing import PathLike
from fairseq2.models.nllb.build import create_nllb_model
from fairseq2.models.nllb.config import get_nllb_config
from fairseq2.models.nllb.tokenizer import NllbTokenizer, load_nllb_tokenizer
from fairseq2.models.transformer import TransformerModel
from fairseq2.models.utils.download import download_model
from fairseq2.models.utils.load import load_parameters

# fmt: off

_MODELS: Final = {
    "dense_1b":           "https://tinyurl.com/nllb200dense1bcheckpoint",
    "dense_3b":           "https://tinyurl.com/nllb200dense3bcheckpoint",
    "dense_distill_1b":   "https://tinyurl.com/nllb200densedst1bcheckpoint",
    "dense_distill_600m": "https://tinyurl.com/nllb200densedst600mcheckpoint",
}

_FAIRCLUSTER_MODELS: Final = {
    "dense_1b":           "/large_experiments/seamless/nllb/opensource/nllb_200_dense_1b/checkpoint.pt",
    "dense_3b":           "/large_experiments/seamless/nllb/opensource/nllb_200_dense_3b/checkpoint.pt",
    "dense_distill_1b":   "/large_experiments/seamless/nllb/opensource/nllb_200_dense_distill_1b/checkpoint.pt",
    "dense_distill_600m": "/large_experiments/seamless/nllb/opensource/nllb_200_dense_distill_600m/checkpoint.pt",
}

# fmt: on


def load_nllb_model(
    variant: str, device: Optional[torch.device] = None, progress: bool = True
) -> Tuple[TransformerModel, NllbTokenizer]:
    """Load the specified NLLB model variant.

    :param variant:
        The model variant.
    :param device:
        The device on which to initialize the model.
    :param progress:
        If ``True``, displays a progress bar to stderr.

    :returns:
        The model and its associated tokenizer.
    """
    cfg = get_nllb_config(variant)

    tokenizer = load_nllb_tokenizer(variant, progress=progress)

    # TODO: Initialize on Meta device!
    model = create_nllb_model(cfg, tokenizer, device)

    parameters = load_nllb_parameters(variant, progress=progress)

    # TODO: Sanity check for unused params.
    model.load_state_dict(parameters)

    return model, tokenizer


def load_nllb_parameters(
    variant: str, map_location: MAP_LOCATION = None, progress: bool = True
) -> Mapping[str, Any]:
    """Load the pretrained parameters of the specified NLLB model variant.

    :param variant:
        The model variant.
    :param map_location:
        Same as the ``map_location`` parameter of :meth:`torch.load`.
    :param progress:
        If ``True``, displays a progress bar to stderr.
    """
    pathname: PathLike

    model_name = f"NLLB ({variant})"

    if "FAIR_ENV_CLUSTER" not in os.environ:
        try:
            url = _MODELS[variant]
        except KeyError:
            raise ValueError(f"{variant} is not a known NLLB model variant name.")

        pathname = download_model(url, model_name, sub_dir="nllb", progress=progress)
    else:
        try:
            pathname = _FAIRCLUSTER_MODELS[variant]
        except KeyError:
            raise ValueError(f"{variant} is not a known NLLB model variant name.")

    return load_parameters(
        pathname, model_name, map_location, param_upgrader=_upgrade_parameters
    )


def _upgrade_parameters(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    # We only care about the model parameters and buffers. The rest of the
    # checkpoint is not relevant for us.
    old_params = checkpoint["model"]

    new_params = OrderedDict()

    old_new_key_map = _get_old_new_key_map()

    # Convert module keys from fairseq to fairseq2.
    for key in old_params.keys():
        modified_key = key

        for old, new in old_new_key_map.items():
            modified_key = modified_key.replace(old, new)

        new_params[modified_key] = old_params[key]

    # Use the built-in version attribute of Module.
    del new_params["encoder.version"]
    del new_params["decoder.version"]

    # Positional embeddings don't have to be stored in the state dictionary
    # since we can generate them on-the-fly.
    del new_params["encoder.embed_positions._float_tensor"]
    del new_params["decoder.embed_positions._float_tensor"]

    embeds = new_params["score_proj.weight"]

    # fairseq checkpoints have duplicated embedding weights.
    new_params["encoder_frontend.embed.weight"] = embeds
    new_params["decoder_frontend.embed.weight"] = embeds

    # The embedding positions of the control tokens do not match the
    # SentencePiece model of the tokenizer.
    with torch.inference_mode():
        # (BOS, PAD, EOS, UNK) -> (PAD, UNK, BOS, EOS)
        embeds[[0, 1, 2, 3]] = embeds[[1, 3, 0, 2]]

    return new_params


def _get_old_new_key_map() -> Dict[str, str]:
    return {
        "encoder.embed_tokens.weight": "encoder_frontend.embed.weight",
        "decoder.embed_tokens.weight": "decoder_frontend.embed.weight",
        ".encoder_attn": ".enc_dec_attn",
        ".fc1.": ".ffn.inner_proj.",
        ".fc2.": ".ffn.out_proj.",
        ".final_layer_norm.": ".ffn_layer_norm.",
        "decoder.output_projection.weight": "score_proj.weight",
    }
