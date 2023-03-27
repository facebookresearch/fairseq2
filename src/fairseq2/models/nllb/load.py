# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import OrderedDict
from typing import Any, Dict, Final, Mapping, Optional, Sequence, Tuple

import torch
from torch.serialization import MAP_LOCATION

from fairseq2.data.typing import PathLike
from fairseq2.models.nllb.build import create_nllb_model, get_nllb_config
from fairseq2.models.nllb.tokenizer import NllbTokenizer
from fairseq2.models.transformer import TransformerModel
from fairseq2.models.utils.download import download_model, download_tokenizer
from fairseq2.models.utils.load import load_parameters

# fmt: off

_ARCHS: Final = {
    "dense_1b":           "1b",
    "dense_3b":           "3b",
    "dense_distill_1b":   "1b",
    "dense_distill_600m": "600m",
}

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
    try:
        arch_name = _ARCHS[variant]
    except KeyError:
        raise ValueError(f"{variant} is not a known NLLB model variant name.")

    cfg = get_nllb_config(arch_name)

    tokenizer = load_nllb_tokenizer(variant, progress=progress)

    # TODO: Initialize on Meta device!
    model = create_nllb_model(cfg, tokenizer.vocab_info, device)

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


def _upgrade_parameters(params: Mapping[str, Any]) -> Dict[str, Any]:
    # We only care about the model parameters and buffers. The rest of the
    # checkpoint is not relevant for us.
    old_params = params["model"]

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


_TOKENIZER: Final = "https://tinyurl.com/flores200sacrebleuspm"

_FAIRCLUSTER_TOKENIZER: Final = "/large_experiments/seamless/nllb/opensource/spm_200/sentencepiece.source.256000.model"


def load_nllb_tokenizer(variant: str, progress: bool = True) -> NllbTokenizer:
    """Load the NLLB tokenizer.

    :param variant:
        The model variant.
    :param progress:
        If ``True``, displays a progress bar to stderr.
    """
    pathname: PathLike

    if "FAIR_ENV_CLUSTER" not in os.environ:
        pathname = download_tokenizer(
            _TOKENIZER, tokenizer_name="NLLB", sub_dir="nllb", progress=progress
        )
    else:
        pathname = _FAIRCLUSTER_TOKENIZER

    langs = _get_all_langs()

    return NllbTokenizer(pathname, langs, default_lang="eng_Latn")


def _get_all_langs() -> Sequence[str]:
    return [
        "ace_Arab",
        "ace_Latn",
        "acm_Arab",
        "acq_Arab",
        "aeb_Arab",
        "afr_Latn",
        "ajp_Arab",
        "aka_Latn",
        "amh_Ethi",
        "apc_Arab",
        "arb_Arab",
        "ars_Arab",
        "ary_Arab",
        "arz_Arab",
        "asm_Beng",
        "ast_Latn",
        "awa_Deva",
        "ayr_Latn",
        "azb_Arab",
        "azj_Latn",
        "bak_Cyrl",
        "bam_Latn",
        "ban_Latn",
        "bel_Cyrl",
        "bem_Latn",
        "ben_Beng",
        "bho_Deva",
        "bjn_Arab",
        "bjn_Latn",
        "bod_Tibt",
        "bos_Latn",
        "bug_Latn",
        "bul_Cyrl",
        "cat_Latn",
        "ceb_Latn",
        "ces_Latn",
        "cjk_Latn",
        "ckb_Arab",
        "crh_Latn",
        "cym_Latn",
        "dan_Latn",
        "deu_Latn",
        "dik_Latn",
        "dyu_Latn",
        "dzo_Tibt",
        "ell_Grek",
        "eng_Latn",
        "epo_Latn",
        "est_Latn",
        "eus_Latn",
        "ewe_Latn",
        "fao_Latn",
        "pes_Arab",
        "fij_Latn",
        "fin_Latn",
        "fon_Latn",
        "fra_Latn",
        "fur_Latn",
        "fuv_Latn",
        "gla_Latn",
        "gle_Latn",
        "glg_Latn",
        "grn_Latn",
        "guj_Gujr",
        "hat_Latn",
        "hau_Latn",
        "heb_Hebr",
        "hin_Deva",
        "hne_Deva",
        "hrv_Latn",
        "hun_Latn",
        "hye_Armn",
        "ibo_Latn",
        "ilo_Latn",
        "ind_Latn",
        "isl_Latn",
        "ita_Latn",
        "jav_Latn",
        "jpn_Jpan",
        "kab_Latn",
        "kac_Latn",
        "kam_Latn",
        "kan_Knda",
        "kas_Arab",
        "kas_Deva",
        "kat_Geor",
        "knc_Arab",
        "knc_Latn",
        "kaz_Cyrl",
        "kbp_Latn",
        "kea_Latn",
        "khm_Khmr",
        "kik_Latn",
        "kin_Latn",
        "kir_Cyrl",
        "kmb_Latn",
        "kon_Latn",
        "kor_Hang",
        "kmr_Latn",
        "lao_Laoo",
        "lvs_Latn",
        "lij_Latn",
        "lim_Latn",
        "lin_Latn",
        "lit_Latn",
        "lmo_Latn",
        "ltg_Latn",
        "ltz_Latn",
        "lua_Latn",
        "lug_Latn",
        "luo_Latn",
        "lus_Latn",
        "mag_Deva",
        "mai_Deva",
        "mal_Mlym",
        "mar_Deva",
        "min_Latn",
        "mkd_Cyrl",
        "plt_Latn",
        "mlt_Latn",
        "mni_Beng",
        "khk_Cyrl",
        "mos_Latn",
        "mri_Latn",
        "zsm_Latn",
        "mya_Mymr",
        "nld_Latn",
        "nno_Latn",
        "nob_Latn",
        "npi_Deva",
        "nso_Latn",
        "nus_Latn",
        "nya_Latn",
        "oci_Latn",
        "gaz_Latn",
        "ory_Orya",
        "pag_Latn",
        "pan_Guru",
        "pap_Latn",
        "pol_Latn",
        "por_Latn",
        "prs_Arab",
        "pbt_Arab",
        "quy_Latn",
        "ron_Latn",
        "run_Latn",
        "rus_Cyrl",
        "sag_Latn",
        "san_Deva",
        "sat_Beng",
        "scn_Latn",
        "shn_Mymr",
        "sin_Sinh",
        "slk_Latn",
        "slv_Latn",
        "smo_Latn",
        "sna_Latn",
        "snd_Arab",
        "som_Latn",
        "sot_Latn",
        "spa_Latn",
        "als_Latn",
        "srd_Latn",
        "srp_Cyrl",
        "ssw_Latn",
        "sun_Latn",
        "swe_Latn",
        "swh_Latn",
        "szl_Latn",
        "tam_Taml",
        "tat_Cyrl",
        "tel_Telu",
        "tgk_Cyrl",
        "tgl_Latn",
        "tha_Thai",
        "tir_Ethi",
        "taq_Latn",
        "taq_Tfng",
        "tpi_Latn",
        "tsn_Latn",
        "tso_Latn",
        "tuk_Latn",
        "tum_Latn",
        "tur_Latn",
        "twi_Latn",
        "tzm_Tfng",
        "uig_Arab",
        "ukr_Cyrl",
        "umb_Latn",
        "urd_Arab",
        "uzn_Latn",
        "vec_Latn",
        "vie_Latn",
        "war_Latn",
        "wol_Latn",
        "xho_Latn",
        "ydd_Hebr",
        "yor_Latn",
        "yue_Hant",
        "zho_Hans",
        "zho_Hant",
        "zul_Latn",
    ]
