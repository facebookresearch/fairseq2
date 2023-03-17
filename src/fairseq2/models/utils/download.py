# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path, PurePath
from urllib.error import HTTPError
from urllib.parse import urlparse

import torch


def download_model(
    url: str, model_name: str, sub_dir: str, progress: bool = True
) -> Path:
    """Download the model at ``url`` to the Torch Hub cache directory.

    :param url:
        The URL to download from.
    :param model_name:
        The model name to display in progress and error messages.
    :param sub_dir:
        The cache directory under which to store the model.
    :param progress:
        If ``True``, displays a progress bar to stderr.

    :returns:
        The pathname of the downloaded model.
    """
    pathname = _get_cached_pathname(url, f"fairseq2/models/{sub_dir}")

    if pathname.exists():
        if progress:
            _print_progress(f"Using the previously cached {model_name} model.")
    else:
        if progress:
            _print_progress(f"Downloading the {model_name} model...")

        try:
            torch.hub.download_url_to_file(url, str(pathname), progress=progress)
        except HTTPError as ex:
            if ex.code == 404 or ex.code >= 500:
                raise RuntimeError(
                    f"The {model_name} model cannot be downloaded. Please file a bug report."
                ) from ex

            raise

    return pathname


def download_tokenizer(
    url: str, tokenizer_name: str, sub_dir: str, progress: bool = True
) -> Path:
    """Download the tokenizer at ``url`` to the Torch Hub cache directory.

    :param url:
        The URL to download from.
    :param tokenizer_name:
        The tokenizer name to display in progress and error messages.
    :param sub_dir:
        The cache directory under which to store the tokenizer.
    :param progress:
        If ``True``, displays a progress bar to stderr.

    :returns:
        The pathname of the downloaded tokenizer.
    """
    pathname = _get_cached_pathname(url, f"fairseq2/tokenizers/{sub_dir}")

    if pathname.exists():
        if progress:
            _print_progress(f"Using the previously cached {tokenizer_name} tokenizer.")
    else:
        if progress:
            _print_progress(f"Downloading the {tokenizer_name} tokenizer...")

        try:
            torch.hub.download_url_to_file(url, str(pathname), progress=progress)
        except HTTPError as ex:
            if ex.code == 404 or ex.code >= 500:
                raise RuntimeError(
                    f"The {tokenizer_name} tokenizer cannot be downloaded. Please file a bug report."
                ) from ex

            raise

    return pathname


def _get_cached_pathname(url: str, sub_dir: str) -> Path:
    """Return the pathname at which to store the file downloaded from ``url``.

    :param url:
        The download URL.
    :param sub_dir:
        The cache directory under which to store the file.
    """
    name = _get_filename(url)

    hub_dir = torch.hub.get_dir()

    cache_pathname = Path(hub_dir).joinpath(sub_dir)

    try:
        cache_pathname.mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        raise RuntimeError(
            f"The Torch Hub cache directory for {url} cannot be created."
        ) from ex

    return cache_pathname.joinpath(name)


def _get_filename(url: str) -> str:
    """Return the filename part of ``url``."""
    try:
        url_parts = urlparse(url)
    except ValueError as ex:
        raise ValueError(f"{url} is not a valid URL.") from ex

    return PurePath(url_parts.path).name


def _print_progress(s: str) -> None:
    print(s, file=sys.stderr)
