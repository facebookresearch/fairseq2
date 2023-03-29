# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import sys
from hashlib import sha1
from pathlib import Path, PurePath
from tempfile import NamedTemporaryFile
from typing import NoReturn, Optional
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
from tqdm import tqdm  # type: ignore[import]


def download_checkpoint(
    url: str,
    model_name: str,
    checkpoint_name: Optional[str] = None,
    shard_idx: Optional[int] = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download the checkpoint at ``url`` to the Torch Hub cache directory.

    :param url:
        The URL to download from.
    :param model_name:
        The name of the associated model.
    :param checkpoint_name:
        The name of the checkpoint.
    :param shard_idx:
        The shard to download if the checkpoint is sharded.
    :param force:
        If ``True``, downloads the checkpoint even if it is already in cache.
    :param progress:
        If ``True``, displays a progress bar to stderr.

    :returns:
        The pathname of the downloaded checkpoint.
    """
    if shard_idx is None:
        actual_url = url
    else:
        actual_url = url.format(shard_idx)

    pathname = _get_cached_pathname(actual_url, sub_dir="fairseq2/checkpoints")

    if not checkpoint_name or checkpoint_name == "default":
        display_name = f"{model_name} checkpoint"
    else:
        display_name = f"{model_name} {checkpoint_name} checkpoint"

    # If the actual and original URLs are not equal, it means a shard index is
    # specified and the original URL has a format field for it.
    if actual_url != url:
        display_name = f"{display_name} (shard {shard_idx})"

    _download_file(url, pathname, display_name, force, progress)

    return pathname


def download_tokenizer(
    url: str,
    model_name: str,
    tokenizer_name: Optional[str] = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download the tokenizer at ``url`` to the Torch Hub cache directory.

    :param url:
        The URL to download from.
    :param model_name:
        The name of the associated model.
    :param tokenizer_name:
        The name of the tokenizer.
    :param force:
        If ``True``, downloads the tokenizer even if it is already in cache.
    :param progress:
        If ``True``, displays a progress bar to stderr.

    :returns:
        The pathname of the downloaded tokenizer.
    """
    pathname = _get_cached_pathname(url, sub_dir="fairseq2/tokenizers")

    if not tokenizer_name or tokenizer_name == "default":
        display_name = f"{model_name} tokenizer"
    else:
        display_name = f"{model_name} {tokenizer_name} tokenizer"

    _download_file(url, pathname, display_name, force, progress)

    return pathname


class DownloadError(RuntimeError):
    """Raised when a download operation fails."""

    def __init__(self, display_name: str, msg: str) -> None:
        super().__init__(
            f"Failed to download the {display_name}. {msg} Please try again and, if the problem persists, file a bug report."
        )


def _get_cached_pathname(url: str, sub_dir: str) -> Path:
    h = _get_url_hash(url)

    hub_dir = torch.hub.get_dir()

    cache_pathname = Path(hub_dir).expanduser().joinpath(sub_dir).joinpath(h)

    try:
        cache_pathname.mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        raise RuntimeError(
            f"Failed to create the '{sub_dir}' Torch Hub cache directory."
        ) from ex

    filename = _get_filename(url)

    return cache_pathname.joinpath(filename)


def _get_url_hash(url: str) -> str:
    h = sha1(url.encode()).hexdigest()

    return h[:16]


def _get_filename(url: str) -> str:
    try:
        url_parts = urlparse(url)
    except ValueError as ex:
        raise ValueError(f"{url} is not a valid URL.") from ex

    filename = PurePath(url_parts.path).name
    if not filename:
        raise ValueError(f"The path of {url} does not end with a filename.")

    return filename


def _download_file(
    url: str, pathname: Path, display_name: str, force: bool, progress: bool
) -> None:
    def raise_connection_error(cause: HTTPError) -> NoReturn:
        if cause.code == 404 or cause.code >= 500:
            raise DownloadError(
                display_name, "The connection to the server cannot be established."
            ) from cause

        raise

    if not force and pathname.exists():
        # Touch the file so that we can have an LRU list for cache cleanup.
        pathname.touch()

        if progress:
            _print_progress(
                f"Using the cached {display_name}. Set `force=True` to download again."
            )

        return

    if progress:
        _print_progress(f"Downloading the {display_name}...")

    try:
        response = urlopen(url)
    except HTTPError as ex:
        raise_connection_error(ex)

    with response, NamedTemporaryFile(delete=False, dir=pathname.parent) as fp:
        headers = response.info()

        try:
            size = int(headers["Content-Length"])
        except TypeError:
            size = None

        try:
            num_bytes_read = 0

            with tqdm(
                total=size,
                disable=not progress,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                while True:
                    try:
                        buffer = response.read(1024 * 8)
                    except HTTPError as ex:
                        raise_connection_error(ex)

                    len_buffer = len(buffer)
                    if len_buffer == 0:
                        break

                    if size is not None:
                        num_bytes_read += len_buffer

                        if num_bytes_read > size:
                            msg = f"The number of bytes sent by the server exceeds the expected size ({size:,})."

                            raise DownloadError(display_name, msg)

                    fp.write(buffer)

                    bar.update(len_buffer)

            if size is not None and num_bytes_read < size:
                msg = f"The number of bytes sent by the server ({num_bytes_read:,}) is less than the expected size ({size:,})."

                raise DownloadError(display_name, msg)

            fp.close()

            shutil.move(fp.name, pathname)
        except:
            fp.close()

            try:
                os.unlink(fp.name)
            except OSError:
                pass

            raise


def _print_progress(s: str) -> None:
    print(s, file=sys.stderr)
