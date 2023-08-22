# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import sys
from abc import ABC, abstractmethod
from hashlib import sha1
from pathlib import Path, PurePath
from tempfile import NamedTemporaryFile
from typing import NoReturn, Optional, final
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
from tqdm import tqdm  # type: ignore[import]

from fairseq2.assets.error import AssetError
from fairseq2.typing import finaloverride


class AssetDownloadManager(ABC):
    @abstractmethod
    def download_checkpoint(
        self,
        uri: str,
        model_name: str,
        checkpoint_name: Optional[str] = None,
        shard_idx: Optional[int] = None,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        """Download the checkpoint at ``uri`` to the asset cache directory.

        :param uri:
            The URI to download from.
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

    @abstractmethod
    def download_tokenizer(
        self,
        uri: str,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        """Download the tokenizer at ``uri`` to the asset cache directory.

        :param uri:
            The URI to download from.
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


@final
class DefaultAssetDownloadManager(AssetDownloadManager):
    @finaloverride
    def download_checkpoint(
        self,
        uri: str,
        model_name: str,
        checkpoint_name: Optional[str] = None,
        shard_idx: Optional[int] = None,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        if shard_idx is not None:
            formatted_uri = uri.format(shard_idx)

            if formatted_uri == uri:
                raise ValueError(
                    "The checkpoint is not sharded, but a `shard_idx` is specified."
                )

            uri = formatted_uri

        pathname = self._try_as_pathname(uri)
        if pathname:
            return pathname

        if not checkpoint_name:
            display_name = f"checkpoint of the model '{model_name}'"
        else:
            display_name = f"'{checkpoint_name}' checkpoint of the model '{model_name}'"

        if shard_idx is not None:
            display_name = f"{display_name} (shard {shard_idx})"

        pathname = self._get_pathname(uri, sub_dir="checkpoints")

        self._download_file(uri, pathname, display_name, force, progress)

        return pathname

    @finaloverride
    def download_tokenizer(
        self,
        uri: str,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        pathname = self._try_as_pathname(uri)
        if pathname:
            return pathname

        if not tokenizer_name:
            display_name = f"tokenizer of the model '{model_name}'"
        else:
            display_name = f"'{tokenizer_name}' tokenizer of the model '{model_name}'"

        pathname = self._get_pathname(uri, sub_dir="tokenizers")

        self._download_file(uri, pathname, display_name, force, progress)

        return pathname

    @staticmethod
    def _try_as_pathname(uri: str) -> Optional[Path]:
        if uri.startswith("file://"):
            return Path(uri[6:])

        return None

    @classmethod
    def _get_pathname(cls, uri: str, sub_dir: str) -> Path:
        hub_dir = Path(torch.hub.get_dir()).expanduser()

        hsh = cls._get_uri_hash(uri)

        filename = cls._get_filename(uri)

        return hub_dir.joinpath("fairseq2", "assets", sub_dir, hsh, filename)

    @staticmethod
    def _get_uri_hash(uri: str) -> str:
        h = sha1(uri.encode()).hexdigest()

        return h[:24]

    @staticmethod
    def _get_filename(uri: str) -> str:
        try:
            uri_parts = urlparse(uri)
        except ValueError as ex:
            raise ValueError(
                f"`uri` must be a valid URI, but is '{uri}' instead."
            ) from ex

        filename = PurePath(uri_parts.path).name
        if not filename:
            raise ValueError(
                f"The path component of the URI '{uri}' must end with a filename."
            )

        return filename

    def _download_file(
        self, uri: str, pathname: Path, display_name: str, force: bool, progress: bool
    ) -> None:
        def raise_connection_error(cause: HTTPError) -> NoReturn:
            if cause.code == 404 or cause.code >= 500:
                raise AssetDownloadError(
                    display_name, "The connection to the server cannot be established."
                ) from cause

            raise

        if not force and pathname.exists():
            # Touch the file so that we can maintain an LRU list for cache cleanup.
            pathname.touch()

            if progress:
                _print_progress(
                    f"Using the cached {display_name}. Set `force=True` to download again."
                )

            return

        if not pathname.parent.exists():
            try:
                pathname.parent.mkdir(parents=True, exist_ok=True)
            except OSError as ex:
                raise RuntimeError(
                    f"The creation of the asset cache directory for the URI '{uri}' has failed."
                ) from ex

        if progress:
            _print_progress(f"Downloading the {display_name}...")

        try:
            response = urlopen(uri)
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
                                msg = f"The number of bytes sent by the server exceeds the expected size of {size:,} bytes."

                                raise AssetDownloadError(display_name, msg)

                        fp.write(buffer)

                        bar.update(len_buffer)

                if size is not None and num_bytes_read < size:
                    msg = f"The server has sent {num_bytes_read:,} bytes which is less than the expected size of {size:,} bytes."

                    raise AssetDownloadError(display_name, msg)

                fp.close()

                shutil.move(fp.name, pathname)
            except:
                fp.close()

                try:
                    os.unlink(fp.name)
                except OSError:
                    pass

                raise


class AssetDownloadError(AssetError):
    """Raised when a download operation fails."""

    def __init__(self, display_name: str, msg: str) -> None:
        super().__init__(
            f"The download of the {display_name} has failed. {msg} Please try again and, if the problem persists, file a bug report."
        )


def _print_progress(s: str) -> None:
    print(s, file=sys.stderr)


download_manager = DefaultAssetDownloadManager()
