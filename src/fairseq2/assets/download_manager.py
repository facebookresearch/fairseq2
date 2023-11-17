# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import sys
from abc import ABC, abstractmethod
from contextlib import ExitStack
from hashlib import sha1
from pathlib import Path, PurePath
from tarfile import TarFile, is_tarfile
from tempfile import NamedTemporaryFile
from typing import Dict, Optional, final
from urllib.error import HTTPError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
from zipfile import ZipFile

import torch
from tqdm import tqdm  # type: ignore[import]

from fairseq2.assets.error import AssetError
from fairseq2.typing import finaloverride


class AssetDownloadManager(ABC):
    """Downloads and caches assets."""

    @abstractmethod
    def download_checkpoint(
        self,
        uri: str,
        model_name: str,
        *,
        shard_idx: Optional[int] = None,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        """Download the checkpoint at ``uri`` to the asset cache directory.

        :param uri:
            The URI to download from.
        :param model_name:
            The name of the associated model.
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
        *,
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

    @abstractmethod
    def download_dataset(
        self,
        uri: str,
        dataset_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        """Download the dataset at ``uri`` to the asset cache directory.

        :param uri:
            The URI to download from.
        :param data_name:
            The name of the dataset.
        :param force:
            If ``True``, downloads the dataset even if it is already in cache.
        :param progress:
            If ``True``, displays a progress bar to stderr.

        :returns:
            The pathname of the downloaded dataset.
        """


@final
class DefaultAssetDownloadManager(AssetDownloadManager):
    """Downloads and caches assets to the file system."""

    @finaloverride
    def download_checkpoint(
        self,
        uri: str,
        model_name: str,
        *,
        shard_idx: Optional[int] = None,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        display_name = f"checkpoint of {model_name}"

        if shard_idx is not None:
            display_name = f"{display_name} (shard {shard_idx})"

        op = _AssetDownloadOp(uri, display_name, force, progress, shard_idx)

        return op.run()

    @finaloverride
    def download_tokenizer(
        self,
        uri: str,
        model_name: str,
        *,
        tokenizer_name: Optional[str] = None,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        if not tokenizer_name:
            display_name = f"tokenizer of {model_name}"
        else:
            display_name = f"{tokenizer_name} tokenizer of {model_name}"

        op = _AssetDownloadOp(uri, display_name, force, progress)

        return op.run()

    @finaloverride
    def download_dataset(
        self,
        uri: str,
        dataset_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        display_name = f"{dataset_name} dataset"

        op = _AssetDownloadOp(uri, display_name, force, progress)

        return op.run()


class _AssetDownloadOp:
    uri: str
    params: Dict[str, str]
    download_path: Path
    download_filename: str
    display_name: str
    force: bool
    progress: bool
    shard_idx: Optional[int]

    def __init__(
        self,
        uri: str,
        display_name: str,
        force: bool,
        progress: bool,
        shard_idx: Optional[int] = None,
    ) -> None:
        self.uri = uri
        self.params = {}
        self.download_path = Path()
        self.download_filename = ""
        self.display_name = display_name
        self.force = force
        self.progress = progress
        self.shard_idx = shard_idx

    def run(self) -> Path:
        self._process_input_uri()

        self._format_uri_with_shard_idx()

        self._check_if_gated_asset()

        if (asset_path := self._try_uri_as_path()) is not None:
            return asset_path

        self._init_download_path()

        self._download_asset()

        self._ensure_asset_extracted()

        return self._get_final_asset_path()

    def _process_input_uri(self) -> None:
        if self.uri.find("://") >= 0:
            uri = self.uri
        else:
            uri = "file://" + self.uri

        try:
            parsed_uri = urlparse(uri)
        except ValueError as ex:
            raise ValueError(
                f"`uri` must be a valid URI, but is '{uri}' instead."
            ) from ex

        if parsed_uri.params:
            for param in parsed_uri.params.split(";"):
                key_value_pair = param.split("=")
                if len(key_value_pair) != 2:
                    raise ValueError(
                        f"`uri` must be a valid URI, but is '{uri}' instead."
                    )

                key, value = key_value_pair

                key = unquote(key).strip()
                if len(key) == 0:
                    raise ValueError(
                        f"`uri` must be a valid URI, but is '{uri}' instead."
                    )

                self.params[key.lower()] = unquote(value).strip()

        if parsed_uri.scheme == "file" and parsed_uri.netloc:
            raise ValueError(
                f"`uri` has the file scheme and must have an absolute pathname, but is '{uri}' instead."
            )

        path = PurePath(parsed_uri.path)

        self.download_filename = path.name
        if not self.download_filename:
            raise ValueError(f"`uri` must point to a file, but is '{uri}' instead.")

        self.uri = parsed_uri._replace(params="").geturl()

    def _format_uri_with_shard_idx(self) -> None:
        if self.shard_idx is None:
            return

        uri_with_shard = self.uri.format(self.shard_idx)

        if uri_with_shard == self.uri:
            raise AssetError(
                f"`shard_idx` is specified, but the {self.display_name} is not sharded."
            )

        self.uri = uri_with_shard

    def _check_if_gated_asset(self) -> None:
        if self.params.get("gated", "false").strip().lower() == "true":
            raise AssetError(
                f"The {self.display_name} is gated. Please visit {self.uri} to learn how to get access."
            )

    def _try_uri_as_path(self) -> Optional[Path]:
        if self.uri.startswith("file://"):
            return Path(self.uri[7:])

        return None

    def _init_download_path(self) -> None:
        cache_root_pathname = os.getenv("FAIRSEQ2_CACHE_DIR")
        if cache_root_pathname:
            try:
                cache_root_path = Path(cache_root_pathname)
            except ValueError as ex:
                raise RuntimeError(
                    f"`FAIRSEQ2_CACHE_DIR` environment variable must contain a valid pathname, but contains '{cache_root_pathname}' instead."
                ) from ex
        else:
            cache_root_path = Path(torch.hub.get_dir()).joinpath("fairseq2")

        hash_ = sha1(self.uri.encode()).hexdigest()

        hash_ = hash_[:24]

        self.download_path = cache_root_path.expanduser().joinpath(
            "assets", hash_, self.download_filename
        )

    def _download_asset(self) -> None:
        dir_path = self.download_path.parent

        if dir_path.exists():
            if not self.force:
                # Touch the cache directory so that we can maintain an LRU list
                # for cache cleanup.
                try:
                    dir_path.touch()
                except OSError:
                    pass

                if self.progress:
                    self._print_progress(
                        f"Using the cached {self.display_name}. Set `force` to `True` to download again."
                    )

                return

            if self.progress:
                self._print_progress(
                    f"Ignoring the cached {self.display_name}. `force` is set to `True`."
                )

            try:
                shutil.rmtree(dir_path)
            except OSError as ex:
                raise AssetDownloadError(
                    f"The asset cache directory for {self.display_name} cannot be cleaned up. See nested exception for details."
                ) from ex

        succeeded = False

        with ExitStack() as cleanup_stack:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except OSError as ex:
                raise AssetDownloadError(
                    f"The asset cache directory for {self.display_name} cannot be created. See nested exception for details."
                ) from ex

            def remove_cache_dir() -> None:
                if not succeeded:
                    try:
                        shutil.rmtree(dir_path)
                    except OSError:
                        pass

            cleanup_stack.callback(remove_cache_dir)

            if self.progress:
                self._print_progress(f"Downloading the {self.display_name}...")

            request = Request(
                self.uri,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux i686; rv:109.0) Gecko/20100101 Firefox/119.0"
                },
            )

            try:
                response = cleanup_stack.enter_context(urlopen(request))
            except HTTPError as ex:
                raise AssetDownloadError(
                    f"The download of the {self.display_name} has failed with the HTTP error code {ex.code}."
                )

            headers = response.info()

            try:
                size = int(headers["Content-Length"])
            except TypeError:
                size = None

            fp = cleanup_stack.enter_context(
                NamedTemporaryFile(delete=False, dir=dir_path)
            )

            num_bytes_read = 0

            bar = cleanup_stack.enter_context(
                tqdm(
                    total=size,
                    disable=not self.progress,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                )
            )

            while True:
                try:
                    buffer = response.read(1024 * 8)
                except HTTPError as ex:
                    raise AssetDownloadError(
                        f"The download of the {self.display_name} has failed with the HTTP error code {ex.code}."
                    )

                len_buffer = len(buffer)
                if len_buffer == 0:
                    break

                if size is not None:
                    num_bytes_read += len_buffer
                    if num_bytes_read > size:
                        raise AssetDownloadError(
                            f"The download of the {self.display_name} has failed. The number of bytes sent by the server exceeded the expected size of {size:,} bytes."
                        )

                fp.write(buffer)

                bar.update(len_buffer)

            if size is not None and num_bytes_read < size:
                raise AssetDownloadError(
                    f"The download of the {self.display_name} has failed. The server sent {num_bytes_read:,} bytes which is less than the expected size of {size:,} bytes."
                )

            fp.close()

            shutil.move(fp.name, self.download_path)

            succeeded = True

    def _ensure_asset_extracted(self) -> None:
        download_path = self.download_path

        if not download_path.exists():
            return

        if download_path.suffix == ".zip":
            if self.progress:
                self._print_progress(f"Extracting the {self.display_name}...")

            try:
                with ZipFile(download_path) as fp:
                    fp.extractall(path=download_path.parent)
            except (KeyError, IOError) as ex:
                raise AssetError(
                    f"The zip file of the {self.display_name} cannot be extracted. See nested exception for details."
                ) from ex

            try:
                os.unlink(download_path)
            except OSError:
                pass
        elif is_tarfile(download_path):
            if self.progress:
                self._print_progress(f"Extracting the {self.display_name}...")

            try:
                with TarFile(download_path) as fp:
                    fp.extractall(path=download_path.parent)
            except (KeyError, IOError) as ex:
                raise AssetError(
                    f"The tar file of the {self.display_name} cannot be extracted. See nested exception for details."
                ) from ex

            try:
                os.unlink(download_path)
            except OSError:
                pass

    def _get_final_asset_path(self) -> Path:
        if self.download_path.exists():
            return self.download_path

        dir_path = self.download_path.parent

        rel_member_pathname = self.params.get("member", "")
        if len(rel_member_pathname) == 0:
            return dir_path

        member_path = dir_path.joinpath(rel_member_pathname).resolve()

        try:
            member_path.relative_to(dir_path)
        except ValueError:
            raise AssetError(
                f"The 'member' parameter of '{self.uri}' points to a path ({rel_member_pathname}) outside of the asset cache directory."
            )

        return member_path

    @staticmethod
    def _print_progress(s: str) -> None:
        print(s, file=sys.stderr)


class AssetDownloadError(AssetError):
    """Raised when a download operation fails."""


download_manager = DefaultAssetDownloadManager()
