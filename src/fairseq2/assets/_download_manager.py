# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import ExitStack
from hashlib import sha1
from pathlib import Path
from shutil import rmtree
from tarfile import TarFile, is_tarfile
from tempfile import NamedTemporaryFile
from typing import final
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from tqdm import tqdm  # type: ignore[import]
from typing_extensions import override

from fairseq2.assets._card import _starts_with_scheme
from fairseq2.logging import log


class AssetDownloadManager(ABC):
    """Downloads assets."""

    @abstractmethod
    def download_checkpoint(
        self,
        uri: str,
        model_name: str,
        *,
        shard_idx: int | None = None,
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
            The path to the downloaded checkpoint.
        """

    @abstractmethod
    def download_tokenizer(
        self,
        uri: str,
        model_name: str,
        *,
        tokenizer_name: str | None = None,
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
            The path to the downloaded tokenizer.
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
            The path to the downloaded dataset.
        """


class AssetDownloadError(Exception):
    pass


@final
class InProcAssetDownloadManager(AssetDownloadManager):
    """Downloads assets in this process."""

    _cache_dir: Path

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    @override
    def download_checkpoint(
        self,
        uri: str,
        model_name: str,
        *,
        shard_idx: int | None = None,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        display_name = f"checkpoint of {model_name}"

        if shard_idx is not None:
            display_name = f"{display_name} (shard {shard_idx})"

        op = _AssetDownloadOp(
            self._cache_dir, uri, display_name, force, progress, shard_idx
        )

        return op.run()

    @override
    def download_tokenizer(
        self,
        uri: str,
        model_name: str,
        *,
        tokenizer_name: str | None = None,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        if not tokenizer_name:
            display_name = f"tokenizer of {model_name}"
        else:
            display_name = f"{tokenizer_name} tokenizer of {model_name}"

        op = _AssetDownloadOp(self._cache_dir, uri, display_name, force, progress)

        return op.run()

    @override
    def download_dataset(
        self,
        uri: str,
        dataset_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        display_name = f"{dataset_name} dataset"

        op = _AssetDownloadOp(self._cache_dir, uri, display_name, force, progress)

        return op.run()


class _AssetDownloadOp:
    _cache_dir: Path
    _uri: str
    _uri_params: dict[str, str]
    _asset_dir: Path | None
    _display_name: str
    _force: bool
    _progress: bool
    _shard_idx: int | None

    def __init__(
        self,
        cache_dir: Path,
        uri: str,
        display_name: str,
        force: bool,
        progress: bool,
        shard_idx: int | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._uri = uri
        self._uri_params = {}
        self._asset_dir = None
        self._display_name = display_name
        self._force = force
        self._progress = progress
        self._shard_idx = shard_idx

    def run(self) -> Path:
        self._process_uri()

        self._format_uri_with_shard_index()

        self._check_if_gated_asset()

        if (asset_path := self._try_uri_as_path()) is not None:
            if not asset_path.exists():
                raise AssetDownloadError(
                    f"The {self._display_name} cannot be found at {asset_path}."
                )

            return asset_path

        self._resolve_asset_dirname()

        self._prepare_op()

        self._download_asset()

        self._ensure_asset_extracted()

        return self._get_final_asset_path()

    def _process_uri(self) -> None:
        uri = self._uri

        try:
            if not _starts_with_scheme(uri):
                uri = Path(uri).as_uri()  # Normalize.

            parsed_uri = urlparse(uri)
        except ValueError:
            raise ValueError(
                f"`uri` must be a URI or an absolute pathname, but is '{uri}' instead."
            ) from None

        if parsed_uri.params:
            for param in parsed_uri.params.split(";"):
                key_value_pair = param.split("=")
                if len(key_value_pair) != 2:
                    raise ValueError(
                        f"`uri` must be a URI or an absolute pathname, but is '{uri}' instead."
                    )

                key, value = key_value_pair

                key = unquote(key).strip()
                if len(key) == 0:
                    raise ValueError(
                        f"`uri` must be a URI or an absolute pathname, but is '{uri}' instead."
                    )

                self._uri_params[key.lower()] = unquote(value).strip()

        self._uri = parsed_uri._replace(params="").geturl()

    def _format_uri_with_shard_index(self) -> None:
        if self._shard_idx is None:
            return

        sharded_uri = self._uri.replace("%7Bshard_idx%7D", str(self._shard_idx))
        if sharded_uri == self._uri:
            raise AssetDownloadError(
                f"`shard_idx` is specified, but the {self._display_name} is not sharded."
            )

        self._uri = sharded_uri

    def _check_if_gated_asset(self) -> None:
        if self._uri_params.get("gated", "false").strip().lower() == "true":
            raise AssetDownloadError(
                f"The {self._display_name} is gated. Please visit {self._uri} to learn how to get access."
            )

    def _try_uri_as_path(self) -> Path | None:
        if self._uri.startswith("file://"):
            return Path(unquote(self._uri[7:]))

        return None

    def _resolve_asset_dirname(self) -> None:
        h = sha1(self._uri.encode()).hexdigest()

        h = h[:24]

        self._asset_dir = self._cache_dir.joinpath(h)

    def _prepare_op(self) -> None:
        asset_dir = self._asset_dir

        assert asset_dir is not None

        if self._force:
            if asset_dir.exists():
                log.info("Ignoring the cached {}. `force` is set to `True`.", self._display_name)  # fmt: skip

                try:
                    rmtree(asset_dir)
                except OSError as ex:
                    raise AssetDownloadError(
                        f"The asset cache directory of the {self._display_name} cannot be deleted. See the nested exception for details."
                    ) from ex

            download_dir = asset_dir.with_suffix(".download")
            if download_dir.exists():
                try:
                    rmtree(download_dir)
                except OSError as ex:
                    raise AssetDownloadError(
                        f"The asset download directory of the {self._display_name} cannot be deleted. See the nested exception for details."
                    ) from ex

            download_dir = asset_dir.with_suffix(".download.tmp")
            if download_dir.exists():
                try:
                    rmtree(download_dir)
                except OSError as ex:
                    raise AssetDownloadError(
                        f"The asset download directory of the {self._display_name} cannot be deleted. See the nested exception for details."
                    ) from ex
        else:
            if asset_dir.exists():
                # Touch the asset directory so that we can maintain an LRU list
                # for cache cleanup.
                try:
                    asset_dir.touch()
                except OSError:
                    pass

                log.info("Using the cached {}. Set `force` to `True` to download again.", self._display_name)  # fmt: skip

    def _download_asset(self) -> None:
        assert self._asset_dir is not None

        download_dir = self._asset_dir.with_suffix(".download")

        # Check if we have already downloaded the asset in a previous call.
        if self._asset_dir.exists() or download_dir.exists():
            return

        succeeded = False

        with ExitStack() as cleanup_stack:
            tmp_dir = self._asset_dir.with_suffix(".download.tmp")

            try:
                tmp_dir.mkdir(parents=True, exist_ok=True)
            except OSError as ex:
                raise AssetDownloadError(
                    f"The asset download directory of the {self._display_name} cannot be created. See the nested exception for details."
                ) from ex

            def remove_tmp_dir() -> None:
                if not succeeded:
                    try:
                        rmtree(tmp_dir)
                    except OSError:
                        pass

            cleanup_stack.callback(remove_tmp_dir)

            log.info("Downloading the {}...", self._display_name)

            request = Request(
                self._uri,
                # Most hosting providers return 403 if the user-agent is not a
                # well-known string. Act like Firefox.
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux i686; rv:109.0) Gecko/20100101 Firefox/119.0"
                },
            )

            try:
                response = cleanup_stack.enter_context(urlopen(request))
            except URLError as ex:
                raise AssetDownloadError(
                    f"The download of the {self._display_name} has failed. See the nested exception for details."
                ) from ex
            except HTTPError as ex:
                raise AssetDownloadError(
                    f"The download of the {self._display_name} has failed with the HTTP error code {ex.code}."
                )

            headers = response.info()

            try:
                size = int(headers["Content-Length"])
            except TypeError:
                size = None

            fp = cleanup_stack.enter_context(
                NamedTemporaryFile(delete=False, dir=tmp_dir)
            )

            num_bytes_read = 0

            progress_bar = cleanup_stack.enter_context(
                tqdm(
                    total=size,
                    disable=not self._progress,
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
                        f"The download of the {self._display_name} has failed with the HTTP error code {ex.code}."
                    )

                buffer_len = len(buffer)
                if buffer_len == 0:
                    break

                if size is not None:
                    num_bytes_read += buffer_len
                    if num_bytes_read > size:
                        raise AssetDownloadError(
                            f"The download of the {self._display_name} has failed. The number of bytes sent by the server exceeded the expected size of {size:,} bytes."
                        )

                fp.write(buffer)

                progress_bar.update(buffer_len)

            if size is not None and num_bytes_read < size:
                raise AssetDownloadError(
                    f"The download of the {self._display_name} has failed. The server sent {num_bytes_read:,} bytes which is less than the expected size of {size:,} bytes."
                )

            fp.close()

            try:
                filename = Path(urlparse(response.geturl()).path).name

                filename = unquote(filename)
            except ValueError:
                filename = "asset"

            asset_file = tmp_dir.joinpath(filename)

            try:
                os.replace(fp.name, asset_file)
            except OSError:
                raise AssetDownloadError(
                    f"The {self._display_name} cannot be saved to the asset download directory. See the nested exception for details."
                )

            try:
                tmp_dir.replace(download_dir)
            except OSError:
                raise AssetDownloadError(
                    f"The asset download directory of the {self._display_name} cannot be renamed. See the nested exception for details."
                )

            succeeded = True

            log.info("Download complete.")

    def _ensure_asset_extracted(self) -> None:
        asset_dir = self._asset_dir

        assert asset_dir is not None

        download_dir = asset_dir.with_suffix(".download")

        # Check if we have already extracted the asset.
        if not download_dir.exists():
            return

        try:
            asset_dir.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            raise AssetDownloadError(
                f"The asset cache directory of the {self._display_name} cannot be created. See the nested exception for details."
            ) from ex

        def iter_dir() -> Iterator[Path]:
            try:
                for path in download_dir.iterdir():
                    yield path
            except OSError as ex:
                raise AssetDownloadError(
                    f"The asset download directory of the {self._display_name} cannot be traversed. See the nested exception for details."
                ) from ex

        for asset_path in iter_dir():
            # There are various file types (e.g. PyTorch tensor files) that
            # internally use the zip format. To be on the safe side we only
            # extract files that have the '.zip' suffix.
            if asset_path.suffix == ".zip":
                log.info("Extracting the {}...", self._display_name)

                try:
                    with ZipFile(asset_path) as zip_fp:
                        zip_fp.extractall(path=asset_dir)
                except (KeyError, OSError, BadZipFile) as ex:
                    raise AssetDownloadError(
                        f"The {self._display_name} cannot be extracted. See the nested exception for details."
                    ) from ex

                try:
                    asset_path.unlink()
                except OSError:
                    pass

                log.info("Extraction complete.")
            elif is_tarfile(asset_path):
                log.info("Extracting the {}...", self._display_name)

                try:
                    with TarFile(asset_path) as tar_fp:
                        tar_fp.extractall(path=asset_dir)
                except (KeyError, OSError) as ex:
                    raise AssetDownloadError(
                        f"The {self._display_name} cannot be extracted. See the nested exception for details."
                    ) from ex

                try:
                    asset_path.unlink()
                except OSError:
                    pass

                log.info("Extraction complete.")
            else:
                try:
                    asset_path.replace(asset_dir.joinpath(asset_path.name))
                except OSError as ex:
                    raise AssetDownloadError(
                        f"The {self._display_name} cannot be moved to the asset cache directory. See the nested exception for details."
                    ) from ex

        try:
            rmtree(download_dir)
        except OSError as ex:
            raise AssetDownloadError(
                f"The asset download directory of the {self._display_name} cannot be deleted. See the nested exception for details."
            ) from ex

    def _get_final_asset_path(self) -> Path:
        asset_dir = self._asset_dir

        assert asset_dir is not None and asset_dir.exists()

        asset_path = None

        asset_pathname = self._uri_params.get("path")
        if asset_pathname:
            asset_path = asset_dir.joinpath(asset_pathname).resolve()

            try:
                asset_path.relative_to(asset_dir)
            except ValueError as ex:
                raise AssetDownloadError(
                    f"The 'path' URI parameter of the {self._display_name} ({asset_pathname}) points to a path outside of the asset cache directory."
                ) from ex

            if not asset_path.exists():
                raise AssetDownloadError(
                    f"The {self._display_name} cannot be found. Please set `force` to `True` and, if the problem persists, file a bug report."
                )

            return asset_path

        # If we have a single file under the asset directory, return the path of
        # the file; otherwise, return the path of the directory.
        try:
            for path in asset_dir.iterdir():
                if asset_path is not None or not path.is_file():
                    asset_path = asset_dir

                    break

                asset_path = path
        except OSError as ex:
            raise AssetDownloadError(
                f"The asset cache directory of the {self._display_name} cannot be traversed. See the nested exception for details."
            ) from ex

        if asset_path is None:
            raise AssetDownloadError(
                f"The asset cache directory of the {self._display_name} is empty. Please set `force` to `True` and, if the problem persists, file a bug report."
            )

        return asset_path
