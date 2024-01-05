# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from abc import ABC, abstractmethod
from contextlib import ExitStack
from hashlib import sha1
from pathlib import Path
from shutil import rmtree
from tarfile import TarFile, is_tarfile
from tempfile import NamedTemporaryFile
from typing import Dict, Iterator, Optional, final
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from tqdm import tqdm  # type: ignore[import]

from fairseq2.assets.error import AssetError
from fairseq2.assets.utils import _get_path_from_env, _starts_with_scheme
from fairseq2.typing import finaloverride


class AssetDownloadManager(ABC):
    """Downloads assets."""

    @abstractmethod
    def download_checkpoint(
        self,
        uri: str,
        model_name: str,
        *,
        shard_idx: Optional[int] = None,
        force: bool = False,
        cache_only: bool = False,
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
        :param cache_only:
            If ``True``, skips the download and returns the cached checkpoint.
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
        cache_only: bool = False,
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
        :param cache_only:
            If ``True``, skips the download and returns the cached tokenizer.
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
        cache_only: bool = False,
        progress: bool = True,
    ) -> Path:
        """Download the dataset at ``uri`` to the asset cache directory.

        :param uri:
            The URI to download from.
        :param data_name:
            The name of the dataset.
        :param force:
            If ``True``, downloads the dataset even if it is already in cache.
        :param cache_only:
            If ``True``, skips the download and returns the cached dataset.
        :param progress:
            If ``True``, displays a progress bar to stderr.

        :returns:
            The pathname of the downloaded dataset.
        """


@final
class InProcAssetDownloadManager(AssetDownloadManager):
    """Downloads assets in this process."""

    cache_dir: Path

    def __init__(self) -> None:
        cache_dir = _get_path_from_env("FAIRSEQ2_CACHE_DIR", missing_ok=True)
        if cache_dir is None:
            cache_dir = _get_path_from_env("XDG_CACHE_HOME")
            if cache_dir is None:
                cache_dir = Path("~/.cache").expanduser()

            cache_dir = cache_dir.joinpath("fairseq2/assets").resolve()

        self.cache_dir = cache_dir

    @finaloverride
    def download_checkpoint(
        self,
        uri: str,
        model_name: str,
        *,
        shard_idx: Optional[int] = None,
        force: bool = False,
        cache_only: bool = False,
        progress: bool = True,
    ) -> Path:
        display_name = f"checkpoint of {model_name}"

        if shard_idx is not None:
            display_name = f"{display_name} (shard {shard_idx})"

        op = _AssetDownloadOp(
            self.cache_dir, uri, display_name, force, cache_only, progress, shard_idx
        )

        return op.run()

    @finaloverride
    def download_tokenizer(
        self,
        uri: str,
        model_name: str,
        *,
        tokenizer_name: Optional[str] = None,
        force: bool = False,
        cache_only: bool = False,
        progress: bool = True,
    ) -> Path:
        if not tokenizer_name:
            display_name = f"tokenizer of {model_name}"
        else:
            display_name = f"{tokenizer_name} tokenizer of {model_name}"

        op = _AssetDownloadOp(
            self.cache_dir, uri, display_name, force, cache_only, progress
        )

        return op.run()

    @finaloverride
    def download_dataset(
        self,
        uri: str,
        dataset_name: str,
        *,
        force: bool = False,
        cache_only: bool = False,
        progress: bool = True,
    ) -> Path:
        display_name = f"{dataset_name} dataset"

        op = _AssetDownloadOp(
            self.cache_dir, uri, display_name, force, cache_only, progress
        )

        return op.run()


class _AssetDownloadOp:
    cache_dir: Path
    uri: str
    uri_params: Dict[str, str]
    asset_dir: Optional[Path]
    display_name: str
    force: bool
    cache_only: bool
    progress: bool
    shard_idx: Optional[int]

    def __init__(
        self,
        cache_dir: Path,
        uri: str,
        display_name: str,
        force: bool,
        cache_only: bool,
        progress: bool,
        shard_idx: Optional[int] = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.uri = uri
        self.uri_params = {}
        self.asset_dir = None
        self.display_name = display_name
        self.force = force
        self.cache_only = cache_only
        self.progress = progress
        self.shard_idx = shard_idx

    def run(self) -> Path:
        self._process_input_uri()

        self._format_uri_with_shard_index()

        self._check_if_gated_asset()

        if (asset_path := self._try_uri_as_path()) is not None:
            if not asset_path.exists():
                raise AssetError(
                    f"The {self.display_name} cannot be found at {asset_path}."
                )

            return asset_path

        self._resolve_asset_dirname()

        self._prepare_op()

        self._download_asset()

        self._ensure_asset_extracted()

        return self._get_final_asset_path()

    def _process_input_uri(self) -> None:
        uri = self.uri

        try:
            if uri.startswith("file://"):
                uri = uri[7:]

            if not _starts_with_scheme(uri):
                uri = Path(uri).as_uri()  # Normalize.

            parsed_uri = urlparse(uri)
        except ValueError as ex:
            raise ValueError(
                f"`uri` must be a URI or an absolute pathname, but is '{uri}' instead."
            ) from ex

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

                self.uri_params[key.lower()] = unquote(value).strip()

        self.uri = parsed_uri._replace(params="").geturl()

    def _format_uri_with_shard_index(self) -> None:
        if self.shard_idx is None:
            return

        uri_with_shard = self.uri.format(self.shard_idx)

        if uri_with_shard == self.uri:
            raise AssetError(
                f"`shard_idx` is specified, but the {self.display_name} is not sharded."
            )

        self.uri = uri_with_shard

    def _check_if_gated_asset(self) -> None:
        if self.uri_params.get("gated", "false").strip().lower() == "true":
            raise AssetError(
                f"The {self.display_name} is gated. Please visit {self.uri} to learn how to get access."
            )

    def _try_uri_as_path(self) -> Optional[Path]:
        if self.uri.startswith("file://"):
            return Path(self.uri[7:])

        return None

    def _resolve_asset_dirname(self) -> None:
        h = sha1(self.uri.encode()).hexdigest()

        h = h[:24]

        self.asset_dir = self.cache_dir.joinpath(h)

    def _prepare_op(self) -> None:
        asset_dir = self.asset_dir

        assert asset_dir is not None

        if self.force and not self.cache_only:
            if asset_dir.exists():
                if self.progress:
                    self._print_progress(
                        f"Ignoring the cached {self.display_name}. `force` is set to `True`."
                    )

                try:
                    rmtree(asset_dir)
                except OSError as ex:
                    raise AssetDownloadError(
                        f"The asset cache directory of the {self.display_name} cannot be deleted. See nested exception for details."
                    ) from ex

            download_dir = asset_dir.with_suffix(".download")
            if download_dir.exists():
                try:
                    rmtree(download_dir)
                except OSError as ex:
                    raise AssetDownloadError(
                        f"The asset download directory of the {self.display_name} cannot be deleted. See nested exception for details."
                    ) from ex

            download_dir = asset_dir.with_suffix(".download.tmp")
            if download_dir.exists():
                try:
                    rmtree(download_dir)
                except OSError as ex:
                    raise AssetDownloadError(
                        f"The asset download directory of the {self.display_name} cannot be deleted. See nested exception for details."
                    ) from ex
        else:
            if asset_dir.exists():
                # Touch the asset directory so that we can maintain an LRU list
                # for cache cleanup.
                try:
                    asset_dir.touch()
                except OSError:
                    pass

                if self.progress:
                    if self.cache_only:
                        self._print_progress(f"Using the cached {self.display_name}.")
                    else:
                        self._print_progress(
                            f"Using the cached {self.display_name}. Set `force` to `True` to download again."
                        )

    def _download_asset(self) -> None:
        if self.cache_only:
            return

        assert self.asset_dir is not None

        download_dir = self.asset_dir.with_suffix(".download")

        # Check if we have already downloaded the asset in a previous call.
        if self.asset_dir.exists() or download_dir.exists():
            return

        succeeded = False

        with ExitStack() as cleanup_stack:
            tmp_dir = self.asset_dir.with_suffix(".download.tmp")

            try:
                tmp_dir.mkdir(parents=True, exist_ok=True)
            except OSError as ex:
                raise AssetError(
                    f"The asset download directory of the {self.display_name} cannot be created. See nested exception for details."
                ) from ex

            def remove_tmp_dir() -> None:
                if not succeeded:
                    try:
                        rmtree(tmp_dir)
                    except OSError:
                        pass

            cleanup_stack.callback(remove_tmp_dir)

            if self.progress:
                self._print_progress(f"Downloading the {self.display_name}...")

            request = Request(
                self.uri,
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
                    f"The download of the {self.display_name} has failed. See nested exception for details."
                ) from ex
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
                NamedTemporaryFile(delete=False, dir=tmp_dir)
            )

            num_bytes_read = 0

            progress_bar = cleanup_stack.enter_context(
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

                buffer_len = len(buffer)
                if buffer_len == 0:
                    break

                if size is not None:
                    num_bytes_read += buffer_len
                    if num_bytes_read > size:
                        raise AssetDownloadError(
                            f"The download of the {self.display_name} has failed. The number of bytes sent by the server exceeded the expected size of {size:,} bytes."
                        )

                fp.write(buffer)

                progress_bar.update(buffer_len)

            if size is not None and num_bytes_read < size:
                raise AssetDownloadError(
                    f"The download of the {self.display_name} has failed. The server sent {num_bytes_read:,} bytes which is less than the expected size of {size:,} bytes."
                )

            fp.close()

            try:
                filename = Path(urlparse(response.geturl()).path).name
            except ValueError:
                filename = "asset"

            asset_file = tmp_dir.joinpath(filename)

            try:
                os.replace(fp.name, asset_file)
            except OSError:
                raise AssetError(
                    f"The {self.display_name} cannot be saved to the asset download directory. See nested exception for details."
                )

            try:
                tmp_dir.replace(download_dir)
            except OSError:
                raise AssetError(
                    f"The asset download directory of the {self.display_name} cannot be renamed. See nested exception for details."
                )

            succeeded = True

    def _ensure_asset_extracted(self) -> None:
        if self.cache_only:
            return

        asset_dir = self.asset_dir

        assert asset_dir is not None

        download_dir = asset_dir.with_suffix(".download")

        # Check if we have already extracted the asset.
        if not download_dir.exists():
            return

        try:
            asset_dir.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            raise AssetError(
                f"The asset cache directory of the {self.display_name} cannot be created. See nested exception for details."
            ) from ex

        def iter_dir() -> Iterator[Path]:
            try:
                for path in download_dir.iterdir():
                    yield path
            except OSError as ex:
                raise AssetError(
                    f"The asset download directory of the {self.display_name} cannot be traversed. See nested exception for details."
                ) from ex

        for asset_path in iter_dir():
            # There are various file types (e.g. PyTorch tensor files) that
            # internally use the zip format. To be on the safe side we only
            # extract files that have the '.zip' suffix.
            if asset_path.suffix == ".zip":
                if self.progress:
                    self._print_progress(f"Extracting the {self.display_name}...")

                try:
                    with ZipFile(asset_path) as zip_fp:
                        zip_fp.extractall(path=asset_dir)
                except (KeyError, OSError, BadZipFile) as ex:
                    raise AssetError(
                        f"The {self.display_name} cannot be extracted. See nested exception for details."
                    ) from ex

                try:
                    asset_path.unlink()
                except OSError:
                    pass
            elif is_tarfile(asset_path):
                if self.progress:
                    self._print_progress(f"Extracting the {self.display_name}...")

                try:
                    with TarFile(asset_path) as tar_fp:
                        tar_fp.extractall(path=asset_dir)
                except (KeyError, OSError) as ex:
                    raise AssetError(
                        f"The {self.display_name} cannot be extracted. See nested exception for details."
                    ) from ex

                try:
                    asset_path.unlink()
                except OSError:
                    pass
            else:
                try:
                    asset_path.replace(asset_dir.joinpath(asset_path.name))
                except OSError as ex:
                    raise AssetError(
                        f"The {self.display_name} cannot be moved to the asset cache directory. See nested exception for details."
                    ) from ex

        try:
            rmtree(download_dir)
        except OSError as ex:
            raise AssetError(
                f"The asset download directory of the {self.display_name} cannot be deleted. See nested exception for details."
            ) from ex

    def _get_final_asset_path(self) -> Path:
        asset_dir = self.asset_dir

        assert asset_dir is not None

        if self.cache_only:
            download_dir = asset_dir.with_suffix(".download")

            # If the asset directory is not found, or if the download/extraction
            # is not finished yet, raise an error.
            if not asset_dir.exists() or download_dir.exists():
                raise AssetError(
                    f"The {self.display_name} cannot be found. Set `cache_only` to `False` to download it."
                )
        else:
            assert asset_dir.exists()

        asset_path = None

        asset_pathname = self.uri_params.get("path")
        if asset_pathname:
            asset_path = asset_dir.joinpath(asset_pathname).resolve()

            try:
                asset_path.relative_to(asset_dir)
            except ValueError as ex:
                raise AssetError(
                    f"The 'path' URI parameter of the {self.display_name} ({asset_pathname}) points to a path outside of the asset cache directory."
                ) from ex

            if not asset_path.exists():
                raise AssetError(
                    f"The {self.display_name} cannot be found. Please set `force` to `True` and, if the problem persists, file a bug report."
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
            raise AssetError(
                f"The asset cache directory of the {self.display_name} cannot be traversed. See nested exception for details."
            ) from ex

        if asset_path is None:
            raise AssetError(
                f"The asset cache directory of the {self.display_name} is empty. Please set `force` to `True` and, if the problem persists, file a bug report."
            )

        return asset_path

    @staticmethod
    def _print_progress(s: str) -> None:
        print(s, file=sys.stderr)


class AssetDownloadError(AssetError):
    """Raised when an asset download operation fails."""


download_manager = InProcAssetDownloadManager()
