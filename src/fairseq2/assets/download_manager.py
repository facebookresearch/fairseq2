# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Set
from contextlib import ExitStack
from hashlib import sha1
from pathlib import Path
from shutil import rmtree
from tarfile import TarFile, is_tarfile
from tempfile import NamedTemporaryFile
from typing import Final, final
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from typing_extensions import override

from fairseq2.error import NotSupportedError
from fairseq2.logging import log
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.uri import Uri


def get_asset_download_manager() -> AssetDownloadManager:
    return get_dependency_resolver().resolve(AssetDownloadManager)


class AssetDownloadManager(ABC):
    @abstractmethod
    def download_model(
        self,
        uri: Uri,
        model_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        """
        Downloads the model checkpoint at ``uri`` to the asset cache directory.

        Returns the path to the downloaded model checkpoint.

        If ``force`` is ``True``, the model checkpoint will be downloaded even
        if it is already in cache.

        ``progress`` is deprecated and will be removed in v0.13. Use
        ``FAIRSEQ2_NO_PROGRESS=1`` environment variable or ``no_progress``
        parameter of :func:`init_fairseq` to disable progress bars.

        :raises AssetDownloadError: If the download operation fails due to a
            network or server error.
        """

    @abstractmethod
    def download_tokenizer(
        self,
        uri: Uri,
        tokenizer_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        """
        Downloads the tokenizer at ``uri`` to the asset cache directory.

        Returns the path to the downloaded tokenizer.

        If ``force`` is ``True``, the tokenizer will be downloaded even if it is
        already in cache.

        ``progress`` is deprecated and will be removed in v0.13. Use
        ``FAIRSEQ2_NO_PROGRESS=1`` environment variable or ``no_progress``
        parameter of :func:`init_fairseq` to disable progress bars.

        :raises AssetDownloadError: If the download operation fails due to a
            network or server error.
        """

    @abstractmethod
    def download_dataset(
        self,
        uri: Uri,
        dataset_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        """
        Downloads the dataset at ``uri`` to the asset cache directory.

        Returns the path to the downloaded dataset.

        If ``force`` is ``True``, the dataset will be downloaded even if it is
        already in cache.

        ``progress`` is deprecated and will be removed in v0.13. Use
        ``FAIRSEQ2_NO_PROGRESS=1`` environment variable or ``no_progress``
        parameter of :func:`init_fairseq` to disable progress bars.

        :raises AssetDownloadError: If the download operation fails due to a
            network or server error.
        """

    @property
    @abstractmethod
    def supported_schemes(self) -> Set[str]: ...


class AssetDownloadError(Exception):
    def __init__(self, asset_name: str, asset_kind: str, message: str) -> None:
        super().__init__(message)

        self.asset_name = asset_name
        self.asset_kind = asset_kind


@final
class DelegatingAssetDownloadManager(AssetDownloadManager):
    def __init__(self, managers: Iterable[AssetDownloadManager]) -> None:
        self._managers = {}
        self._schemes: set[str] = set()

        for manager in managers:
            for scheme in manager.supported_schemes:
                if scheme in self._schemes:
                    raise ValueError(
                        f"`managers` must support disjoint set of schemes, but {scheme} scheme is supported by more than one download manager."
                    )

                self._managers[scheme] = manager

            self._schemes.update(manager.supported_schemes)

    @override
    def download_model(
        self,
        uri: Uri,
        model_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        manager = self._get_download_manager(uri)

        return manager.download_model(uri, model_name, force=force, progress=progress)

    @override
    def download_tokenizer(
        self,
        uri: Uri,
        tokenizer_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        manager = self._get_download_manager(uri)

        return manager.download_tokenizer(
            uri, tokenizer_name, force=force, progress=progress
        )

    @override
    def download_dataset(
        self,
        uri: Uri,
        dataset_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        manager = self._get_download_manager(uri)

        return manager.download_dataset(
            uri, dataset_name, force=force, progress=progress
        )

    def _get_download_manager(self, uri: Uri) -> AssetDownloadManager:
        manager = self._managers.get(uri.scheme)
        if manager is None:
            raise NotSupportedError(
                f"`uri.scheme` must be a supported URI scheme, but is {uri.scheme} instead."
            )

        return manager

    @property
    @override
    def supported_schemes(self) -> Set[str]:
        return self._schemes


@final
class LocalAssetDownloadManager(AssetDownloadManager):
    _SCHEMES: Final = {"file"}

    @override
    def download_model(
        self,
        uri: Uri,
        model_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        return self._to_path(uri)

    @override
    def download_tokenizer(
        self,
        uri: Uri,
        tokenizer_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        return self._to_path(uri)

    @override
    def download_dataset(
        self,
        uri: Uri,
        dataset_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        return self._to_path(uri)

    @staticmethod
    def _to_path(uri: Uri) -> Path:
        if uri.scheme != "file":
            raise NotSupportedError(
                f"`uri.scheme` must be a supported URI scheme, but is {uri.scheme} instead."
            )

        s = str(uri)

        try:
            return Path(s[5:])
        except ValueError:
            raise ValueError(
                f"`uri` must represent a pathname, but is '{uri}' instead."
            ) from None

    @property
    @override
    def supported_schemes(self) -> Set[str]:
        return self._SCHEMES


@final
class HuggingFaceHub(AssetDownloadManager):
    _SCHEMES: Final = {"hg"}

    @override
    def download_model(
        self,
        uri: Uri,
        model_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        repo_id = self._get_repo_id(uri)

        try:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns="*.safetensors",
                force_download=force,
            )
        except HfHubHTTPError as ex:
            msg = f"{model_name} model cannot be downloaded from Hugging Face Hub."

            raise AssetDownloadError(model_name, "model", msg) from ex

        return Path(path)

    @override
    def download_tokenizer(
        self,
        uri: Uri,
        tokenizer_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        repo_id = self._get_repo_id(uri)

        try:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns="tokenizer*.json",
                force_download=force,
            )
        except HfHubHTTPError as ex:
            msg = f"{tokenizer_name} tokenizer cannot be downloaded from Hugging Face Hub."

            raise AssetDownloadError(tokenizer_name, "tokenizer", msg) from ex

        return Path(path)

    @override
    def download_dataset(
        self,
        uri: Uri,
        dataset_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        repo_id = self._get_repo_id(uri)

        try:
            path = snapshot_download(
                repo_id=repo_id, repo_type="dataset", force_download=force
            )
        except HfHubHTTPError as ex:
            msg = f"{dataset_name} dataset cannot be downloaded from Hugging Face Hub."

            raise AssetDownloadError(dataset_name, "dataset", msg) from ex

        return Path(path)

    @staticmethod
    def _get_repo_id(uri: Uri) -> str:
        if uri.scheme != "hg":
            raise NotSupportedError(
                f"`uri.scheme` must be a supported URI scheme, but is {uri.scheme} instead."
            )

        s = str(uri)

        return s[5:]

    @property
    @override
    def supported_schemes(self) -> Set[str]:
        return self._SCHEMES


@final
class StandardAssetDownloadManager(AssetDownloadManager):
    _SCHEMES: Final = {"http", "https"}

    def __init__(self, cache_dir: Path, progress_reporter: ProgressReporter) -> None:
        self._cache_dir = cache_dir
        self._progress_reporter = progress_reporter

    @override
    def download_model(
        self,
        uri: Uri,
        model_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        kind = "model"

        op = _AssetDownloadOp(
            self._cache_dir, uri, model_name, kind, force, self._progress_reporter
        )

        return op.run()

    @override
    def download_tokenizer(
        self,
        uri: Uri,
        tokenizer_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        kind = "tokenizer"

        op = _AssetDownloadOp(
            self._cache_dir, uri, tokenizer_name, kind, force, self._progress_reporter
        )

        return op.run()

    @override
    def download_dataset(
        self,
        uri: Uri,
        dataset_name: str,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        kind = "dataset"

        op = _AssetDownloadOp(
            self._cache_dir, uri, dataset_name, kind, force, self._progress_reporter
        )

        return op.run()

    @property
    @override
    def supported_schemes(self) -> Set[str]:
        return self._SCHEMES


class _AssetDownloadOp:
    def __init__(
        self,
        cache_dir: Path,
        uri: Uri,
        asset_name: str,
        kind: str,
        force: bool,
        progress_reporter: ProgressReporter,
    ) -> None:
        self._cache_dir = cache_dir
        self._uri = str(uri)
        self._uri_params: dict[str, str] = {}
        self._asset_dir: Path | None = None
        self._asset_name = asset_name
        self._asset_kind = kind
        self._force = force
        self._progress_reporter = progress_reporter

    def run(self) -> Path:
        self._process_uri()

        self._check_if_gated_asset()

        if (asset_path := self._try_uri_as_path()) is not None:
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

    def _check_if_gated_asset(self) -> None:
        if self._uri_params.get("gated", "false").strip().lower() == "true":
            msg = f"{self._asset_name} is gated. Please visit {self._uri} to learn how to get access."

            raise AssetDownloadError(self._asset_name, self._asset_kind, msg)

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
                log.info("Ignoring the cached {}. `force` is set to `True`.", self._asset_name)  # fmt: skip

                rmtree(asset_dir)

            download_dir = asset_dir.with_suffix(".download")
            if download_dir.exists():
                rmtree(download_dir)

            download_dir = asset_dir.with_suffix(".download.tmp")
            if download_dir.exists():
                rmtree(download_dir)
        else:
            if asset_dir.exists():
                # Touch the asset directory so that we can maintain an LRU list
                # for cache cleanup.
                try:
                    asset_dir.touch()
                except OSError:
                    pass

                log.info("Using the cached {}. Set `force` to `True` to download again.", self._asset_name)  # fmt: skip

    def _download_asset(self) -> None:
        assert self._asset_dir is not None

        download_dir = self._asset_dir.with_suffix(".download")

        # Check if we have already downloaded the asset in a previous call.
        if self._asset_dir.exists() or download_dir.exists():
            return

        succeeded = False

        with ExitStack() as cleanup_stack:
            tmp_dir = self._asset_dir.with_suffix(".download.tmp")

            tmp_dir.mkdir(parents=True, exist_ok=True)

            def remove_tmp_dir() -> None:
                if not succeeded:
                    try:
                        rmtree(tmp_dir)
                    except OSError:
                        pass

            cleanup_stack.callback(remove_tmp_dir)

            log.info("Downloading {}...", self._asset_name)

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
                msg = "Download failed."

                raise AssetDownloadError(
                    self._asset_name, self._asset_kind, msg
                ) from ex
            except HTTPError as ex:
                msg = f"Download failed with HTTP error code {ex.code}."

                raise AssetDownloadError(
                    self._asset_name, self._asset_kind, msg
                ) from None

            headers = response.info()

            try:
                size = int(headers["Content-Length"])
            except TypeError:
                size = None

            fp = cleanup_stack.enter_context(
                NamedTemporaryFile(delete=False, dir=tmp_dir)
            )

            num_bytes_read = 0

            cleanup_stack.enter_context(self._progress_reporter)

            task = self._progress_reporter.create_task("download", total=size)

            cleanup_stack.enter_context(task)

            while True:
                try:
                    buffer = response.read(1024 * 8)
                except HTTPError as ex:
                    msg = f"Download failed with HTTP error code {ex.code}."

                    raise AssetDownloadError(
                        self._asset_name, self._asset_kind, msg
                    ) from None

                buffer_len = len(buffer)
                if buffer_len == 0:
                    break

                if size is not None:
                    num_bytes_read += buffer_len
                    if num_bytes_read > size:
                        msg = f"The number of bytes sent by the server exceeded the expected size of {size:,} bytes."

                        raise AssetDownloadError(
                            self._asset_name, self._asset_kind, msg
                        )

                fp.write(buffer)

                task.step(buffer_len)

            if size is not None and num_bytes_read < size:
                msg = f"The server sent {num_bytes_read:,} bytes which is less than the expected size of {size:,} bytes."

                raise AssetDownloadError(self._asset_name, self._asset_kind, msg)

            fp.close()

            filename = None

            content_disposition = response.headers.get("Content-Disposition")
            if content_disposition is not None:
                match = re.search(r"filename=\"?([^\"\n]+)\"?", content_disposition)
                if match:
                    filename = match.group(1)

            if filename is None:
                try:
                    filename = Path(urlparse(response.geturl()).path).name

                    filename = unquote(filename)
                except ValueError:
                    filename = "asset"

            asset_file = tmp_dir.joinpath(filename)

            os.replace(fp.name, asset_file)

            tmp_dir.replace(download_dir)

            succeeded = True

            log.info("Download complete.")

    def _ensure_asset_extracted(self) -> None:
        asset_dir = self._asset_dir

        assert asset_dir is not None

        download_dir = asset_dir.with_suffix(".download")

        # Check if we have already extracted the asset.
        if not download_dir.exists():
            return

        asset_dir.mkdir(parents=True, exist_ok=True)

        def iter_dir() -> Iterator[Path]:
            for path in download_dir.iterdir():
                yield path

        for asset_path in iter_dir():
            # There are various file types (e.g. PyTorch tensor files) that
            # internally use the zip format. To be on the safe side we only
            # extract files that have the '.zip' suffix.
            if asset_path.suffix == ".zip":
                log.info("Extracting {}...", self._asset_name)

                try:
                    with ZipFile(asset_path) as zip_fp:
                        zip_fp.extractall(path=asset_dir)
                except (KeyError, OSError, BadZipFile) as ex:
                    msg = f"{asset_path} cannot be extracted."

                    raise AssetDownloadError(
                        self._asset_name, self._asset_kind, msg
                    ) from ex

                try:
                    asset_path.unlink()
                except OSError:
                    pass

                log.info("Extraction complete.")
            elif is_tarfile(asset_path):
                log.info("Extracting {}...", self._asset_name)

                try:
                    with TarFile(asset_path) as tar_fp:
                        tar_fp.extractall(path=asset_dir)
                except (KeyError, OSError) as ex:
                    msg = f"{asset_path} cannot be extracted."

                    raise AssetDownloadError(
                        self._asset_name, self._asset_kind, msg
                    ) from ex

                try:
                    asset_path.unlink()
                except OSError:
                    pass

                log.info("Extraction complete.")
            else:
                asset_path.replace(asset_dir.joinpath(asset_path.name))

        rmtree(download_dir)

    def _get_final_asset_path(self) -> Path:
        asset_dir = self._asset_dir

        assert asset_dir is not None and asset_dir.exists()

        asset_path = None

        asset_pathname = self._uri_params.get("path")
        if asset_pathname:
            asset_path = asset_dir.joinpath(asset_pathname).resolve()

            try:
                asset_path.relative_to(asset_dir)
            except ValueError:
                msg = "path parameter of the URI points to a path outside of the asset cache directory."

                raise AssetDownloadError(
                    self._asset_name, self._asset_kind, msg
                ) from None

            return asset_path

        # If we have a single file under the asset directory, return the path of
        # the file; otherwise, return the path of the directory.
        for path in asset_dir.iterdir():
            if asset_path is not None or not path.is_file():
                asset_path = asset_dir

                break

            asset_path = path

        assert asset_path is not None

        return asset_path


_SCHEME_REGEX: Final = re.compile("^[a-zA-Z0-9]+://")


def _starts_with_scheme(s: str) -> bool:
    return re.match(_SCHEME_REGEX, s) is not None
