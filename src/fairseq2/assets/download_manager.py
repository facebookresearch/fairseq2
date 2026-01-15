# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence, Set
from contextlib import ExitStack, contextmanager
from hashlib import sha1
from pathlib import Path
from tarfile import TarFile, is_tarfile
from tempfile import NamedTemporaryFile
from typing import Final, final
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import HfHubHTTPError
from typing_extensions import NoReturn, override

from fairseq2.error import InternalError, NotSupportedError
from fairseq2.file_system import FileSystem, _flush_nfs_lookup_cache
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.uri import Uri
from fairseq2.utils.warn import _warn_deprecated


def get_asset_download_manager() -> AssetDownloadManager:
    return get_dependency_resolver().resolve(AssetDownloadManager)


class AssetDownloadManager(ABC):
    @abstractmethod
    def download_model(
        self,
        uri: Uri,
        model_name: str = "",
        *,
        force: bool = False,
        local_only: bool = False,
        progress: bool = True,
    ) -> Path:
        """
        Downloads the model checkpoint at the specified URI to the asset cache
        directory.

        Returns the path to the downloaded model checkpoint.

        If ``force`` is ``True``, the model checkpoint will be downloaded even
        if it is already in cache.

        If ``local_only`` is ``True``, the cached path of the model checkpoint
        will be returned. If not cached, an :class:`AssetDownloadError` will be
        raised.

        ``model_name`` is deprecated and will be removed in v0.13.

        ``progress`` is deprecated and will be removed in v0.13. Use
        ``FAIRSEQ2_NO_PROGRESS=1`` environment variable or ``no_progress``
        parameter of :func:`init_fairseq` to disable progress bars.

        :raises AssetDownloadError: The download operation failed due to a
            network or server error.
        """

    @abstractmethod
    def download_tokenizer(
        self,
        uri: Uri,
        tokenizer_name: str = "",
        *,
        force: bool = False,
        local_only: bool = False,
        progress: bool = True,
    ) -> Path:
        """
        Downloads the tokenizer at the specified URI to the asset cache
        directory.

        Returns the path to the downloaded tokenizer.

        If ``force`` is ``True``, the tokenizer will be downloaded even if it is
        already in cache.

        If ``local_only`` is ``True``, the cached path of the tokenizer will be
        returned. If not cached, an :class:`AssetDownloadError` will be raised.

        ``tokenizer_name`` is deprecated and will be removed in v0.13.

        ``progress`` is deprecated and will be removed in v0.13. Use
        ``FAIRSEQ2_NO_PROGRESS=1`` environment variable or ``no_progress``
        parameter of :func:`init_fairseq` to disable progress bars.

        :raises AssetDownloadError: The download operation failed due to a
            network or server error.
        """

    @abstractmethod
    def download_dataset(
        self,
        uri: Uri,
        dataset_name: str = "",
        *,
        force: bool = False,
        local_only: bool = False,
        progress: bool = True,
    ) -> Path:
        """
        Downloads the dataset at the specified URI to the asset cache directory.

        Returns the path to the downloaded dataset.

        If ``force`` is ``True``, the dataset will be downloaded even if it is
        already in cache.

        If ``local_only`` is ``True``, the cached path of the dataset will be
        returned. If not cached, an :class:`AssetDownloadError` will be raised.

        ``dataset_name`` is deprecated and will be removed in v0.13.

        ``progress`` is deprecated and will be removed in v0.13. Use
        ``FAIRSEQ2_NO_PROGRESS=1`` environment variable or ``no_progress``
        parameter of :func:`init_fairseq` to disable progress bars.

        :raises AssetDownloadError: The download operation failed due to a
            network or server error.
        """

    @property
    @abstractmethod
    def supported_schemes(self) -> Set[str]:
        """URI schemes supported by this download manager."""


class AssetDownloadError(Exception):
    def __init__(self, uri: Uri, reason: str) -> None:
        super().__init__(f"Download of {uri} failed. {reason}")

        self.uri = uri


@final
class DelegatingAssetDownloadManager(AssetDownloadManager):
    def __init__(self, managers: Sequence[AssetDownloadManager]) -> None:
        scheme_to_manager: dict[str, AssetDownloadManager] = {}

        schemes: set[str] = set()

        for manager in managers:
            for scheme in manager.supported_schemes:
                if scheme in schemes:
                    raise ValueError(
                        f"`managers` must support disjoint set of schemes, but {scheme} scheme is supported by more than one download manager."
                    )

                scheme_to_manager[scheme] = manager

            schemes.update(manager.supported_schemes)

        self._managers = managers
        self._scheme_to_manager = scheme_to_manager
        self._schemes = schemes

    @override
    def download_model(
        self,
        uri: Uri,
        model_name: str = "",
        *,
        force: bool = False,
        local_only: bool = False,
        progress: bool = True,
    ) -> Path:
        if model_name:
            _warn_deprecated(
                "`model_name` parameter is deprecated and will be removed in v0.13"
            )

        manager = self._get_download_manager(uri)

        return manager.download_model(uri, "", force=force, local_only=local_only)

    @override
    def download_tokenizer(
        self,
        uri: Uri,
        tokenizer_name: str = "",
        *,
        force: bool = False,
        local_only: bool = False,
        progress: bool = True,
    ) -> Path:
        if tokenizer_name:
            _warn_deprecated(
                "`tokenizer_name` parameter is deprecated and will be removed in v0.13"
            )

        manager = self._get_download_manager(uri)

        return manager.download_tokenizer(uri, "", force=force, local_only=local_only)

    @override
    def download_dataset(
        self,
        uri: Uri,
        dataset_name: str = "",
        *,
        force: bool = False,
        local_only: bool = False,
        progress: bool = True,
    ) -> Path:
        if dataset_name:
            _warn_deprecated(
                "`dataset_name` parameter is deprecated and will be removed in v0.13"
            )

        manager = self._get_download_manager(uri)

        return manager.download_dataset(uri, "", force=force, local_only=local_only)

    def _get_download_manager(self, uri: Uri) -> AssetDownloadManager:
        manager = self._scheme_to_manager.get(uri.scheme)
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
        model_name: str = "",
        *,
        force: bool = False,
        local_only: bool = False,
        progress: bool = True,
    ) -> Path:
        return self._to_path(uri)

    @override
    def download_tokenizer(
        self,
        uri: Uri,
        tokenizer_name: str = "",
        *,
        force: bool = False,
        local_only: bool = False,
        progress: bool = True,
    ) -> Path:
        return self._to_path(uri)

    @override
    def download_dataset(
        self,
        uri: Uri,
        dataset_name: str = "",
        *,
        force: bool = False,
        local_only: bool = False,
        progress: bool = True,
    ) -> Path:
        return self._to_path(uri)

    @staticmethod
    def _to_path(uri: Uri) -> Path:
        if uri.scheme != "file":
            raise NotSupportedError(
                f"`uri.scheme` must be a supported URI scheme, but is {uri.scheme} instead."
            )

        return uri.to_path()

    @property
    @override
    def supported_schemes(self) -> Set[str]:
        return self._SCHEMES


@final
class HuggingFaceHub(AssetDownloadManager):
    _SCHEMES: Final = {"hg"}

    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

    @override
    def download_model(
        self,
        uri: Uri,
        model_name: str = "",
        *,
        force: bool = False,
        local_only: bool = False,
        progress: bool = True,
    ) -> Path:
        with self._handle_download(uri) as repo_id:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns=["*.safetensors*", "*.pt"],
                force_download=force,
                local_files_only=local_only,
            )

        return Path(path)

    @override
    def download_tokenizer(
        self,
        uri: Uri,
        tokenizer_name: str = "",
        *,
        force: bool = False,
        local_only: bool = False,
        progress: bool = True,
    ) -> Path:
        with self._handle_download(uri) as repo_id:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns="tokenizer*.json",
                force_download=force,
                local_files_only=local_only,
            )

        return Path(path)

    @override
    def download_dataset(
        self,
        uri: Uri,
        dataset_name: str = "",
        *,
        force: bool = False,
        local_only: bool = False,
        progress: bool = True,
    ) -> Path:
        with self._handle_download(uri) as repo_id:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                force_download=force,
                local_files_only=local_only,
            )

        return Path(path)

    @contextmanager
    def _handle_download(self, uri: Uri) -> Generator[str, None, None]:
        repo_id = self._get_repo_id(uri)

        try:
            cache_dir = Path(HF_HUB_CACHE)
        except ValueError:
            raise InternalError(f"{HF_HUB_CACHE} is not a valid directory.")

        _flush_nfs_lookup_cache(cache_dir)

        try:
            yield repo_id
        except HfHubHTTPError as ex:
            self._raise_download_error(uri, ex)

        _flush_nfs_lookup_cache(cache_dir)

    @staticmethod
    def _raise_download_error(uri: Uri, cause: Exception) -> NoReturn:
        reason = "Hugging Face Hub returned an error."

        raise AssetDownloadError(uri, reason) from cause

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

    def __init__(
        self,
        cache_dir: Path,
        file_system: FileSystem,
        progress_reporter: ProgressReporter,
        download_progress_reporter: ProgressReporter,
    ) -> None:
        self._cache_dir = cache_dir
        self._file_system = file_system
        self._progress_reporter = progress_reporter
        self._download_progress_reporter = download_progress_reporter

    @override
    def download_model(
        self,
        uri: Uri,
        model_name: str = "",
        *,
        force: bool = False,
        local_only: bool = False,
        progress: bool = True,
    ) -> Path:
        return self._download_asset(uri, force, local_only)

    @override
    def download_tokenizer(
        self,
        uri: Uri,
        tokenizer_name: str = "",
        *,
        force: bool = False,
        local_only: bool = False,
        progress: bool = True,
    ) -> Path:
        return self._download_asset(uri, force, local_only)

    @override
    def download_dataset(
        self,
        uri: Uri,
        dataset_name: str = "",
        *,
        force: bool = False,
        local_only: bool = False,
        progress: bool = True,
    ) -> Path:
        return self._download_asset(uri, force, local_only)

    def _download_asset(self, uri: Uri, force: bool, local_only: bool) -> Path:
        if uri.scheme not in self._SCHEMES:
            raise NotSupportedError(
                f"`uri.scheme` must be a supported URI scheme, but is {uri.scheme} instead."
            )

        op = _AssetDownloadOperation(
            self._cache_dir,
            uri,
            force,
            local_only,
            self._file_system,
            self._progress_reporter,
            self._download_progress_reporter,
        )

        return op.run()

    @property
    @override
    def supported_schemes(self) -> Set[str]:
        return self._SCHEMES


@final
class _AssetDownloadOperation:
    def __init__(
        self,
        cache_dir: Path,
        uri: Uri,
        force: bool,
        local_only: bool,
        file_system: FileSystem,
        progress_reporter: ProgressReporter,
        download_progress_reporter: ProgressReporter,
    ) -> None:
        h = self._get_uri_hash(uri)

        asset_dir = cache_dir.joinpath(h)

        self._cache_dir = cache_dir
        self._uri = uri
        self._asset_dir = asset_dir
        self._force = force
        self._local_only = local_only
        self._file_system = file_system
        self._progress_reporter = progress_reporter
        self._download_progress_reporter = download_progress_reporter

    @staticmethod
    def _get_uri_hash(uri: Uri) -> str:
        h = sha1(str(uri).encode()).hexdigest()

        return h[:24]

    def run(self) -> Path:
        _flush_nfs_lookup_cache(self._asset_dir)

        try:
            if not self._local_only:
                if self._force:
                    self._clean_cached_asset()

                self._download_asset()

                self._ensure_asset_extracted()

            path = self._retrieve_asset_path()
        except OSError as ex:
            self._raise_download_error("A system error occurred.", ex)

        _flush_nfs_lookup_cache(self._asset_dir)

        return path

    def _clean_cached_asset(self) -> None:
        asset_dir = self._asset_dir

        try:
            self._file_system.remove_directory(asset_dir)
        except FileNotFoundError:
            pass

        download_dir = asset_dir.with_suffix(".download")

        try:
            self._file_system.remove_directory(download_dir)
        except FileNotFoundError:
            pass

        tmp_download_dir = asset_dir.with_suffix(".download.tmp")

        try:
            self._file_system.remove_directory(tmp_download_dir)
        except FileNotFoundError:
            pass

    def _download_asset(self) -> None:
        asset_dir = self._asset_dir

        if self._file_system.exists(asset_dir):
            return

        download_dir = asset_dir.with_suffix(".download")

        if self._file_system.exists(download_dir):
            return

        succeeded = False

        with ExitStack() as exit_stack:
            tmp_download_dir = asset_dir.with_suffix(".download.tmp")

            self._file_system.make_directory(tmp_download_dir)

            def remove_tmp_dir() -> None:
                if not succeeded:
                    try:
                        self._file_system.remove_directory(tmp_download_dir)
                    except OSError:
                        pass

            exit_stack.callback(remove_tmp_dir)

            request = Request(
                str(self._uri),
                # Most hosting providers return 403 if the user-agent is not a
                # well-known string. Act like Firefox.
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux i686; rv:109.0) Gecko/20100101 Firefox/119.0"
                },
            )

            try:
                response = urlopen(request)
            except URLError as ex:
                self._raise_download_error("Address is not reachable.", ex)
            except HTTPError as ex:
                self._raise_download_error(
                    f"Server returned HTTP error code {ex.code}."
                )

            exit_stack.enter_context(response)

            headers = response.info()

            total_size = None

            content_length = headers.get("Content-Length")
            if content_length is not None:
                try:
                    total_size = int(content_length)
                except ValueError:
                    pass

            fp = NamedTemporaryFile(delete=False, dir=tmp_download_dir)

            exit_stack.enter_context(fp)

            num_bytes_read = 0

            exit_stack.enter_context(self._download_progress_reporter)

            task = self._download_progress_reporter.create_task(
                "download", total=total_size
            )

            exit_stack.enter_context(task)

            while True:
                try:
                    buffer = response.read(1024 * 8)
                except HTTPError as ex:
                    self._raise_download_error(
                        f"Server returned HTTP error code {ex.code}."
                    )

                buffer_len = len(buffer)
                if buffer_len == 0:
                    break

                if total_size is not None:
                    num_bytes_read += buffer_len
                    if num_bytes_read > total_size:
                        self._raise_download_error(
                            f"Number of bytes sent by the server exceeds the expected size of {total_size:,} bytes."
                        )

                fp.write(buffer)

                task.step(buffer_len)

            if total_size is not None:
                if num_bytes_read < total_size:
                    self._raise_download_error(
                        f"Server sent {num_bytes_read:,} bytes which is less than the expected size of {total_size:,} bytes."
                    )

            fp.close()

            tmp_file = Path(fp.name)

            filename = None

            content_disposition = response.headers.get("Content-Disposition")
            if content_disposition is not None:
                match = re.search(r"filename=\"?([^\"\n]+)\"?", content_disposition)
                if match:
                    filename = match.group(1)

            if filename is None:
                try:
                    filename = unquote(Path(urlparse(response.geturl()).path).name)
                except ValueError:
                    filename = "asset"

            download_file = tmp_download_dir.joinpath(filename)

            self._file_system.move(tmp_file, download_file)

            self._file_system.move(tmp_download_dir, download_dir)

            succeeded = True

    def _ensure_asset_extracted(self) -> None:
        asset_dir = self._asset_dir

        download_dir = asset_dir.with_suffix(".download")

        if not self._file_system.exists(download_dir):
            return

        self._file_system.make_directory(asset_dir)

        it = download_dir.iterdir()

        try:
            download_file = next(it)
        except StopIteration:
            return

        # There are various file types (e.g. PyTorch tensor files) that
        # internally use the zip format. To be on the safe side we only
        # extract files that have the .zip suffix.
        if download_file.suffix == ".zip":
            with self._progress_reporter:
                task = self._progress_reporter.create_task(
                    "extract", total=None, start=False
                )

                with task:
                    try:
                        with ZipFile(download_file) as zip_fp:
                            zip_fp.extractall(path=asset_dir)
                    except (KeyError, OSError, EOFError, BadZipFile) as ex:
                        self._raise_download_error(
                            f"{download_file} cannot be extracted.", ex
                        )

                    try:
                        self._file_system.remove(download_file)
                    except OSError:
                        pass
        elif is_tarfile(download_file):
            with self._progress_reporter:
                task = self._progress_reporter.create_task(
                    "extract", total=None, start=False
                )

                with task:
                    try:
                        with TarFile(download_file) as tar_fp:
                            tar_fp.extractall(path=asset_dir)
                    except (KeyError, OSError, EOFError) as ex:
                        self._raise_download_error(
                            f"{download_file} cannot be extracted.", ex
                        )

                    try:
                        self._file_system.remove(download_file)
                    except OSError:
                        pass
        else:
            asset_path = asset_dir.joinpath(download_file.name)

            self._file_system.move(download_file, asset_path)

        self._file_system.remove_directory(download_dir)

    def _retrieve_asset_path(self) -> Path:
        asset_dir = self._asset_dir

        if not self._file_system.exists(asset_dir):
            if self._local_only:
                reason = "Asset is not cached, but `local_only` is `True`."
            else:
                reason = f"{asset_dir} asset directory is not found."

            self._raise_download_error(reason)

        it = asset_dir.iterdir()

        try:
            path = next(it)
        except StopIteration:
            self._raise_download_error(f"{asset_dir} asset directory is empty.")

        # If we have a single file under the asset directory, return the path of
        # the file; otherwise, return the path of the directory.
        try:
            next(it)
        except StopIteration:
            pass
        else:
            return asset_dir

        if self._file_system.is_file(path):
            return path

        return asset_dir

    def _raise_download_error(
        self, msg: str, cause: Exception | None = None
    ) -> NoReturn:
        raise AssetDownloadError(self._uri, msg) from cause
