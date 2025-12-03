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
from dataclasses import dataclass
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
from typing_extensions import override

from fairseq2.error import InternalError, NotSupportedError
from fairseq2.file_system import FileSystem, _flush_nfs_lookup_cache
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.uri import Uri


def get_asset_download_manager() -> AssetDownloadManager:
    return get_dependency_resolver().resolve(AssetDownloadManager)


@dataclass(kw_only=True)
class AssetDownloadOptions:
    force: bool = False
    """
    Indicates whether the model checkpoint should be downloaded even if it is
    already in cache.
    """

    local_only: bool = False
    """
    Indicates whether the cached path of the model checkpoint should be returned.
    If not cached, an :class:`AssetDownloadError` will be raised.
    """


class AssetDownloadManager(ABC):
    @abstractmethod
    def download_model(
        self, uri: Uri, options: AssetDownloadOptions | None = None
    ) -> Path:
        """
        Downloads the model checkpoint at the specified URI to the asset cache
        directory.

        Returns the path to the downloaded model checkpoint.

        :raises AssetNotFoundError: Asset at the specified URI is not found.

        :raises AssetDownloadNetworkError: Download operation failed due to a
            network or server error.

        :raises CorruptAssetError: Downloaded asset is corrupt.

        :raises CorruptAssetCacheError: Local asset cache directory is corrupt.

        :raises AssetDownloadError: Download operation failed due to an error.

        :raises NotSupportedError: Scheme of the specified URI is not supported.
        """

    @abstractmethod
    def download_tokenizer(
        self, uri: Uri, options: AssetDownloadOptions | None = None
    ) -> Path:
        """
        Downloads the tokenizer at the specified URI to the asset cache
        directory.

        Returns the path to the downloaded tokenizer.

        :raises AssetNotFoundError: Asset at the specified URI is not found.

        :raises AssetDownloadNetworkError: Download operation failed due to a
            network or server error.

        :raises CorruptAssetError: Downloaded asset is corrupt.

        :raises CorruptAssetCacheError: Local asset cache directory is corrupt.

        :raises AssetDownloadError: Download operation failed due to an error.

        :raises NotSupportedError: Scheme of the specified URI is not supported.
        """

    @abstractmethod
    def download_dataset(
        self, uri: Uri, options: AssetDownloadOptions | None = None
    ) -> Path:
        """
        Downloads the dataset at the specified URI to the asset cache directory.

        Returns the path to the downloaded dataset.

        :raises AssetNotFoundError: Asset at the specified URI is not found.

        :raises AssetDownloadNetworkError: Download operation failed due to a
            network or server error.

        :raises CorruptAssetError: Downloaded asset is corrupt.

        :raises CorruptAssetCacheError: Local asset cache directory is corrupt.

        :raises AssetDownloadError: Download operation failed due to an error.

        :raises NotSupportedError: Scheme of the specified URI is not supported.
        """

    @property
    @abstractmethod
    def supported_schemes(self) -> Set[str]:
        """URI schemes supported by this download manager."""


class AssetDownloadError(Exception):
    pass


class AssetNotFoundError(AssetDownloadError):
    pass


class AssetDownloadNetworkError(AssetDownloadError):
    pass


class CorruptAssetError(AssetDownloadError):
    pass


class CorruptAssetCacheError(AssetDownloadError):
    pass


@final
class _DelegatingAssetDownloadManager(AssetDownloadManager):
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
        self, uri: Uri, options: AssetDownloadOptions | None = None
    ) -> Path:
        manager = self._get_download_manager(uri)

        return manager.download_model(uri, options)

    @override
    def download_tokenizer(
        self, uri: Uri, options: AssetDownloadOptions | None = None
    ) -> Path:
        manager = self._get_download_manager(uri)

        return manager.download_tokenizer(uri, options)

    @override
    def download_dataset(
        self, uri: Uri, options: AssetDownloadOptions | None = None
    ) -> Path:
        manager = self._get_download_manager(uri)

        return manager.download_dataset(uri, options)

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
class _LocalAssetDownloadManager(AssetDownloadManager):
    _SCHEMES: Final = {"file"}

    @override
    def download_model(
        self, uri: Uri, options: AssetDownloadOptions | None = None
    ) -> Path:
        return self._to_path(uri)

    @override
    def download_tokenizer(
        self, uri: Uri, options: AssetDownloadOptions | None = None
    ) -> Path:
        return self._to_path(uri)

    @override
    def download_dataset(
        self, uri: Uri, options: AssetDownloadOptions | None = None
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
class _HuggingFaceHub(AssetDownloadManager):
    _SCHEMES: Final = {"hg"}

    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

    @override
    def download_model(
        self, uri: Uri, options: AssetDownloadOptions | None = None
    ) -> Path:
        if options is None:
            options = AssetDownloadOptions()

        with self._handle_download(uri) as repo_id:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns="*.safetensors",
                force_download=options.force,
                local_files_only=options.local_only,
            )

        return Path(path)

    @override
    def download_tokenizer(
        self, uri: Uri, options: AssetDownloadOptions | None = None
    ) -> Path:
        if options is None:
            options = AssetDownloadOptions()

        with self._handle_download(uri) as repo_id:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns="tokenizer*.json",
                force_download=options.force,
                local_files_only=options.local_only,
            )

        return Path(path)

    @override
    def download_dataset(
        self, uri: Uri, options: AssetDownloadOptions | None = None
    ) -> Path:
        if options is None:
            options = AssetDownloadOptions()

        with self._handle_download(uri) as repo_id:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                force_download=options.force,
                local_files_only=options.local_only,
            )

        return Path(path)

    @contextmanager
    def _handle_download(self, uri: Uri) -> Generator[str, None, None]:
        repo_id = self._get_repo_id(uri)

        try:
            cache_dir = Path(HF_HUB_CACHE)
        except ValueError:
            raise InternalError(f"{HF_HUB_CACHE} is not a valid pathname.") from None

        _flush_nfs_lookup_cache(cache_dir)

        try:
            yield repo_id
        except HfHubHTTPError as ex:
            if ex.response.status_code == 404:
                raise AssetNotFoundError(
                    "Asset not found on Hugging Face Hub."
                ) from None
            else:
                raise AssetDownloadNetworkError(
                    f"Hugging Face Hub returned HTTP error code {ex.response.status_code}."
                ) from ex

        _flush_nfs_lookup_cache(cache_dir)

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
class _StandardAssetDownloadManager(AssetDownloadManager):
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
        self, uri: Uri, options: AssetDownloadOptions | None = None
    ) -> Path:
        return self._download_asset(uri, options)

    @override
    def download_tokenizer(
        self, uri: Uri, options: AssetDownloadOptions | None = None
    ) -> Path:
        return self._download_asset(uri, options)

    @override
    def download_dataset(
        self, uri: Uri, options: AssetDownloadOptions | None = None
    ) -> Path:
        return self._download_asset(uri, options)

    def _download_asset(self, uri: Uri, options: AssetDownloadOptions | None) -> Path:
        if uri.scheme not in self._SCHEMES:
            raise NotSupportedError(
                f"`uri.scheme` must be a supported URI scheme, but is {uri.scheme} instead."
            )

        if options is None:
            options = AssetDownloadOptions()

        op = _AssetDownloadOperation(
            self._cache_dir,
            uri,
            options,
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
        options: AssetDownloadOptions,
        file_system: FileSystem,
        progress_reporter: ProgressReporter,
        download_progress_reporter: ProgressReporter,
    ) -> None:
        h = self._get_uri_hash(uri)

        asset_dir = cache_dir.joinpath(h)

        self._cache_dir = cache_dir
        self._uri = uri
        self._asset_dir = asset_dir
        self._options = options
        self._file_system = file_system
        self._progress_reporter = progress_reporter
        self._download_progress_reporter = download_progress_reporter

    @staticmethod
    def _get_uri_hash(uri: Uri) -> str:
        h = sha1(str(uri).encode()).hexdigest()

        return h[:24]

    def run(self) -> Path:
        _flush_nfs_lookup_cache(self._asset_dir)

        if not self._options.local_only:
            if self._options.force:
                self._clean_cached_asset()

            self._download_asset()

            self._ensure_asset_extracted()

        path = self._retrieve_asset_path()

        _flush_nfs_lookup_cache(self._asset_dir)

        return path

    def _clean_cached_asset(self) -> None:
        asset_dir = self._asset_dir

        self._delete_directory(asset_dir)

        download_dir = asset_dir.with_suffix(".download")

        self._delete_directory(download_dir)

        tmp_download_dir = asset_dir.with_suffix(".download.tmp")

        self._delete_directory(tmp_download_dir)

    def _download_asset(self) -> None:
        asset_dir = self._asset_dir

        if self._path_exists(asset_dir):
            return

        download_dir = asset_dir.with_suffix(".download")

        if self._path_exists(download_dir):
            return

        succeeded = False

        with ExitStack() as exit_stack:
            tmp_download_dir = asset_dir.with_suffix(".download.tmp")

            self._make_directory(tmp_download_dir)

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
                raise AssetDownloadError("Address is not reachable.") from ex
            except HTTPError as ex:
                if ex.code == 404:
                    raise AssetNotFoundError(
                        "Asset not found on remote machine."
                    ) from None
                else:
                    raise AssetDownloadNetworkError(
                        f"Server returned HTTP error code {ex.code}."
                    ) from None

            exit_stack.enter_context(response)

            headers = response.info()

            total_size = None

            content_length = headers.get("Content-Length")
            if content_length is not None:
                try:
                    total_size = int(content_length)
                except ValueError:
                    pass

            try:
                fp = NamedTemporaryFile(delete=False, dir=tmp_download_dir)
            except OSError as ex:
                raise AssetDownloadError(
                    f"Failed to create a temporary file under '{tmp_download_dir}' directory."
                ) from ex

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
                    raise AssetDownloadNetworkError(
                        f"Server returned HTTP error code {ex.code}."
                    ) from None

                buffer_len = len(buffer)
                if buffer_len == 0:
                    break

                if total_size is not None:
                    num_bytes_read += buffer_len
                    if num_bytes_read > total_size:
                        raise AssetDownloadNetworkError(
                            f"Number of bytes sent by the server exceeds the expected size of {total_size:,} bytes."
                        )

                try:
                    fp.write(buffer)
                except OSError as ex:
                    raise AssetDownloadError(
                        "Failed to write to the temporary download file."
                    ) from ex

                task.step(buffer_len)

            if total_size is not None:
                if num_bytes_read < total_size:
                    raise AssetDownloadNetworkError(
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

            self._rename_file(tmp_file, download_file)

            self._rename_file(tmp_download_dir, download_dir)

            succeeded = True

    def _ensure_asset_extracted(self) -> None:
        asset_dir = self._asset_dir

        download_dir = asset_dir.with_suffix(".download")

        if not self._path_exists(download_dir):
            return

        self._make_directory(asset_dir)

        try:
            it = download_dir.iterdir()

            try:
                download_file = next(it)
            except StopIteration:
                return
        except OSError as ex:
            raise AssetDownloadError(
                f"Failed to traverse '{asset_dir}' directory."
            ) from ex

        def is_tarfile_(file: Path) -> bool:
            try:
                return is_tarfile(file)
            except OSError as ex:
                raise AssetDownloadError(
                    f"Failed to determine whether '{file}' file is a TAR file."
                ) from ex

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
                    except (KeyError, EOFError, BadZipFile) as ex:
                        raise CorruptAssetError(
                            f"Failed to extract {download_file}."
                        ) from ex
                    except OSError as ex:
                        raise AssetDownloadError(
                            f"Failed to extract {download_file}."
                        ) from ex

                    try:
                        self._file_system.remove(download_file)
                    except OSError:
                        pass
        elif is_tarfile_(download_file):
            with self._progress_reporter:
                task = self._progress_reporter.create_task(
                    "extract", total=None, start=False
                )

                with task:
                    try:
                        with TarFile(download_file) as tar_fp:
                            tar_fp.extractall(path=asset_dir)
                    except (KeyError, EOFError) as ex:
                        raise CorruptAssetError(
                            f"Failed to extract {download_file}."
                        ) from ex
                    except OSError as ex:
                        raise AssetDownloadError(
                            f"Failed to extract {download_file}."
                        ) from ex

                    try:
                        self._file_system.remove(download_file)
                    except OSError:
                        pass
        else:
            asset_path = asset_dir.joinpath(download_file.name)

            self._rename_file(download_file, asset_path)

        self._delete_directory(download_dir)

    def _retrieve_asset_path(self) -> Path:
        asset_dir = self._asset_dir

        if not self._path_exists(asset_dir):
            if self._options.local_only:
                raise AssetDownloadError(
                    "Asset is not cached, but `local_only` is `True`."
                )
            else:
                raise CorruptAssetCacheError(
                    f"{asset_dir} asset directory is not found. Likely indicates that the cache is corrupt. Try with `options.force=True` or manually delete the directory."
                )

        try:
            it = asset_dir.iterdir()

            try:
                path = next(it)
            except StopIteration:
                raise CorruptAssetCacheError(
                    f"{asset_dir} asset directory is empty."
                ) from None

            # If we have a single file under the directory, return the path of
            # the file; otherwise, return the path of the directory.
            try:
                next(it)
            except StopIteration:
                pass
            else:
                return asset_dir
        except OSError as ex:
            raise AssetDownloadError(
                f"Failed to traverse '{asset_dir}' directory."
            ) from ex

        try:
            is_file = self._file_system.is_file(path)
        except OSError as ex:
            raise AssetDownloadError(f"Failed to access '{path}' path.") from ex

        if is_file:
            return path

        return asset_dir

    def _path_exists(self, path: Path) -> bool:
        try:
            return self._file_system.exists(path)
        except OSError as ex:
            raise AssetDownloadError(f"Failed to access '{path}' path.") from ex

    def _make_directory(self, path: Path) -> None:
        try:
            self._file_system.make_directory(path)
        except OSError as ex:
            raise AssetDownloadError(f"Failed to create '{path}' directory.") from ex

    def _delete_directory(self, path: Path) -> None:
        try:
            self._file_system.remove_directory(path)
        except FileNotFoundError:
            pass
        except OSError as ex:
            raise AssetDownloadError(f"Failed to delete '{path}' directory.") from ex

    def _rename_file(self, source: Path, target: Path) -> None:
        try:
            self._file_system.move(source, target)
        except OSError as ex:
            raise AssetDownloadError(
                f"Failed to rename '{source}' file to '{target}'."
            ) from ex
