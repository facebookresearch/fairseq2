# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from enum import Enum
from errno import ENOENT
from os import scandir, strerror
from pathlib import Path
from shutil import copytree, rmtree

from tempfile import TemporaryDirectory
from typing import Any, BinaryIO, Dict, List, TextIO, Tuple, TypeAlias, cast, final

import fsspec  # type: ignore[import-untyped]
import fsspec.implementations.local as fs_local  # type: ignore[import-untyped]
from fsspec.registry import (  # type: ignore[import-untyped]
    available_protocols,
    filesystem,
)
from typing_extensions import override

from fairseq2.typing import ContextManager
from fairseq2.context import RuntimeContext
from fairseq2.logging import log


class FileMode(Enum):
    READ = 0
    WRITE = 1
    APPEND = 2


class FileSystem(ABC):
    @abstractmethod
    def is_file(self, path: Path) -> bool: ...

    @abstractmethod
    def is_dir(self, path: Path) -> bool: ...

    @abstractmethod
    def open(self, path: Path, mode: FileMode = FileMode.READ) -> BinaryIO: ...

    @abstractmethod
    def cat(self, path: Path) -> bytes:
        """Read the contents of a file and return as bytes."""
        ...

    @abstractmethod
    def open_text(self, path: Path, mode: FileMode = FileMode.READ) -> TextIO: ...

    @abstractmethod
    def exists(self, path: Path) -> bool: ...

    @abstractmethod
    def move(self, old_path: Path, new_path: Path) -> None: ...

    @abstractmethod
    def remove(self, path: Path) -> None: ...

    @abstractmethod
    def make_directory(self, path: Path) -> None: ...

    @abstractmethod
    def copy_directory(self, source_path: Path, target_path: Path) -> None: ...

    @abstractmethod
    def remove_directory(self, path: Path) -> None: ...

    @abstractmethod
    def glob(self, path: Path, pattern: str) -> Iterator[Path]: ...

    @abstractmethod
    def walk_directory(
        self, path: Path, *, on_error: Callable[[OSError], None] | None
    ) -> Iterator[tuple[str, Sequence[str]]]: ...

    @abstractmethod
    def tmp_directory(self, base_dir: Path) -> ContextManager[Path]: ...

    @abstractmethod
    def resolve(self, path: Path) -> Path: ...

    @property
    @abstractmethod
    def is_local(self) -> bool:
        """Returns True if this filesystem only handles local paths.

        For filesystems that can handle both local and remote paths (like
        GlobalFileSystem), this returns False. Use is_local_path(path) to
        check if a specific path is local.
        """
        ...

    def is_local_path(self, path: Path) -> bool:  # noqa: ARG002
        """Check if a specific path is on a local filesystem.

        This method can be overridden by filesystems that handle both local
        and remote paths. The default implementation returns self.is_local.
        """
        return self.is_local


PathOrPaths: TypeAlias = str | Path | Sequence[str | Path]


@final
class FSspecFileSystem(FileSystem):
    """
    Wrapper around fsspec to provide a FileSystem interface.
    >>> from s3fs import S3FileSystem
    >>> fs = S3FileSystem()
    >>> fs = FSspecFileSystem(fs)
    """

    _fsspec: fsspec.AbstractFileSystem
    _prefix: str

    def __init__(self, fs: fsspec.AbstractFileSystem, prefix: str) -> None:
        self._fsspec = fs
        self._prefix = prefix

    def _normalize_uri_path(self, path_str: str) -> str:
        """
        Handle the case where pathlib.Path normalizes scheme:// to scheme:/
        by reconstructing the double slash for known URI schemes.
        """
        if self._prefix and "://" in self._prefix:
            scheme = self._prefix.split("://")[0]
            # Check if path looks like a URI with single slash (normalized by pathlib)
            if path_str.startswith(f"{scheme}:/") and not path_str.startswith(
                f"{scheme}://"
            ):
                # Restore the double slash
                return f"{scheme}://{path_str[len(scheme) + 2:]}"
        return path_str

    def get_short_uri(self, path: PathOrPaths) -> str | list[str]:
        """
        Removes prefix from paths and ensures they exist.

        Args:
            path: A path or sequence of paths to preprocess

        Returns:
            The path(s) with prefix removed
        """
        if isinstance(path, Sequence) and not isinstance(path, (str, Path)):
            return [self.get_short_uri(p) for p in path]  # type: ignore

        assert isinstance(path, (str, Path))
        path_str = self._normalize_uri_path(str(path))

        if not self._prefix:
            return path_str
        if path_str.startswith(self._prefix):
            return path_str[len(self._prefix) :].lstrip("/")
        else:
            raise ValueError(f"Path {path} does not start with prefix {self._prefix}")

    def get_long_uri(self, path: PathOrPaths) -> str | list[str]:
        """
        Adds prefix to paths to create full URIs.

        Args:
            path: A path or sequence of paths to preprocess

        Returns:
            The path(s) with prefix added
        """
        if isinstance(path, Sequence) and not isinstance(path, (str, Path)):
            return [self.get_long_uri(p) for p in path]  # type: ignore

        assert isinstance(path, (str, Path))
        path_str = self._normalize_uri_path(str(path))

        if not self._prefix:
            return path_str
        if not path_str.startswith(self._prefix):
            return self._prefix + path_str
        else:
            raise ValueError(f"Path {path} already starts with prefix {self._prefix}")

    @override
    def is_file(self, path: Path) -> bool:
        return bool(self._fsspec.isfile(self.get_short_uri(path)))

    @override
    def is_dir(self, path: Path) -> bool:
        return bool(self._fsspec.isdir(self.get_short_uri(path)))

    @override
    def exists(self, path: Path) -> bool:
        return bool(self._fsspec.exists(self.get_short_uri(path)))

    @override
    def open(self, path: Path, mode: FileMode = FileMode.READ) -> BinaryIO:
        mode_str = "rb"
        if mode == FileMode.WRITE:
            mode_str = "wb"
        elif mode == FileMode.APPEND:
            mode_str = "ab"

        return cast(BinaryIO, self._fsspec.open(self.get_short_uri(path), mode_str))

    @override
    def open_text(self, path: Path, mode: FileMode = FileMode.READ) -> TextIO:
        mode_str = "r"
        if mode == FileMode.WRITE:
            mode_str = "w"
        elif mode == FileMode.APPEND:
            mode_str = "a"

        return cast(
            TextIO,
            self._fsspec.open(self.get_short_uri(path), mode_str, encoding="utf-8"),
        )

    @override
    def cat(self, path: Path) -> bytes:
        return cast(bytes, self._fsspec.cat(self.get_short_uri(path)))

    @override
    def move(self, old_path: Path, new_path: Path) -> None:
        self._fsspec.mv(
            self.get_short_uri(old_path), self.get_short_uri(new_path), recursive=True
        )

    @override
    def remove(self, path: Path) -> None:
        # only one file !
        self._fsspec.rm(self.get_short_uri(path), recursive=False)

    @override
    def make_directory(self, path: Path) -> None:
        self._fsspec.makedirs(self.get_short_uri(path), exist_ok=True)

    @override
    def copy_directory(self, source_path: Path, target_path: Path) -> None:
        self._fsspec.copy(
            self.get_short_uri(source_path),
            self.get_short_uri(target_path),
            recursive=True,
        )

    @override
    def remove_directory(self, path: Path) -> None:
        # For S3/remote filesystems, rmdir only removes empty directories
        # Use rm with recursive=True to delete directory and all contents
        self._fsspec.rm(self.get_short_uri(path), recursive=True)

    @override
    def glob(self, path: Path, pattern: str) -> Iterator[Path]:
        paths = self._fsspec.glob(self.get_short_uri(path.joinpath(pattern)))
        return iter([Path(self.get_long_uri(p)) for p in paths])  # type: ignore

    @override
    def walk_directory(
        self, path: Path, *, on_error: Callable[[OSError], None] | None
    ) -> Iterator[tuple[str, Sequence[str]]]:
        results = self._fsspec.walk(self.get_short_uri(path), on_error=on_error)  # type: ignore
        for dir_pathname, _, filenames in results:
            yield self.get_long_uri(dir_pathname), self.get_long_uri(filenames)  # type: ignore

    @override
    @contextmanager
    def tmp_directory(self, base_dir: Path) -> Iterator[Path]:
        if self.is_local:
            with TemporaryDirectory(dir=base_dir) as dirname:
                yield Path(dirname)
        else:
            import uuid

            tmp_path = base_dir / f"tmp_{uuid.uuid4().hex}"
            self.make_directory(tmp_path)
            try:
                yield tmp_path
            finally:
                try:
                    self._fsspec.rm(self.get_short_uri(tmp_path), recursive=True)
                except Exception:
                    pass

    @override
    def resolve(self, path: Path) -> Path:
        if self.is_local:
            return path.expanduser().resolve()
        raise NotImplementedError("resolve is not implemented for this filesystem")

    @property
    @override
    def is_local(self) -> bool:
        try:
            return isinstance(self._fsspec, fs_local.LocalFileSystem)  # type: ignore
        except ImportError:
            return False


class FileSystemRegistry:
    """
    Registry for resolving FileSystem instances based on path patterns.
    """

    _resolvers: List[Tuple[Callable[[PathOrPaths], bool], Callable[[], FileSystem]]] = (
        []
    )
    _fs_cache: Dict[Any, FileSystem] = {}
    _local_fs: FileSystem = FSspecFileSystem(fs_local.LocalFileSystem(), "")

    @classmethod
    def register(
        cls,
        pattern_check: Callable[[PathOrPaths], bool],
        fs_factory: Callable[[], FileSystem],
    ) -> None:
        """
        Register a new filesystem resolver rule.

        Args:
            pattern_check: Function that checks if a path matches this filesystem type
            fs_factory: Function that creates a filesystem instance
        """
        cls._resolvers.append((pattern_check, fs_factory))

    @classmethod
    def resolve_filesystem(cls, path: PathOrPaths) -> FileSystem:
        """
        Resolve a FileSystem instance from a path.

        Args:
            path: A string, Path object, or list of strings/Paths representing file path(s).

        Returns:
            A FileSystem instance appropriate for the given path(s).
        """
        # Handle list of paths - use the first path to determine the filesystem
        if isinstance(path, Sequence) and not isinstance(path, (str, Path)):
            if not path:  # Handle empty sequence case
                return NativeLocalFileSystem()

            # Use the first path to determine the filesystem
            fs = cls.resolve_filesystem(path[0])

            # Check if all paths resolve to the same filesystem type
            for p in path[1:]:
                other_fs = cls.resolve_filesystem(p)
                if type(other_fs) is not type(fs):
                    raise ValueError(
                        f"Paths in the sequence resolve to different filesystem types: {type(fs)} and {type(other_fs)}"
                    )

            return fs

        path_str = str(path) if isinstance(path, Path) else path

        # Check registered resolvers
        for pattern_check, fs_factory in cls._resolvers:
            if pattern_check(path_str):
                if pattern_check in cls._fs_cache:
                    # XXX: caching lambda functions should be ok, but not the best practice
                    return cls._fs_cache[pattern_check]
                else:
                    fs = fs_factory()
                    cls._fs_cache[pattern_check] = fs
                    return fs

        # Default to native local filesystem
        return cls._local_fs


def _register_omnilingual_s3_filesystem() -> None:
    """Register custom S3 filesystem with omnilingual AWS profile for specific bucket access."""
    try:
        from stopes.wrapped_s3fs import get_s3_filesystem_with_reconnection
    except ImportError:
        log.debug(
            "stopes.wrapped_s3fs not available, skipping omnilingual S3 registration"
        )
        return

    profile_name = "omnilingual-nouserdata--use2-az3--x-s3-rw"
    bucket_pattern = "omnilingual-nouserdata--use2-az3--x-s3"

    try:
        pa_s3_fs = get_s3_filesystem_with_reconnection(
            profile=profile_name, as_fsspec=True
        )

        def s3_pattern_check(path: PathOrPaths) -> bool:
            path_str = str(path)
            return path_str.startswith("s3:/") and bucket_pattern in path_str

        wrapped_fs = FSspecFileSystem(pa_s3_fs, "s3://")
        FileSystemRegistry.register(s3_pattern_check, lambda: wrapped_fs)
        log.info(
            f"Registered S3 filesystem for {bucket_pattern} with profile {profile_name}"
        )
    except Exception as e:
        log.debug(f"Failed to register omnilingual S3 filesystem: {e}")


def _register_filesystems(context: Any) -> None:
    # Protocols to skip - either problematic or not needed
    # ftp: Has cleanup issues (__del__ AttributeError)
    # Only register protocols that have been tested and verified to work correctly.
    # - file/local: Local filesystem (both are aliases in fsspec)
    # - s3: Amazon S3 storage
    # Please extened if needed !
    TESTED_PROTOCOLS = {"file", "local", "s3"}

    activated_protocol = []
    for protocol in available_protocols():
        if protocol not in TESTED_PROTOCOLS:
            continue

        # FIXME: to propagate different credentials from the context
        storage_options: Dict[str, Any] = {}
        if hasattr(context, "storage_options"):
            all_storage_options = getattr(context, "storage_options", {})
            storage_options = all_storage_options.get(protocol, {})

        try:
            fsspec_instance = filesystem(protocol, **storage_options)
            prefix = f"{protocol}://"

            def make_pattern_check(
                pref: str,
            ) -> Callable[[PathOrPaths], bool]:
                return lambda p: str(p).startswith(pref)

            def make_fs_factory(
                fs: fsspec.AbstractFileSystem, pref: str
            ) -> Callable[[], FileSystem]:
                return lambda: FSspecFileSystem(fs, pref)

            FileSystemRegistry.register(
                make_pattern_check(prefix),
                make_fs_factory(fsspec_instance, prefix),
            )
            activated_protocol.append(protocol)
        except Exception:
            pass
    log.info(f"Activated FileSystem protocols:\n {activated_protocol}")


path_fs_resolver = FileSystemRegistry.resolve_filesystem


@final
class GlobalFileSystem(FileSystem):

    @override
    def is_dir(self, path: Path) -> bool:
        return path_fs_resolver(path).is_dir(path)

    @override
    def open(self, path: Path, mode: FileMode = FileMode.READ) -> BinaryIO:
        return path_fs_resolver(path).open(path, mode)

    @override
    def is_file(self, path: Path) -> bool:
        return path_fs_resolver(path).is_file(path)

    @override
    def open_text(self, path: Path, mode: FileMode = FileMode.READ) -> TextIO:
        return path_fs_resolver(path).open_text(path, mode)

    @override
    def cat(self, path: Path) -> bytes:
        return path_fs_resolver(path).cat(path)

    @override
    def exists(self, path: Path) -> bool:
        return path_fs_resolver(path).exists(path)

    @override
    def move(self, old_path: Path, new_path: Path) -> None:
        path_fs_resolver(old_path).move(old_path, new_path)

    @override
    def remove(self, path: Path) -> None:
        path_fs_resolver(path).remove(path)

    @override
    def make_directory(self, path: Path) -> None:
        path_fs_resolver(path).make_directory(path)

    @override
    def copy_directory(self, source_path: Path, target_path: Path) -> None:
        path_fs_resolver(source_path).copy_directory(source_path, target_path)

    @override
    def remove_directory(self, path: Path) -> None:
        path_fs_resolver(path).remove_directory(path)

    @override
    def glob(self, path: Path, pattern: str) -> Iterator[Path]:
        return path_fs_resolver(path).glob(path, pattern)

    @override
    def walk_directory(
        self, path: Path, *, on_error: Callable[[OSError], None] | None
    ) -> Iterator[tuple[str, Sequence[str]]]:
        return path_fs_resolver(path).walk_directory(path, on_error=on_error)

    @override
    def tmp_directory(self, base_dir: Path) -> ContextManager[Path]:
        return path_fs_resolver(base_dir).tmp_directory(base_dir)

    @override
    def resolve(self, path: Path) -> Path:
        return path_fs_resolver(path).resolve(path)

    @property
    @override
    def is_local(self) -> bool:
        return False

    @override
    def is_local_path(self, path: Path) -> bool:
        """Check if a specific path is on a local filesystem."""
        return path_fs_resolver(path).is_local


@final
class NativeLocalFileSystem(FileSystem):
    # keeping it for debugging purposes, it should be equivalent to LocalFileSystem

    @override
    def is_file(self, path: Path) -> bool:
        return path.is_file()

    @override
    def is_dir(self, path: Path) -> bool:
        return path.is_dir()

    @override
    def open(self, path: Path, mode: FileMode = FileMode.READ) -> BinaryIO:
        match mode:
            case FileMode.READ:
                m = "rb"
            case FileMode.WRITE:
                m = "wb"
            case FileMode.APPEND:
                m = "ab"
            case _:
                raise ValueError(
                    f"`mode` must be a valid `FileMode` value, but is `{mode}` instead."
                )

        fp = path.open(m)

        return cast(BinaryIO, fp)

    @override
    def open_text(self, path: Path, mode: FileMode = FileMode.READ) -> TextIO:
        match mode:
            case FileMode.READ:
                m = "r"
            case FileMode.WRITE:
                m = "w"
            case FileMode.APPEND:
                m = "a"
            case _:
                raise ValueError(
                    f"`mode` must be a valid `FileMode` value, but is `{mode}` instead."
                )

        fp = path.open(m, encoding="utf-8")

        return cast(TextIO, fp)

    @override
    def cat(self, path: Path) -> bytes:
        return path.read_bytes()

    @override
    def exists(self, path: Path) -> bool:
        return path.exists()

    @override
    def move(self, old_path: Path, new_path: Path) -> None:
        old_path.replace(new_path)

    @override
    def remove(self, path: Path) -> None:
        path.unlink()

    @override
    def make_directory(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @override
    def copy_directory(self, source_path: Path, target_path: Path) -> None:
        copytree(source_path, target_path, dirs_exist_ok=True)

    @override
    def remove_directory(self, path: Path) -> None:
        rmtree(path)

    @override
    def glob(self, path: Path, pattern: str) -> Iterator[Path]:
        return path.glob(pattern)

    @override
    def walk_directory(
        self, path: Path, *, on_error: Callable[[OSError], None] | None
    ) -> Iterator[tuple[str, Sequence[str]]]:
        for dir_pathname, _, filenames in os.walk(path, onerror=on_error):
            yield dir_pathname, filenames

    @override
    @contextmanager
    def tmp_directory(self, base_dir: Path) -> Iterator[Path]:
        with TemporaryDirectory(dir=base_dir) as dirname:
            yield Path(dirname)

    @override
    def resolve(self, path: Path) -> Path:
        return path.expanduser().resolve()

    @property
    @override
    def is_local(self) -> bool:
        return True


def raise_if_not_exists(file_system: FileSystem, path: Path) -> None:
    """Raises a :class:`FileNotFoundError` if ``path`` does not exist."""
    if not file_system.exists(path):
        raise FileNotFoundError(ENOENT, strerror(ENOENT), path)


LocalFileSystem = NativeLocalFileSystem


def _flush_nfs_lookup_cache(path: Path) -> None:
    # Use the `opendir`/`readdir`/`closedir` trick to drop all cached NFS
    # LOOKUP results.
    while path != path.parent:
        try:
            it = scandir(path)
        except FileNotFoundError:
            path = path.parent

            continue
        except OSError:
            break

        try:
            next(it)
        except StopIteration:
            pass
        finally:
            it.close()

        break
