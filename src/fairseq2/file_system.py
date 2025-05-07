# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from enum import Enum
from pathlib import Path
from shutil import copytree, rmtree
from typing import Any, BinaryIO, Dict, List, TextIO, Tuple, TypeAlias, cast, final

import fsspec  # type: ignore[import-untyped]
import fsspec.implementations.local as fs_local  # type: ignore[import-untyped]
from fsspec.registry import (  # type: ignore[import-untyped]
    available_protocols,
    filesystem,
)
from typing_extensions import override

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
    def glob(self, path: Path, pattern: str) -> Iterable[Path]: ...

    @abstractmethod
    def walk_directory(
        self, path: Path, *, on_error: Callable[[OSError], None] | None
    ) -> Iterable[tuple[str, Sequence[str]]]: ...

    @abstractmethod
    def resolve(self, path: Path) -> Path: ...

    @property
    @abstractmethod
    def is_local(self) -> bool: ...


ExtenedPath: TypeAlias = str | Path | Sequence[str | Path]


@final
class FSspecFileSystem(FileSystem):
    """
    Wrapper around fsspec to provide a FileSystem interface.
    >>> from s3fs import S3FileSystem
    >>> fs = S3FileSystem()
    >>> fs = FSspecFileSystem(fs)
    """

    __fsspec: fsspec.AbstractFileSystem
    __prefix: str

    def __init__(self, fs: fsspec.AbstractFileSystem, prefix: str) -> None:
        self.__fsspec = fs
        self.__prefix = prefix

    def get_short_uri(self, path: ExtenedPath) -> str | list[str]:
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
        path_str = str(path)
        if self.is_local:
            return path_str
        if path_str.startswith(self.__prefix):
            return path_str[len(self.__prefix) :].lstrip("/")
        else:
            raise ValueError(f"Path {path} does not start with prefix {self.__prefix}")

    def get_long_uri(self, path: ExtenedPath) -> str | list[str]:
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
        path_str = str(path)
        if not path_str.startswith(self.__prefix):
            return self.__prefix + path_str
        else:
            raise ValueError(f"Path {path} already starts with prefix {self.__prefix}")

    @override
    def is_file(self, path: Path) -> bool:
        return bool(self.__fsspec.isfile(self.get_short_uri(path)))

    @override
    def is_dir(self, path: Path) -> bool:
        return bool(self.__fsspec.isdir(self.get_short_uri(path)))

    @override
    def exists(self, path: Path) -> bool:
        return bool(self.__fsspec.exists(self.get_short_uri(path)))

    @override
    def open(self, path: Path, mode: FileMode = FileMode.READ) -> BinaryIO:
        mode_str = "rb"
        if mode == FileMode.WRITE:
            mode_str = "wb"
        elif mode == FileMode.APPEND:
            mode_str = "ab"

        return cast(BinaryIO, self.__fsspec.open(self.get_short_uri(path), mode_str))

    @override
    def open_text(self, path: Path, mode: FileMode = FileMode.READ) -> TextIO:
        mode_str = "r"
        if mode == FileMode.WRITE:
            mode_str = "w"
        elif mode == FileMode.APPEND:
            mode_str = "a"

        return cast(
            TextIO,
            self.__fsspec.open(self.get_short_uri(path), mode_str, encoding="utf-8"),
        )

    @override
    def move(self, old_path: Path, new_path: Path) -> None:
        self.__fsspec.mv(
            self.get_short_uri(old_path), self.get_short_uri(new_path), recursive=True
        )

    @override
    def remove(self, path: Path) -> None:
        # only one file !
        self.__fsspec.rm(self.get_short_uri(path), recursive=False)

    @override
    def make_directory(self, path: Path) -> None:
        self.__fsspec.makedirs(self.get_short_uri(path), exist_ok=True)

    @override
    def copy_directory(self, source_path: Path, target_path: Path) -> None:
        self.__fsspec.copy(
            self.get_short_uri(source_path),
            self.get_short_uri(target_path),
            recursive=True,
        )

    @override
    def remove_directory(self, path: Path) -> None:
        self.__fsspec.rmdir(self.get_short_uri(path))

    @override
    def glob(self, path: Path, pattern: str) -> Iterable[Path]:
        paths = self.__fsspec.glob(self.get_short_uri(path.joinpath(pattern)))
        return [Path(self.get_long_uri(p)) for p in paths]  # type: ignore

    @override
    def walk_directory(
        self, path: Path, *, on_error: Callable[[OSError], None] | None
    ) -> Iterable[tuple[str, Sequence[str]]]:
        results = self.__fsspec.walk(self.get_short_uri(path), on_error=on_error)  # type: ignore
        for dir_pathname, _, filenames in results:
            yield self.get_long_uri(dir_pathname), self.get_long_uri(filenames)  # type: ignore

    @override
    def resolve(self, path: Path) -> Path:
        # Use expanduser if available, otherwise just convert to string and back
        if self.is_local:
            return path.expanduser().resolve()
        raise NotImplementedError("resolve is not implemented for this filesystem")

    @property
    @override
    def is_local(self) -> bool:
        try:
            return isinstance(self.__fsspec, fs_local.LocalFileSystem)  # type: ignore
        except ImportError:
            return False


class FileSystemRegistry:
    """
    Registry for resolving FileSystem instances based on path patterns.
    """

    _resolvers: List[Tuple[Callable[[ExtenedPath], bool], Callable[[], FileSystem]]] = (
        []
    )
    _fs_cache: Dict[Any, FileSystem] = {}
    _local_fs: FileSystem = FSspecFileSystem(fs_local.LocalFileSystem(), "")

    @classmethod
    def register(
        cls,
        pattern_check: Callable[[ExtenedPath], bool],
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
    def resolve_filesystem(cls, path: ExtenedPath) -> FileSystem:
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
                if type(other_fs) != type(fs):
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


def register_filesystems(context: Any) -> None:
    activated_protocol = []
    for protocol in available_protocols():
        # FIXME: to propagate different credentials from the context
        storage_options: Dict[str, Any] = {}
        if hasattr(context, "storage_options"):
            all_storage_options = getattr(context, "storage_options", {})
            storage_options = all_storage_options.get(protocol, {})

        try:
            fsspec = filesystem(protocol, **storage_options)
            prefix = f"{protocol}://"
            FileSystemRegistry.register(
                lambda p: str(p).startswith(prefix),
                lambda: FSspecFileSystem(fsspec, prefix),
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
    def glob(self, path: Path, pattern: str) -> Iterable[Path]:
        return path_fs_resolver(path).glob(path, pattern)

    @override
    def walk_directory(
        self, path: Path, *, on_error: Callable[[OSError], None] | None
    ) -> Iterable[tuple[str, Sequence[str]]]:
        return path_fs_resolver(path).walk_directory(path, on_error=on_error)

    @override
    def resolve(self, path: Path) -> Path:
        return path_fs_resolver(path).resolve(path)

    @property
    @override
    def is_local(self) -> bool:
        return False


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
    def glob(self, path: Path, pattern: str) -> Iterable[Path]:
        return path.glob(pattern)

    @override
    def walk_directory(
        self, path: Path, *, on_error: Callable[[OSError], None] | None
    ) -> Iterable[tuple[str, Sequence[str]]]:
        for dir_pathname, _, filenames in os.walk(path, onerror=on_error):
            yield dir_pathname, filenames

    @override
    def resolve(self, path: Path) -> Path:
        return path.expanduser().resolve()

    @final
    @property
    @override
    def is_local(self) -> bool:
        return True
