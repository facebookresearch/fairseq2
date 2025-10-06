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
from os import strerror
from pathlib import Path
from shutil import copytree, rmtree
from tempfile import TemporaryDirectory
from typing import BinaryIO, TextIO, cast, final

from typing_extensions import override

from fairseq2.typing import ContextManager


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
    def is_local(self) -> bool: ...


@final
class LocalFileSystem(FileSystem):
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
