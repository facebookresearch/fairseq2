# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from pickle import PickleError
from shutil import copytree, rmtree
from typing import BinaryIO, TextIO, TypeAlias, cast, final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.error import NotSupportedError
from fairseq2.typing import Device


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

        fp = path.open(m)

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


MapLocation: TypeAlias = (
    Callable[[Tensor, str], Tensor] | Device | str | dict[str, str] | None
)


class TensorLoader(ABC):
    """Loads tensors from files."""

    @abstractmethod
    def load(
        self, path: Path, *, map_location: MapLocation = None
    ) -> dict[str, object]:
        """
        :param path:
            The path to the file.
        :param map_location:
            Same as the ``map_location`` parameter of :meth:`torch.load`.
        """


class TensorDumper(ABC):
    """Dumps tensors to files."""

    @abstractmethod
    def dump(self, data: Mapping[str, object], path: Path) -> None:
        """
        :param data:
            The dictionary containing tensors and other auxiliary data.
        :param path:
            The path to the file.
        """


@final
class TorchTensorLoader(TensorLoader):
    _file_system: FileSystem
    _restrict: bool

    def __init__(self, file_system: FileSystem, restrict: bool = True) -> None:
        self._file_system = file_system
        self._restrict = restrict

    @override
    def load(
        self, path: Path, *, map_location: MapLocation = None
    ) -> dict[str, object]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message=r".*You are using `torch\.load` with `weights_only=False`.*"  # fmt: skip
            )

            def load_error() -> TensorLoadError:
                return TensorLoadError(
                    path, f"The '{path}' tensor file cannot be loaded. See the nested exception for details."  # fmt: skip
                )

            try:
                fp = self._file_system.open(path)
            except FileNotFoundError:
                raise
            except OSError as ex:
                raise load_error() from ex

            try:
                data: dict[str, object] = torch.load(
                    fp, map_location, weights_only=self._restrict  # type: ignore[arg-type]
                )
            except (RuntimeError, OSError, PickleError) as ex:
                raise load_error() from ex
            finally:
                fp.close()

        return data


@final
class TorchTensorDumper(TensorDumper):
    _file_system: FileSystem

    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

    @override
    def dump(self, data: Mapping[str, object], path: Path) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message=r".*Please use DTensor instead.*"
            )

            def dump_error() -> TensorDumpError:
                return TensorDumpError(
                    path, f"The '{path}' tensor file cannot be dumped. See the nested exception for details.",  # fmt: skip
                )

            try:
                fp = self._file_system.open(path, mode=FileMode.WRITE)
            except OSError as ex:
                raise dump_error() from ex

            try:
                torch.save(data, fp)
            except (RuntimeError, OSError, PickleError) as ex:
                raise dump_error() from ex
            finally:
                fp.close()


@final
class SafetensorLoader(TensorLoader):
    """Loads the Hugging Face Safetensors file(s)."""

    _file_system: FileSystem

    def __init__(self, file_system: FileSystem) -> None:
        if not file_system.is_local:
            raise NotSupportedError("Safetensors supports only local file system.")

        self._file_system = file_system

    @override
    def load(
        self, path: Path, *, map_location: MapLocation = None
    ) -> dict[str, object]:
        try:
            from safetensors import safe_open  # type: ignore[import-not-found]
        except ImportError:
            raise NotSupportedError(
                "Safetensors not found in your Python environment. Use `pip install safetensors`."
            )

        if map_location is not None:
            if not isinstance(map_location, (Device, str)):
                raise NotSupportedError(
                    "Safetensors only supports `torch.device` and `str` for the `map_location` parameter."
                )

        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            raise TensorLoadError(
                path, f"The '{path}' path cannot be accessed. See the nested exception for details."  # fmt: skip
            ) from ex

        if is_dir:
            try:
                files = list(self._file_system.glob(path, "*.safetensors"))
            except OSError as ex:
                raise TensorLoadError(
                    path, f"The '{path}' directory cannot be traversed. See the nested exception for details."  # fmt: skip
                ) from ex

            if not files:
                raise TensorLoadError(
                    path, f"No Safetensors file found under the '{path}' directory."  # fmt: skip
                )
        else:
            files = [path]

        tensors = {}

        for file in files:
            try:
                with safe_open(file, framework="pt", device=str(map_location)) as f:  # type: ignore[attr-defined]
                    for k in f.keys():
                        if k in tensors:
                            raise TensorLoadError(
                                path, f"The '{k}' key exists in more than one Safetensors file under the '{path}' directory."  # fmt: skip
                            )

                        tensors[k] = f.get_tensor(k)
            except FileNotFoundError:
                raise
            except (RuntimeError, OSError, PickleError) as ex:
                raise TensorLoadError(
                    file, f"The '{file}' tensor file cannot be loaded. See the nested exception for details."  # fmt: skip
                ) from ex

        return tensors


@final
class StandardTensorLoader(TensorLoader):
    _file_system: FileSystem
    _default_tensor_loader: TensorLoader

    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

        self._default_tensor_loader = TorchTensorLoader(self._file_system)

    @override
    def load(
        self, path: Path, *, map_location: MapLocation = None
    ) -> dict[str, object]:
        def has_files(extension: str) -> bool:
            try:
                next(iter(self._file_system.glob(path, f"*{extension}")))
            except OSError as ex:
                raise TensorLoadError(
                    path, f"The '{path}' directory cannot be traversed. See the nested exception for details."  # fmt: skip
                ) from ex
            except StopIteration:
                return False

            return True

        loader: TensorLoader

        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            raise TensorLoadError(
                path, f"The '{path}' path cannot be accessed. See the nested exception for details."  # fmt: skip
            ) from ex

        if is_dir:
            if not has_files(".safetensors"):
                raise TensorLoadError(
                    path, f"The '{path}' directory does not contain any supported tensor files."  # fmt: skip
                )

            loader = SafetensorLoader(self._file_system)
        elif path.suffix == ".safetensors":
            loader = SafetensorLoader(self._file_system)
        else:
            loader = self._default_tensor_loader

        return loader.load(path, map_location=map_location)


class TensorLoadError(Exception):
    path: Path

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)

        self.path = path


class TensorDumpError(Exception):
    path: Path

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)

        self.path = path
