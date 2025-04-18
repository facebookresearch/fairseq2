# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
from pickle import PickleError
from typing import TypeAlias, final

import fsspec
import torch
from fsspec.implementations.local import LocalFileSystem as fsspec_LocalFileSystem
from torch import Tensor
from typing_extensions import override

try:
    import safetensors  # type: ignore[import-not-found]
    import safetensors.torch  # type: ignore[import-not-found]
except ImportError:
    _has_safetensors = False
else:
    _has_safetensors = True

from fairseq2.device import CPU, Device
from fairseq2.error import NotSupportedError
from fairseq2.file_system import FileMode, FileSystem


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
class NativeLocalFileSystem(FileSystem):
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


class FSspecFileSystem(FileSystem):
    """
    Wrapper around fsspec to provide a FileSystem interface.
    >>> from s3fs import S3FileSystem
    >>> fs = S3FileSystem()
    >>> fs = FSspecFileSystem(fs)
    """

    __fs: fsspec.AbstractFileSystem

    def __init__(self, fs: fsspec.AbstractFileSystem) -> None:
        self.__fs = fs

    @override
    def is_file(self, path: Path) -> bool:
        return self.__fs.isfile(str(path))

    @override
    def is_dir(self, path: Path) -> bool:
        return self.__fs.isdir(str(path))

    @override
    def exists(self, path: Path) -> bool:
        return self.__fs.exists(str(path))

    @override
    def open(self, path: Path, mode: FileMode = FileMode.READ) -> BinaryIO:
        mode_str = "rb"
        if mode == FileMode.WRITE:
            mode_str = "wb"
        elif mode == FileMode.APPEND:
            mode_str = "ab"

        return cast(BinaryIO, self.__fs.open(str(path), mode_str))

    @override
    def open_text(self, path: Path, mode: FileMode = FileMode.READ) -> TextIO:
        mode_str = "r"
        if mode == FileMode.WRITE:
            mode_str = "w"
        elif mode == FileMode.APPEND:
            mode_str = "a"

        return cast(TextIO, self.__fs.open(str(path), mode_str, encoding="utf-8"))

    @override
    def move(self, old_path: Path, new_path: Path) -> None:
        self.__fs.mv(str(old_path), str(new_path), recursive=True)

    @override
    def remove(self, path: Path) -> None:
        # only one file !
        self.__fs.rm(str(path), recursive=False)

    @override
    def make_directory(self, path: Path) -> None:
        self.__fs.makedirs(str(path), exist_ok=True)

    @override
    def copy_directory(self, source_path: Path, target_path: Path) -> None:
        self.__fs.copy(str(source_path), str(target_path), recursive=True)

    @override
    def remove_directory(self, path: Path) -> None:
        self.__fs.rmdir(str(path))

    @override
    def glob(self, path: Path, pattern: str) -> Iterable[Path]:
        paths = self.__fs.glob(str(path.joinpath(pattern)))
        return [Path(p) for p in paths]

    @override
    def walk_directory(
        self, path: Path, *, on_error: Callable[[OSError], None] | None
    ) -> Iterable[tuple[str, Sequence[str]]]:
        results = self.__fs.walk(str(path), on_error=on_error)  # type: ignore[arg-type]
        for dir_pathname, _, filenames in results:
            yield str(dir_pathname), filenames  # type: ignore[return-value]

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
            return isinstance(self.__fs, fsspec_LocalFileSystem)
        except ImportError:
            return False


@final
class LocalFileSystem(FSspecFileSystem):
    def __init__(self) -> None:
        super().__init__(fsspec_LocalFileSystem())


@final
class S3FileSystem(FSspecFileSystem):
    def __init__(self, *arg, **kwargs) -> None:
        from s3fs import S3FileSystem

        super().__init__(S3FileSystem(*arg, **kwargs))


>>>>>>> 6b11ad9d (adding integration with fsspec FileSystem):src/fairseq2/utils/file.py
MapLocation: TypeAlias = (
    Callable[[Tensor, str], Tensor] | Device | str | dict[str, str] | None
)


class TensorLoader(ABC):
    """Loads tensors from PyTorch binary files."""

    @abstractmethod
    def load(
        self,
        file: Path,
        *,
        map_location: MapLocation = None,
        mmap: bool = False,
        restrict: bool = True,
    ) -> dict[str, object]:
        """
        :param file: The path to the file.
        :param map_location: Same as the ``map_location`` parameter of
            :meth:`torch.load`.
        """


class TensorDumper(ABC):
    """Dumps tensors to PyTorch binary files."""

    @abstractmethod
    def dump(self, data: Mapping[str, object], file: Path) -> None:
        """
        :param data: The dictionary containing tensors and other auxiliary data.
        :param file: The path to the file.
        """


@final
class TorchTensorLoader(TensorLoader):
    _file_system: FileSystem

    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

    @override
    def load(
        self,
        file: Path,
        *,
        map_location: MapLocation = None,
        mmap: bool = False,
        restrict: bool = True,
    ) -> dict[str, object]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message=r".*You are using `torch\.load` with `weights_only=False`.*"  # fmt: skip
            )

            def load_error() -> TensorLoadError:
                return TensorLoadError(
                    file, f"The '{file}' tensor file cannot be loaded. See the nested exception for details."  # fmt: skip
                )

            try:
                data: dict[str, object] = torch.load(
                    file, map_location, weights_only=restrict, mmap=mmap  # type: ignore[arg-type]
                )
            except FileNotFoundError:
                raise
            except (RuntimeError, OSError, PickleError) as ex:
                raise load_error() from ex

        return data


@final
class TorchTensorDumper(TensorDumper):
    _file_system: FileSystem

    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

    @override
    def dump(self, data: Mapping[str, object], file: Path) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message=r".*Please use DTensor instead.*"
            )

            def dump_error() -> TensorDumpError:
                return TensorDumpError(
                    file, f"The '{file}' tensor file cannot be dumped. See the nested exception for details.",  # fmt: skip
                )

            try:
                fp = self._file_system.open(file, mode=FileMode.WRITE)
            except OSError as ex:
                raise dump_error() from ex

            try:
                torch.save(data, fp)
            except (RuntimeError, OSError, PickleError) as ex:
                raise dump_error() from ex
            finally:
                fp.close()


class SafetensorsLoader(ABC):
    """Loads Safetensors files."""

    @abstractmethod
    def load(
        self, file: Path, *, device: Device | None = None, mmap: bool = False
    ) -> dict[str, object]: ...


@final
class HuggingFaceSafetensorsLoader(SafetensorsLoader):
    _file_system: FileSystem

    def __init__(self, file_system: FileSystem) -> None:
        if not file_system.is_local:
            raise NotSupportedError("Safetensors supports only local file systems.")

        self._file_system = file_system

    @override
    def load(
        self, file: Path, *, device: Device | None = None, mmap: bool = False
    ) -> dict[str, object]:
        if not _has_safetensors:
            raise RuntimeError(
                "Safetensors is not found in your Python environment. Use `pip install safetensors`."
            )

        if device is None:
            device = CPU

        data = {}

        try:
            if mmap:
                with safetensors.safe_open(
                    file, framework="pt", device=str(device)
                ) as f:
                    for key in f.keys():
                        data[key] = f.get_tensor(key)
            else:
                with open(file, "rb") as f:
                    bits = f.read()

                tensors = safetensors.torch.load(bits)

                for key, tensor in tensors.items():
                    data[key] = tensor.to(device)

        except FileNotFoundError:
            raise
        except (RuntimeError, OSError, PickleError) as ex:
            raise TensorLoadError(
                file, f"The '{file}' tensor file cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

        return data


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
