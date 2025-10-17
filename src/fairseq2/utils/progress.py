# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Final, TypeAlias, final

from tqdm.auto import tqdm as auto_tqdm
from typing_extensions import Self, override

from fairseq2.runtime.closable import Closable


class ProgressReporter(ABC):
    @abstractmethod
    def create_task(
        self, name: str, total: int | None, completed: int = 0, *, start: bool = True
    ) -> ProgressTask: ...

    @abstractmethod
    def __enter__(self) -> Self: ...

    @abstractmethod
    def __exit__(self, *ex: Any) -> None: ...

    def maybe_get_tqdm_kls(self) -> type | None:
        """
        Returns a tqdm class that when instantiated reports progress the same
        way as this reporter.

        The returned class is passed to internally called third-party APIs, such
        as Hugging Face Hub’s ``snapshot_download()``, to provide a consistent
        progress reporting interface.

        If ``None`` is returned, the default tqdm class expected by the
        third-party API will be used.
        """
        return None


class ProgressTask(Closable):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def step(self, value: int = 1) -> None: ...

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *ex: Any) -> None:
        self.close()


@final
class _NoopProgressReporter(ProgressReporter):
    @override
    def create_task(
        self, name: str, total: int | None, completed: int = 0, *, start: bool = True
    ) -> ProgressTask:
        return _NoopProgressTask()

    @override
    def __enter__(self) -> Self:
        return self

    @override
    def __exit__(self, *ex: Any) -> None:
        pass

    @override
    def maybe_get_tqdm_kls(self) -> type | None:
        return _noop_tqdm


NOOP_PROGRESS_REPORTER: Final = _NoopProgressReporter()


@final
class _NoopProgressTask(ProgressTask):
    @override
    def start(self) -> None:
        pass

    @override
    def step(self, value: int = 1) -> None:
        pass

    @override
    def close(self) -> None:
        pass


if TYPE_CHECKING:
    _tqdm: TypeAlias = auto_tqdm[Any]
else:
    _tqdm: TypeAlias = auto_tqdm


class _noop_tqdm(_tqdm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["disable"] = True

        super().__init__(*args, **kwargs)
