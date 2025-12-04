# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod

from fairseq2.runtime.closable import Closable


class Task(Closable):
    @abstractmethod
    def run(self) -> None: ...

    @abstractmethod
    def request_stop(self) -> None: ...

    @property
    @abstractmethod
    def step_nr(self) -> int: ...


class TaskStopException(Exception):
    def __init__(self) -> None:
        super().__init__("Task stopped.")
