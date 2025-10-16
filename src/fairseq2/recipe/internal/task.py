# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from signal import SIGUSR1, signal
from types import FrameType
from typing import final

from fairseq2.error import raise_operational_system_error
from fairseq2.gang import GangError, Gangs, raise_operational_gang_error
from fairseq2.logging import log
from fairseq2.recipe.task import Task, TaskStopException
from fairseq2.utils.stopwatch import Stopwatch


@final
class _TaskRunner:
    def __init__(self, gangs: Gangs, wall_watch: Stopwatch) -> None:
        self._gangs = gangs
        self._wall_watch = wall_watch

    def run(self, task: Task) -> None:
        log.info("Running on {} process(es).", self._gangs.root.size)

        # Use SIGUSR1 as the stop signal.
        def request_stop(signum: int, frame: FrameType | None) -> None:
            log.info("SIGUSR1 received. Requesting recipe to stop.")

            task.request_stop()

        original_signal_handler = signal(SIGUSR1, request_stop)

        try:
            task.run()
        except OSError as ex:
            raise_operational_system_error(ex)
        except GangError as ex:
            raise_operational_gang_error(ex)
        except TaskStopException:
            elapsed_time = int(self._wall_watch.get_elapsed_time())

            if task.step_nr == 0:
                log.info("Task stopped after {:,} second(s)!", elapsed_time)
            else:
                log.info("Task stopped after {:,} second(s) at step {}!", elapsed_time, task.step_nr)  # fmt: skip

            raise
        except KeyboardInterrupt:
            elapsed_time = int(self._wall_watch.get_elapsed_time())

            if task.step_nr == 0:
                log.info("Task canceled after {:,} second(s)!", elapsed_time)
            else:
                log.info("Task canceled after {:,} second(s) at step {}!", elapsed_time, task.step_nr)  # fmt: skip

            raise
        else:
            elapsed_time = int(self._wall_watch.get_elapsed_time())

            if task.step_nr == 0:
                log.info("Task finished in {:,} second(s)!", elapsed_time)
            else:
                log.info("Task finished in {:,} second(s) after {} step(s)!", elapsed_time, task.step_nr)  # fmt: skip
        finally:
            task.close()

            signal(SIGUSR1, original_signal_handler)
