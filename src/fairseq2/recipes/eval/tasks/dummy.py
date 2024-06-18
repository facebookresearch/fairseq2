# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, List, Optional, TYPE_CHECKING, Sequence
from fairseq2.recipes.eval.api import Decoder, Example, MetricFn, Task, TaskConfig, TaskResult
from fairseq2.generation.generator import SequenceGenerator

if TYPE_CHECKING:
    from numpy.random import RandomState


class DummyTaskConfig(TaskConfig):
    """"""


class DummyTask(Task):
    """
    A simple task to demonstrate the fairseq2 eval API.
    
    Each task named XYZ must have a config class with the named
    XYZConfig, that specifies the list of parameters to customize
    the task's behaviour. Those parameters can also be given by
    the user via the CLI.

    A task is registered to the TaskRegistry via register_task()
    decorator that should give a unique task name, plus the
    parameters to set up the task config. A common pratice is to
    tie the task to a dataset and some metrics (for instance from
    HuggingFace's evaluate library, or else where) to evaluate a
    fairseq2 model. 
    """

    def __init__(
        self,
        dataset,  # Add basic dataloader 
        metric_fns: Optional[Sequence[MetricFn]],
    ):
        #TODO: continue
        pass

    @staticmethod
    def from_config(cfg: DummyTaskConfig) -> "DummyTask":
        #TODO: continue
        pass

    def run(
        self,
        generator: SequenceGenerator,
        decoder: Optional[Decoder] = None,
        random_state: Optional[RandomState] = None,
        max_samples: Optional[int] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TaskResult:
        #TODO: continue
        pass
