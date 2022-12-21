import bdb
import pdb
import traceback
from typing import Any, Union

import torchtnt.framework
import torchtnt.utils
from torchtnt.framework.callback import Callback
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit


class Debugger(Callback):
    def on_exception(
        self,
        state: torchtnt.framework.State,
        unit: Union[TTrainUnit[Any], TEvalUnit[Any], TPredictUnit[Any]],
        exc: BaseException,
    ) -> None:
        if torchtnt.utils.get_global_rank() != 0:
            # TODO: use rpdb in distributed env
            return
        if isinstance(exc, bdb.BdbQuit):
            return
        if not isinstance(exc, Exception):
            return
        traceback.print_exc()
        pdb.post_mortem()
