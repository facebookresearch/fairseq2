# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping

import pytest

from fairseq2.utils.state import StatefulObjectBag, StateHandler


class TestStatefulObjectBag:
    def test_state_dict_works(self) -> None:
        class Foo:
            def state_dict(self) -> Dict[str, Any]:
                return {"foo4": "value4"}

            def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
                assert state_dict == {"foo4": "value4"}

        class FooStateHandler(StateHandler):
            def load_state(self, stateful: Any, state: Any) -> None:
                assert stateful == "value2" and state == "handler-value2"

            def extract_state(self, stateful: Any) -> Any:
                assert stateful == "value2"

                return "handler-value2"

        bag = StatefulObjectBag()

        bag.register_stateful("foo1", "value1")
        bag.register_stateful("foo2", "value2", FooStateHandler())

        bag.foo3 = Foo()

        bag.register_non_stateful("foo4", Foo())

        state_dict = bag.state_dict()

        assert state_dict == {
            "foo1": "value1",
            "foo2": "handler-value2",
            "foo3": {"foo4": "value4"},
        }

        del bag.foo3

        with pytest.raises(ValueError, match="^`state_dict` must contain items"):
            bag.load_state_dict(state_dict)

        bag.foo3 = Foo()

        bag.foo1 = "abc"

        assert bag.foo1 == "abc"

        bag.load_state_dict(state_dict)

        assert bag.foo1 == "value1"
