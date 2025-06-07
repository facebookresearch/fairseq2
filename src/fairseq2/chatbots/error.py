# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


class UnknownChatbotError(LookupError):
    model_name: str

    def __init__(self, model_name: str) -> None:
        super().__init__(f"'{model_name}' model does not have a chatbot.")

        self.model_name = model_name
