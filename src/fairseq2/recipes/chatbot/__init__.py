# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipes.chatbot.run import RunChatbotCommand
from fairseq2.recipes.cli import Cli


def _setup_chatbot_cli(cli: Cli) -> None:
    group = cli.add_group("chatbot", help="chatbot demo")

    group.add_command(
        "run",
        RunChatbotCommand(),
        help="run a terminal-based chatbot demo",
    )
