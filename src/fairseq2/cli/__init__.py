# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.cli._cli import Cli as Cli
from fairseq2.cli._cli import CliArgumentError as CliArgumentError
from fairseq2.cli._cli import CliCommand as CliCommand
from fairseq2.cli._cli import CliCommandError as CliCommandError
from fairseq2.cli._cli import CliCommandHandler as CliCommandHandler
from fairseq2.cli._cli import CliGroup as CliGroup
from fairseq2.cli._logging import setup_logging as setup_logging
from fairseq2.cli._main import main as main
from fairseq2.cli._setup import setup_cli as setup_cli
