# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

class DataSource:
    @staticmethod
    def list_files(paths: List[str], pattern: Optional[str] = None) -> DataSource: ...
