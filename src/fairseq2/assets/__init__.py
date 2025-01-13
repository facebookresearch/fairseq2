# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.assets.card import AssetCard as AssetCard
from fairseq2.assets.download_manager import AssetDownloadError as AssetDownloadError
from fairseq2.assets.download_manager import (
    AssetDownloadManager as AssetDownloadManager,
)
from fairseq2.assets.download_manager import (
    InProcAssetDownloadManager as InProcAssetDownloadManager,
)
from fairseq2.assets.error import AssetCardError as AssetCardError
from fairseq2.assets.error import (
    AssetCardFieldNotFoundError as AssetCardFieldNotFoundError,
)
from fairseq2.assets.error import AssetCardNotFoundError as AssetCardNotFoundError
from fairseq2.assets.error import AssetError as AssetError
from fairseq2.assets.metadata_provider import (
    AbstractAssetMetadataProvider as AbstractAssetMetadataProvider,
)
from fairseq2.assets.metadata_provider import AssetMetadataError as AssetMetadataError
from fairseq2.assets.metadata_provider import (
    AssetMetadataNotFoundError as AssetMetadataNotFoundError,
)
from fairseq2.assets.metadata_provider import (
    AssetMetadataProvider as AssetMetadataProvider,
)
from fairseq2.assets.metadata_provider import (
    FileAssetMetadataProvider as FileAssetMetadataProvider,
)
from fairseq2.assets.metadata_provider import (
    InProcAssetMetadataProvider as InProcAssetMetadataProvider,
)
from fairseq2.assets.metadata_provider import (
    PackageAssetMetadataProvider as PackageAssetMetadataProvider,
)
from fairseq2.assets.metadata_provider import PackageFileLister as PackageFileLister
from fairseq2.assets.metadata_provider import (
    WheelPackageFileLister as WheelPackageFileLister,
)
from fairseq2.assets.metadata_provider import load_metadata_file as load_metadata_file
from fairseq2.assets.store import AssetStore as AssetStore
from fairseq2.assets.store import EnvironmentResolver as EnvironmentResolver
from fairseq2.assets.store import StandardAssetStore as StandardAssetStore
from fairseq2.assets.store import default_asset_store as default_asset_store
from fairseq2.assets.store import get_asset_dir as get_asset_dir
from fairseq2.assets.store import get_user_asset_dir as get_user_asset_dir
