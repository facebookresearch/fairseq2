# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.assets.card import AssetCard as AssetCard
from fairseq2.assets.card import AssetCardError as AssetCardError
from fairseq2.assets.card import (
    AssetCardFieldNotFoundError as AssetCardFieldNotFoundError,
)
from fairseq2.assets.card import AssetCardNotFoundError as AssetCardNotFoundError
from fairseq2.assets.dirs import AssetDirectories as AssetDirectories
from fairseq2.assets.download_manager import AssetDownloadError as AssetDownloadError
from fairseq2.assets.download_manager import (
    AssetDownloadManager as AssetDownloadManager,
)
from fairseq2.assets.download_manager import (
    CompositeAssetDownloadManager as CompositeAssetDownloadManager,
)
from fairseq2.assets.download_manager import HuggingFaceHub as HuggingFaceHub
from fairseq2.assets.download_manager import (
    InProcAssetDownloadManager as InProcAssetDownloadManager,
)
from fairseq2.assets.download_manager import (
    NoopAssetDownloadManager as NoopAssetDownloadManager,
)
from fairseq2.assets.download_manager import (
    create_asset_download_manager as create_asset_download_manager,
)
from fairseq2.assets.metadata_provider import (
    AssetMetadataFileLoader as AssetMetadataFileLoader,
)
from fairseq2.assets.metadata_provider import (
    AssetMetadataLoadError as AssetMetadataLoadError,
)
from fairseq2.assets.metadata_provider import (
    AssetMetadataNotFoundError as AssetMetadataNotFoundError,
)
from fairseq2.assets.metadata_provider import (
    AssetMetadataProvider as AssetMetadataProvider,
)
from fairseq2.assets.metadata_provider import (
    AssetMetadataSaveError as AssetMetadataSaveError,
)
from fairseq2.assets.metadata_provider import (
    CachedAssetMetadataProvider as CachedAssetMetadataProvider,
)
from fairseq2.assets.metadata_provider import (
    FileBackedAssetMetadataLoader as FileBackedAssetMetadataLoader,
)
from fairseq2.assets.metadata_provider import (
    InProcAssetMetadataLoader as InProcAssetMetadataLoader,
)
from fairseq2.assets.metadata_provider import (
    PackageBackedAssetMetadataLoader as PackageBackedAssetMetadataLoader,
)
from fairseq2.assets.metadata_provider import PackageFileLister as PackageFileLister
from fairseq2.assets.metadata_provider import (
    WheelPackageFileLister as WheelPackageFileLister,
)
from fairseq2.assets.metadata_provider import (
    YamlAssetMetadataFileLoader as YamlAssetMetadataFileLoader,
)
from fairseq2.assets.metadata_provider import (
    load_package_asset_provider as load_package_asset_provider,
)
from fairseq2.assets.metadata_provider import (
    register_package_asset_provider as register_package_asset_provider,
)
from fairseq2.assets.store import AssetLookupScope as AssetLookupScope
from fairseq2.assets.store import AssetStore as AssetStore
from fairseq2.assets.store import EnvironmentResolver as EnvironmentResolver
from fairseq2.assets.store import StandardAssetStore as StandardAssetStore
from fairseq2.assets.store import load_asset_store as load_asset_store
