# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.assets.card import AssetCard as AssetCard
from fairseq2.assets.card import AssetCardError as AssetCardError
from fairseq2.assets.card import AssetConfigLoader as AssetConfigLoader
from fairseq2.assets.card import StandardAssetConfigLoader as StandardAssetConfigLoader
from fairseq2.assets.dirs import AssetDirectoryAccessor as AssetDirectoryAccessor
from fairseq2.assets.dirs import (
    StandardAssetDirectoryAccessor as StandardAssetDirectoryAccessor,
)
from fairseq2.assets.download_manager import AssetDownloadError as AssetDownloadError
from fairseq2.assets.download_manager import (
    AssetDownloadManager as AssetDownloadManager,
)
from fairseq2.assets.download_manager import (
    DelegatingAssetDownloadManager as DelegatingAssetDownloadManager,
)
from fairseq2.assets.download_manager import HuggingFaceHub as HuggingFaceHub
from fairseq2.assets.download_manager import (
    LocalAssetDownloadManager as LocalAssetDownloadManager,
)
from fairseq2.assets.download_manager import (
    StandardAssetDownloadManager as StandardAssetDownloadManager,
)
from fairseq2.assets.download_manager import (
    get_asset_download_manager as get_asset_download_manager,
)
from fairseq2.assets.metadata_provider import AssetMetadataError as AssetMetadataError
from fairseq2.assets.metadata_provider import (
    AssetMetadataFileLoader as AssetMetadataFileLoader,
)
from fairseq2.assets.metadata_provider import (
    AssetMetadataProvider as AssetMetadataProvider,
)
from fairseq2.assets.metadata_provider import AssetMetadataSource as AssetMetadataSource
from fairseq2.assets.metadata_provider import (
    AssetSourceNotFoundError as AssetSourceNotFoundError,
)
from fairseq2.assets.metadata_provider import (
    CachedAssetMetadataProvider as CachedAssetMetadataProvider,
)
from fairseq2.assets.metadata_provider import (
    FileAssetMetadataLoader as FileAssetMetadataLoader,
)
from fairseq2.assets.metadata_provider import (
    FileAssetMetadataSource as FileAssetMetadataSource,
)
from fairseq2.assets.metadata_provider import (
    InMemoryAssetMetadataSource as InMemoryAssetMetadataSource,
)
from fairseq2.assets.metadata_provider import (
    PackageAssetMetadataLoader as PackageAssetMetadataLoader,
)
from fairseq2.assets.metadata_provider import (
    PackageAssetMetadataSource as PackageAssetMetadataSource,
)
from fairseq2.assets.metadata_provider import PackageFileLister as PackageFileLister
from fairseq2.assets.metadata_provider import (
    StandardFileAssetMetadataLoader as StandardFileAssetMetadataLoader,
)
from fairseq2.assets.metadata_provider import (
    StandardPackageAssetMetadataLoader as StandardPackageAssetMetadataLoader,
)
from fairseq2.assets.metadata_provider import (
    StandardPackageFileLister as StandardPackageFileLister,
)
from fairseq2.assets.metadata_provider import (
    WellKnownAssetMetadataSource as WellKnownAssetMetadataSource,
)
from fairseq2.assets.metadata_provider import (
    YamlAssetMetadataFileLoader as YamlAssetMetadataFileLoader,
)
from fairseq2.assets.metadata_provider import (
    canonicalize_asset_name as canonicalize_asset_name,
)
from fairseq2.assets.metadata_provider import (
    load_in_memory_asset_metadata as load_in_memory_asset_metadata,
)
from fairseq2.assets.metadata_provider import (
    sanitize_base_asset_name as sanitize_base_asset_name,
)
from fairseq2.assets.store import AssetEnvironmentDetector as AssetEnvironmentDetector
from fairseq2.assets.store import AssetEnvironmentResolver as AssetEnvironmentResolver
from fairseq2.assets.store import AssetNotFoundError as AssetNotFoundError
from fairseq2.assets.store import AssetStore as AssetStore
from fairseq2.assets.store import StandardAssetStore as StandardAssetStore
from fairseq2.assets.store import get_asset_store as get_asset_store
