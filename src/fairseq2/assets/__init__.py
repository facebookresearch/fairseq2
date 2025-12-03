# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.assets.card import AssetCard as AssetCard
from fairseq2.assets.card import AssetCardFieldError as AssetCardFieldError
from fairseq2.assets.card import AssetCardFieldFormatError as AssetCardFieldFormatError
from fairseq2.assets.card import (
    AssetCardFieldNotFoundError as AssetCardFieldNotFoundError,
)
from fairseq2.assets.card import AssetCardFieldTypeError as AssetCardFieldTypeError
from fairseq2.assets.card import AssetConfigLoader as AssetConfigLoader
from fairseq2.assets.card import StandardAssetConfigLoader as StandardAssetConfigLoader
from fairseq2.assets.dirs import AssetPathVariableError as AssetPathVariableError
from fairseq2.assets.dirs import _AssetDirectoryAccessor as _AssetDirectoryAccessor
from fairseq2.assets.dirs import (
    _StandardAssetDirectoryAccessor as _StandardAssetDirectoryAccessor,
)
from fairseq2.assets.download_manager import AssetDownloadError as AssetDownloadError
from fairseq2.assets.download_manager import (
    AssetDownloadManager as AssetDownloadManager,
)
from fairseq2.assets.download_manager import (
    AssetDownloadNetworkError as AssetDownloadNetworkError,
)
from fairseq2.assets.download_manager import AssetNotFoundError as AssetNotFoundError
from fairseq2.assets.download_manager import (
    CorruptAssetCacheError as CorruptAssetCacheError,
)
from fairseq2.assets.download_manager import CorruptAssetError as CorruptAssetError
from fairseq2.assets.download_manager import (
    _DelegatingAssetDownloadManager as _DelegatingAssetDownloadManager,
)
from fairseq2.assets.download_manager import _HuggingFaceHub as _HuggingFaceHub
from fairseq2.assets.download_manager import (
    _LocalAssetDownloadManager as _LocalAssetDownloadManager,
)
from fairseq2.assets.download_manager import (
    _StandardAssetDownloadManager as _StandardAssetDownloadManager,
)
from fairseq2.assets.download_manager import (
    get_asset_download_manager as get_asset_download_manager,
)
from fairseq2.assets.metadata_provider import AssetMetadataError as AssetMetadataError
from fairseq2.assets.metadata_provider import (
    AssetMetadataProvider as AssetMetadataProvider,
)
from fairseq2.assets.metadata_provider import AssetMetadataSource as AssetMetadataSource
from fairseq2.assets.metadata_provider import (
    AssetMetadataSourceNotFoundError as AssetMetadataSourceNotFoundError,
)
from fairseq2.assets.metadata_provider import (
    BadAssetMetadataError as BadAssetMetadataError,
)
from fairseq2.assets.metadata_provider import (
    BadAssetMetadataFileError as BadAssetMetadataFileError,
)
from fairseq2.assets.metadata_provider import (
    CachedAssetMetadataProvider as CachedAssetMetadataProvider,
)
from fairseq2.assets.metadata_provider import (
    FileAssetMetadataLoader as FileAssetMetadataLoader,
)
from fairseq2.assets.metadata_provider import (
    _AssetMetadataFileLoader as _AssetMetadataFileLoader,
)
from fairseq2.assets.metadata_provider import (
    _FileAssetMetadataSource as _FileAssetMetadataSource,
)
from fairseq2.assets.metadata_provider import (
    _InMemoryAssetMetadataSource as _InMemoryAssetMetadataSource,
)
from fairseq2.assets.metadata_provider import (
    _load_in_memory_asset_metadata as _load_in_memory_asset_metadata,
)
from fairseq2.assets.metadata_provider import (
    _PackageAssetMetadataLoader as _PackageAssetMetadataLoader,
)
from fairseq2.assets.metadata_provider import (
    _PackageAssetMetadataSource as _PackageAssetMetadataSource,
)
from fairseq2.assets.metadata_provider import _PackageFileLister as _PackageFileLister
from fairseq2.assets.metadata_provider import (
    _StandardFileAssetMetadataLoader as _StandardFileAssetMetadataLoader,
)
from fairseq2.assets.metadata_provider import (
    _StandardPackageAssetMetadataLoader as _StandardPackageAssetMetadataLoader,
)
from fairseq2.assets.metadata_provider import (
    _StandardPackageFileLister as _StandardPackageFileLister,
)
from fairseq2.assets.metadata_provider import (
    _WellKnownAssetMetadataSource as _WellKnownAssetMetadataSource,
)
from fairseq2.assets.metadata_provider import (
    _YamlAssetMetadataFileLoader as _YamlAssetMetadataFileLoader,
)
from fairseq2.assets.metadata_provider import (
    canonicalize_asset_name as canonicalize_asset_name,
)
from fairseq2.assets.metadata_provider import (
    sanitize_base_asset_name as sanitize_base_asset_name,
)
from fairseq2.assets.store import AssetCardNotFoundError as AssetCardNotFoundError
from fairseq2.assets.store import AssetEnvironmentResolver as AssetEnvironmentResolver
from fairseq2.assets.store import AssetStore as AssetStore
from fairseq2.assets.store import AssetStoreError as AssetStoreError
from fairseq2.assets.store import (
    BaseAssetCardNotFoundError as BaseAssetCardNotFoundError,
)
from fairseq2.assets.store import _AssetEnvironmentDetector as _AssetEnvironmentDetector
from fairseq2.assets.store import _StandardAssetStore as _StandardAssetStore
from fairseq2.assets.store import get_asset_store as get_asset_store
