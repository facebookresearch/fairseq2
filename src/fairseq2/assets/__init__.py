# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.assets._card import AssetCard as AssetCard
from fairseq2.assets._card import AssetCardError as AssetCardError
from fairseq2.assets._card import (
    AssetCardFieldNotFoundError as AssetCardFieldNotFoundError,
)
from fairseq2.assets._card import AssetCardNotFoundError as AssetCardNotFoundError
from fairseq2.assets._dirs import AssetDirectories as AssetDirectories
from fairseq2.assets._download_manager import AssetDownloadError as AssetDownloadError
from fairseq2.assets._download_manager import (
    AssetDownloadManager as AssetDownloadManager,
)
from fairseq2.assets._download_manager import (
    InProcAssetDownloadManager as InProcAssetDownloadManager,
)
from fairseq2.assets._metadata_provider import (
    AssetMetadataFileLoader as AssetMetadataFileLoader,
)
from fairseq2.assets._metadata_provider import (
    AssetMetadataLoadError as AssetMetadataLoadError,
)
from fairseq2.assets._metadata_provider import (
    AssetMetadataNotFoundError as AssetMetadataNotFoundError,
)
from fairseq2.assets._metadata_provider import (
    AssetMetadataProvider as AssetMetadataProvider,
)
from fairseq2.assets._metadata_provider import (
    AssetMetadataSaveError as AssetMetadataSaveError,
)
from fairseq2.assets._metadata_provider import (
    CachedAssetMetadataProvider as CachedAssetMetadataProvider,
)
from fairseq2.assets._metadata_provider import (
    FileAssetMetadataLoader as FileAssetMetadataLoader,
)
from fairseq2.assets._metadata_provider import (
    InProcAssetMetadataLoader as InProcAssetMetadataLoader,
)
from fairseq2.assets._metadata_provider import (
    PackageAssetMetadataLoader as PackageAssetMetadataLoader,
)
from fairseq2.assets._metadata_provider import PackageFileLister as PackageFileLister
from fairseq2.assets._metadata_provider import (
    StandardAssetMetadataFileLoader as StandardAssetMetadataFileLoader,
)
from fairseq2.assets._metadata_provider import (
    WheelPackageFileLister as WheelPackageFileLister,
)
from fairseq2.assets._store import AssetLookupScope as AssetLookupScope
from fairseq2.assets._store import AssetStore as AssetStore
from fairseq2.assets._store import EnvironmentResolver as EnvironmentResolver
from fairseq2.assets._store import StandardAssetStore as StandardAssetStore
