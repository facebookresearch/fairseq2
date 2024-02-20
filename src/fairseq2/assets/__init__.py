# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.assets.card import AssetCard as AssetCard
from fairseq2.assets.card import AssetCardError as AssetCardError
from fairseq2.assets.card import (
    AssetCardFieldNotFoundError as AssetCardFieldNotFoundError,
)
from fairseq2.assets.download_manager import AssetDownloadError as AssetDownloadError
from fairseq2.assets.download_manager import (
    AssetDownloadManager as AssetDownloadManager,
)
from fairseq2.assets.download_manager import (
    InProcAssetDownloadManager as InProcAssetDownloadManager,
)
from fairseq2.assets.download_manager import (
    default_download_manager as default_download_manager,
)
from fairseq2.assets.error import AssetError as AssetError
from fairseq2.assets.metadata_provider import AssetMetadataError as AssetMetadataError
from fairseq2.assets.metadata_provider import (
    AssetMetadataProvider as AssetMetadataProvider,
)
from fairseq2.assets.metadata_provider import AssetNotFoundError as AssetNotFoundError
from fairseq2.assets.metadata_provider import (
    FileAssetMetadataProvider as FileAssetMetadataProvider,
)
from fairseq2.assets.metadata_provider import (
    InProcAssetMetadataProvider as InProcAssetMetadataProvider,
)
from fairseq2.assets.store import AssetStore as AssetStore
from fairseq2.assets.store import StandardAssetStore as StandardAssetStore
from fairseq2.assets.store import default_asset_store as default_asset_store

# For backwards-compatibility with v0.2
asset_store = default_asset_store
download_manager = default_download_manager
