# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from setuptools import find_namespace_packages, setup

version = "0.8.0.dev0"

# If this is a local development install, allow nightly fairseq2n builds to
# take precedence.
if version.endswith(".dev0"):
    fairseq2n_version_spec = f">={version},<={version[:-5]}"
else:
    p = version.split("+", maxsplit=1)

    fairseq2n_version_spec = "==" + p[0]


setup(
    name="fairseq2",
    version=version,
    description="FAIR Sequence Modeling Toolkit",
    long_description="https://github.com/facebookresearch/fairseq2",
    long_description_content_type="text/plain",
    url="https://github.com/facebookresearch/fairseq2",
    license="MIT",
    author="Fundamental AI Research (FAIR) at Meta",
    keywords=["machine learning"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    package_data={
        "fairseq2": ["py.typed"],
        "fairseq2.assets.cards": ["**/*.yaml"],
    },
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=[
        "clusterscope~=0.0.31",
        "editdistance~=0.8",
        "fairseq2n" + fairseq2n_version_spec,
        "huggingface_hub~=0.32",
        "importlib_metadata~=7.0",
        "importlib_resources~=6.4",
        "mypy-extensions~=1.0",
        "numpy~=1.23",
        "packaging~=24.1",
        "psutil~=5.9",
        "ruamel.yaml~=0.18",
        "rich~=13.7",
        "sacrebleu~=2.4",
        "safetensors~=0.6",
        "tiktoken~=0.7",
        "torcheval~=0.0.6",
        "typing_extensions~=4.12",
        # This dependency is required for tiktoken.load.read_file, but it's
        # listed as optional in tiktoken's pyproject.toml
        # (https://github.com/openai/tiktoken/blob/main/pyproject.toml#L9)
        "blobfile~=3.0.0",
    ],
    extras_require={
        "arrow": [
            "pyarrow>=17.0.0",
            "retrying~=1.3.4",
            "polars>=1.19.0",
            "xxhash~=3.5",
        ],
    },
)
