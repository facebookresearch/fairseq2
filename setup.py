# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

version = "0.3.0.dev0"

# If this is a local development install, allow nightly fairseq2n builds to
# take precedence.
if version.endswith(".dev0"):
    fairseq2n_version_spec = f">={version},<={version[:-5]}"
else:
    fairseq2n_version_spec = f"=={version}"

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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"fairseq2": ["py.typed", "assets/cards/*.yaml"]},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "fairseq2n" + fairseq2n_version_spec,
        "numpy~=1.23",
        "overrides~=7.3",
        "packaging~=23.1",
        "psutil~=5.9",
        "pyyaml~=6.0",
        "torcheval~=0.0.6",
        "tqdm~=4.62",
        "typing_extensions~=4.3;python_version<'3.10'",
    ],
    extras_require={
        "arrow": ["pyarrow>=13.0.0", "pandas~=2.0.0"],
    },
)
