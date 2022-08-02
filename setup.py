# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup


def read_long_description() -> str:
    with open("README.md") as f:
        return f.read()


def main() -> None:
    setup(
        name="fairseq2",
        version="2.0.0.dev0",
        description="FAIR Sequence Modeling Toolkit",
        long_description=read_long_description(),
        long_description_content_type="text/markdown",
        url="https://github.com/facebookresearch/fairseq",
        license="MIT",
        author="Fundamental AI Research (FAIR) at Meta",
        keywords=["machine learning"],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        packages=["fairseq2"],
        package_data={"fairseq2": ["py.typed", "*.pyi"]},
        zip_safe=False,
        python_requires=">=3.7",
        install_requires=[
            "torch>=1.11.0",
            "typing_extensions>=4.3.0",
        ],
    )


if __name__ == "__main__":
    main()
