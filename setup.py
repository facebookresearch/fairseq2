# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from setuptools import find_namespace_packages, setup
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess
import os

version = "0.3.0.dev0"

# If this is a local development install, allow nightly fairseq2n builds to
# take precedence.
if version.endswith(".dev0"):
    fairseq2n_version_spec = f">={version},<={version[:-5]}"
else:
    fairseq2n_version_spec = f"=={version}"

class PostInstallCommand:
    """Base class for post-installation commands."""
    def _post_install(self):
        completion_scripts = {
            'bash': '/etc/bash_completion.d/fairseq2.sh',
            'zsh': '/usr/share/zsh/site-functions/_fairseq2',
            'tcsh': '/etc/profile.d/fairseq2.csh',
        }
        for shell, path in completion_scripts.items():
            try:
                output = subprocess.check_output(['my_cli', '--print-completions', shell], text=True)
                with open(path, 'w') as f:
                    f.write(output)
                print(f"Installed completion for {shell} in {path}")
            except Exception as e:
                print(f"Could not install completion for {shell}: {e}")

class DevelopCommand(develop, PostInstallCommand):
    """Post-installation for development mode."""
    def run(self):
        print("Running develop command")
        develop.run(self)
        self._post_install()

class InstallCommand(install, PostInstallCommand):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        self._post_install()

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
        "Programming Language :: Python :: 3.9",
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
    python_requires=">=3.9",
    install_requires=[
        "editdistance~=0.8",
        "fairseq2n" + fairseq2n_version_spec,
        "importlib_metadata~=7.0",
        "importlib_resources~=6.4",
        "mypy-extensions~=1.0",
        "numpy~=1.23",
        "packaging~=23.1",
        "psutil~=5.9",
        "pyyaml~=6.0",
        "rich~=13.7",
        "sacrebleu~=2.4",
        "tiktoken~=0.7",
        "torcheval~=0.0.6",
        "tqdm~=4.62",
        "typing_extensions~=4.12;python_version<'3.10'",
    ],
    extras_require={
        "arrow": ["pyarrow>=13.0.0", "pandas~=2.0.0"],
    },
    entry_points={"console_scripts": ["fairseq2=fairseq2.recipes:main"]},
    cmdclass={
        'install': InstallCommand,
        'develop': DevelopCommand,
    },
)
