<p align="center">
  <img src="doc/static/img/logo.png" width="150"><br />
</p>

# fairseq2: FAIR Sequence Modeling Toolkit 2

[![Nightly](https://github.com/facebookresearch/fairseq2/actions/workflows/nightly.yaml/badge.svg)](https://github.com/facebookresearch/fairseq2/actions/workflows/nightly.yaml)
[![PyPI version](https://img.shields.io/pypi/v/fairseq2)](https://pypi.org/project/fairseq2/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Documentation: [Stable](https://facebookresearch.github.io/fairseq2/stable), [Nightly](https://facebookresearch.github.io/fairseq2/nightly)** | **Install: [Linux](#installing-on-linux), [macOS](#installing-on-macos), [Windows](#installing-on-windows), [From Source](INSTALL_FROM_SOURCE.md)** | **Contribute: [Guidelines](CONTRIBUTING.md)**

fairseq2 is a sequence modeling toolkit that allows researchers and developers
to train custom models for translation, summarization, language modeling, and
other content generation tasks. It is also the successor of
[fairseq](https://github.com/facebookresearch/fairseq).


## Getting Started
You can find our full documentation including tutorials and API reference
[here](https://facebookresearch.github.io/fairseq2/stable).

For recent changes, you can check out our [changelog](CHANGELOG.md).


## Models
As of today, the following models are available in fairseq2:

 * [LLaMA](recipes/llama)
 * [LLaMA 2](recipes/llama)
 * [Mistral 7B](recipes/mistral)
 * [NLLB-200](src/fairseq2/models/nllb)
 * [S2T Transformer + Conformer](src/fairseq2/models/s2t_transformer)
 * [w2v-BERT](src/fairseq2/models/w2vbert)
 * [wav2vec 2.0](src/fairseq2/models/wav2vec2)

fairseq2 is also used by various external projects such as:

 * [Seamless Communication](https://github.com/facebookresearch/seamless_communication)
 * [SONAR](https://github.com/facebookresearch/SONAR)


## Installing on Linux

### System Dependencies
fairseq2 has a dependency on
[libsndfile](https://github.com/libsndfile/libsndfile) that can be installed via
the system package manager on most Linux distributions. For Ubuntu-based
systems, run:

```sh
sudo apt install libsndfile1
```

Similarly, on Fedora, run:

```sh
sudo dnf install libsndfile
```

For other Linux distributions, please consult its documentation on how to
install packages.

### pip
To install fairseq2 on Linux x86-64, run:

```sh
pip install fairseq2
```

This command will install a version of fairseq2 that is compatible with PyTorch
hosted on PyPI.

At this time, we do not offer a pre-built package for ARM-based systems such as
Raspberry PI or NVIDIA Jetson. Please refer to
[Install From Source](INSTALL_FROM_SOURCE.md) to learn how to build and install
fairseq2 on those systems.

### Variants
Besides PyPI, fairseq2 also has pre-built packages available for different
PyTorch and CUDA versions hosted on FAIR's package repository. The following
matrix shows the supported combinations.

| PyTorch          | Python            | Variant*               | Arch     |
| ---------------- | ----------------- | ---------------------- | -------- |
| `2.1.0`, `2.1.1` | `>=3.8`, `<=3.11` | `cpu`, `cu118` `cu121` | `x86_64` |
| `2.0.0`, `2.0.1` | `>=3.8`, `<=3.11` | `cpu`, `cu117` `cu118` | `x86_64` |
| `1.13.1`         | `>=3.8`, `<=3.10` | `cpu`, `cu116`         | `x86_64` |

*\* cuXYZ refers to CUDA XY.Z (e.g. cu118 means CUDA 11.8)*

To install a specific combination, first follow the installation instructions on
[pytorch.org](https://pytorch.org) for the desired PyTorch version, and then use
the following command (shown for PyTorch `2.1.1` and variant `cu118`):

```sh
pip install fairseq2\
  --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.1.1/cu118
```


> [!WARNING]
> fairseq2 relies on the C++ API of PyTorch which has no API/ABI compatibility
> between releases. This means **you have to install the fairseq2 variant that
> exactly matches your PyTorch version**. Otherwise, you might experience issues
> like immediate process crashes or spurious segfaults. For the same reason, if
> you upgrade your PyTorch version, you must also upgrade your fairseq2
> installation.

### Nightlies
For Linux, we also host nightly builds on FAIR's package repository. The
supported variants are identical to the ones listed in *Variants* above. Once
you have installed the desired PyTorch version, you can use the following
command to install the corresponding nightly package  (shown for PyTorch `2.1.1`
and variant `cu118`):

```sh
pip install fairseq2\
  --pre --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt2.1.1/cu118
```


## Installing on macOS

### System Dependencies
fairseq2 has a dependency on
[libsndfile](https://github.com/libsndfile/libsndfile) that can be installed via
Homebrew:

```sh
brew install libsndfile
```

### pip
To install fairseq2 on ARM64-based (i.e. Apple silicon) Mac computers, run:

```sh
pip install fairseq2
```

This command will install a version of fairseq2 that is compatible with PyTorch
hosted on PyPI.

At this time, we do not offer a pre-built package for Intel-based Mac computers.
Please refer to [Install From Source](INSTALL_FROM_SOURCE.md) to learn how to
build and install fairseq2 on Intel machines.


## Installing on Windows
fairseq2 does not have native support for Windows and there are no plans to
support it in the foreseeable future. However, you can use fairseq2 via the
[Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/about)
(a.k.a. WSL) along with full CUDA support introduced in WSL 2. Please follow the
instructions in the [Installing on Linux](#installing-on-linux) section for a
WSL-based installation.


## Installing from Source
See [here](INSTALL_FROM_SOURCE.md).


## Contributing
We always welcome contributions to fairseq2! Please refer to
[Contribution Guidelines](CONTRIBUTING.md) to learn how to format, test, and
submit your work.


## Citing fairseq2
If you use fairseq2 in your research and wish to refer to it, please use the
following BibTeX entry.

```
@software{balioglu2023fairseq2,
  author = {Can Balioglu},
  title = {fairseq2},
  url = {http://github.com/facebookresearch/fairseq2},
  year = {2023},
}
```


## License
This project is MIT licensed, as found in the [LICENSE](LICENSE) file.
