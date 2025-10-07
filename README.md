<p align="center">
  <img src="doc/source/_static/img/logo.svg" width="150"><br />
</p>

# fairseq2: FAIR Sequence Modeling Toolkit 2

[![Nightly](https://github.com/facebookresearch/fairseq2/actions/workflows/nightly.yaml/badge.svg)](https://github.com/facebookresearch/fairseq2/actions/workflows/nightly.yaml)
[![PyPI version](https://img.shields.io/pypi/v/fairseq2)](https://pypi.org/project/fairseq2/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Documentation: [Stable](https://facebookresearch.github.io/fairseq2/stable), [Nightly](https://facebookresearch.github.io/fairseq2/nightly)** | **Install: [Linux](#installing-on-linux), [macOS](#installing-on-macos), [Windows](#installing-on-windows), [From Source](INSTALL_FROM_SOURCE.md)** | **Contribute: [Guidelines](CONTRIBUTING.md)**

fairseq2 is a sequence modeling toolkit that allows researchers to train custom models for content generation tasks.

### Some of the Recent FAIR Research Based on fairseq2
* [Bridging Offline and Online Reinforcement Learning for LLMs](https://arxiv.org/pdf/2506.21495)
* [R.I.P.: Better Models by Survival of the Fittest Prompts](https://arxiv.org/abs/2501.18578)
* [Learning to reason for factuality](https://arxiv.org/abs/2508.05618)
* [Diverse Preference Optimization](https://arxiv.org/abs/2501.18101)
* [Large Concept Models: Language Modeling in a Sentence Representation Space](https://arxiv.org/abs/2412.08821)
* [Seamless: Multilingual Expressive and Streaming Speech Translation](https://arxiv.org/abs/2312.05187)

### How is fairseq2 different from the original fairseq?
fairseq2 is a start-from-scratch project that can be considered a reboot of the original [fairseq](https://github.com/facebookresearch/fairseq) to provide a clean, modular API. Notably, it differs from its predecessor in its design philosophy, moving from a monolithic framework to an extensible, much less intrusive architecture allowing researchers to independently own their project code base.

> As fairseq2 is a complete new project rather than an incremental update to the original fairseq, we intentionally avoided labeling it as fairseq version 2, reflecting its distinct and separate identity.

## Features
* First-party recipes for language model [instruction finetuning](https://facebookresearch.github.io/fairseq2/stable/tutorials/end_to_end_fine_tuning.html) and [preference optimization](https://facebookresearch.github.io/fairseq2/stable/tutorials/preference_optimization.html)
* Multi-GPU, multi-node [training](https://facebookresearch.github.io/fairseq2/stable/basics/trainer.html) using DDP, FSDP, and tensor parallelism. Supports 70B+ models.
* Native support for vLLM along with built-in sampling and beam search sequence generators
* Extensible with setuptools [extension mechanism](https://facebookresearch.github.io/fairseq2/stable/basics/runtime_extensions.html). Easily register new models, optimizers, lr schedulers, trainer units without forking/branching the library.
* Modern PyTorch tooling. Uses composability (i.e. torch.compile), PyTorch FSDP, and other relevant features
* Streaming-based, high throughput [data pipeline API](https://facebookresearch.github.io/fairseq2/stable/basics/data_pipeline.html) written in C++ with support for speech and (soon) video decoding
* Programmatic [asset cards](https://facebookresearch.github.io/fairseq2/stable/basics/assets.html) for version controlled access to models, datasets, and tokenizers
* Flexible, but deterministic configuration based on the built-in *structured* API

## Getting Started
Visit our [documentation website](https://facebookresearch.github.io/fairseq2/stable/) to learn more about fairseq2.

## Installing on Linux

### System Dependencies
fairseq2 depends on [libsndfile](https://github.com/libsndfile/libsndfile),
which can be installed via the system package manager on most Linux
distributions. For Ubuntu-based systems, run:

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

<table>
  <thead>
    <th>fairseq2</th>
    <th>PyTorch</th>
    <th>Python</th>
    <th>Variant*</th>
    <th>Arch</th>
  </thead>
  <tbody>
    <tr>
      <td rowspan=3><code>HEAD</code></td>
      <td><code>2.8.0</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu126</code>, <code>cu128</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td><code>2.7.1</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu126</code>, <code>cu128</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td><code>2.6.0</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu124</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td rowspan=3><code>0.6</code></td>
      <td><code>2.8.0</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu126</code>, <code>cu128</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td><code>2.7.1</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu126</code>, <code>cu128</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td><code>2.6.0</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu124</code></td>
      <td><code>x86_64</code></td>
    </tr>
  </tbody>
</table>

*\* cuXYZ refers to CUDA XY.Z (e.g. cu118 means CUDA 11.8)*

To install a specific combination, first follow the installation instructions on
[pytorch.org](https://pytorch.org/get-started/locally) for the desired PyTorch
version, and then use the following command (shown for PyTorch `2.8.0` and
variant `cu126`):

```sh
pip install fairseq2\
  --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu126
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
command to install the corresponding nightly package  (shown for PyTorch `2.8.0`
and variant `cu128`):

```sh
pip install fairseq2\
  --pre --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt2.8.0/cu128
```


## Installing on macOS

### System Dependencies
fairseq2 depends on [libsndfile](https://github.com/libsndfile/libsndfile),
which can be installed via Homebrew:

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

### Variants
Besides PyPI, fairseq2 also has pre-built packages available for different
PyTorch versions hosted on FAIR's package repository. The following matrix shows
the supported combinations.

<table>
  <thead>
    <th>fairseq2</th>
    <th>PyTorch</th>
    <th>Python</th>
    <th>Arch</th>
  </thead>
  <tbody>
    <tr>
      <td rowspan=2><code>HEAD</code></td>
      <td><code>2.8.0</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>arm64</code></td>
    </tr>
    <tr>
      <td><code>2.7.1</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>arm64</code></td>
    </tr>
    <tr>
      <td rowspan=2><code>0.6</code></td>
      <td><code>2.8.0</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>arm64</code></td>
    </tr>
    <tr>
      <td><code>2.7.1</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>arm64</code></td>
    </tr>
  </tbody>
</table>

To install a specific combination, first follow the installation instructions on
[pytorch.org](https://pytorch.org/get-started/locally) for the desired PyTorch
version, and then use the following command (shown for PyTorch `2.8.0`):

```sh
pip install fairseq2\
  --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cpu
```

> [!WARNING]
> fairseq2 relies on the C++ API of PyTorch which has no API/ABI compatibility
> between releases. This means **you have to install the fairseq2 variant that
> exactly matches your PyTorch version**. Otherwise, you might experience issues
> like immediate process crashes or spurious segfaults. For the same reason, if
> you upgrade your PyTorch version, you must also upgrade your fairseq2
> installation.

### Nightlies
For macOS, we also host nightly builds on FAIR's package repository. The
supported variants are identical to the ones listed in *Variants* above. Once
you have installed the desired PyTorch version, you can use the following
command to install the corresponding nightly package  (shown for PyTorch `2.8.0`):

```sh
pip install fairseq2\
  --pre --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt2.8.0/cpu
```


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
  author = {Can Balioglu and Alexander Erben and Martin Gleize and Artyom Kozhevnikov and Ilia Kulikov and Julien Yao},
  title = {fairseq2},
  url = {http://github.com/facebookresearch/fairseq2},
  year = {2023},
}
```


## License
This project is MIT licensed, as found in the [LICENSE](LICENSE) file.
