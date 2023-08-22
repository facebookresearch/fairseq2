<p align="center">
  <img src="doc/static/img/logo.png" width="150"><br />
</p>

# fairseq2: FAIR Sequence Modeling Toolkit 2

[![Nightly](https://github.com/facebookresearch/fairseq2/actions/workflows/nightly.yaml/badge.svg)](https://github.com/facebookresearch/fairseq2/actions/workflows/nightly.yaml)
[![PyPI version](https://img.shields.io/pypi/v/fairseq2)](https://pypi.org/project/fairseq2/)

[**Documentation**](https://facebookresearch.github.io/fairseq2/stable) ([**Nightly**](https://facebookresearch.github.io/fairseq2/nightly)) | **Install From: [PyPI](#install-from-pypi), [Source](#install-from-source)**

> ❗fairseq2 is still under heavy development (early beta quality). Please use with caution and do not hesitate to share feedback with us!

fairseq2 is a sequence modeling toolkit that allows researchers and developers
to train custom models for translation, summarization, language modeling, and
other content generation tasks. It is also the successor of [fairseq](https://github.com/facebookresearch/fairseq).


## Getting Started
You can find our full documentation including tutorials and API reference
[here](https://facebookresearch.github.io/fairseq2/stable)
([nightly](https://facebookresearch.github.io/fairseq2/nightly)).

For recent changes, you can check out our [changelog](CHANGELOG.md).

Note that fairseq2 mainly supports Linux. There is partial support for macOS with limited
feature set (i.e. no CUDA). Windows support is not planned.


## Pre-trained Models and Examples
As of today, the following projects use fairseq2:

- [SeamlessM4T](https://github.com/facebookresearch/seamless_communication): A state-of-the-art, all-in-one, multimodel translation model
- [SONAR](https://github.com/facebookresearch/SONAR): A multilingual and multimodal fixed-size sentence embedding space, with a full suite of speech and text encoders and decoders

In the following 0.x releases, we will gradually add examples and recipes for training, fine-tuning, and evaluation of different model architectures.


## Install From PyPI

> As of today, we only provide pre-built wheels for Linux x86-64. For installation on macOS, please
> follow the instructions at "Install from Source".

The easiest way to start with fairseq2 is to install it via pip. Before you proceed, make sure that you
have an up-to-date version of pip.

```sh
pip install --upgrade pip
```

And, use the following command:

```sh
pip install fairseq2
```

This will install a version of fairseq2 that is compatible with the latest PyTorch version hosted on PyPI.


## Install From Source
fairseq2 consists of two packages; the user-facing fairseq2 package implemented in pure Python, and fairseq2n that contains
the C++ and CUDA pieces of the library. If you are interested in Python parts only, you can use the following
instructions. For C++/CUDA development, please follow the instructions
[here](https://facebookresearch.github.io/fairseq2/stable/installation/from_source).

First, clone the fairseq2 repository to your machine:

```sh
git clone https://github.com/facebookresearch/fairseq2.git

cd fairseq2
```

And, install the fairseq2 package from the repo directory:

```sh
pip install .
```

If you are interested in editing Python code and/or contributing to fairseq2, you can instead use the editable mode:

```sh
pip install -r requirements-devel.txt

pip install -e .
```

## Contributing
We always welcome contributions to fairseq2! Please refer to our
[contribution guidelines](./CONTRIBUTING.md) to learn how to format, test, and
submit your work.


## License
This project is MIT licensed, as found in the [LICENSE](LICENSE) file.


## Legal
[Terms of Use](https://opensource.fb.com/legal/terms), [Privacy Policy](https://opensource.fb.com/legal/privacy)

Copyright © Meta Platforms, Inc
