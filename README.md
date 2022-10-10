<p align="center">
  <img src="doc/static/img/fairseq2_logo.png" width="150">
</p>

--------------------------------------------------------------------------------

[**Installation**](#installation) | [**Getting Started**](#getting-started) | [**Documentation**](#documentation)

fairseq2 is a sequence modeling toolkit that allows researchers and developers
to train custom models for translation, summarization, language modeling, and
other content generation tasks.

## Dependencies
fairseq2 versions corresponding to each PyTorch release:

| `fairseq2`   | `torch`     | `python`          |
| ------------ | ----------- | ----------------- |
| `main`       | `>=1.13.0`  | `>=3.8`, `<=3.10` |

## Installation
fairseq2 supports Linux and macOS operating systems. Please note though that
pre-built Conda and PyPI packages are *only* available for Linux. For
installation on macOS you can follow the instructions in the
[From Source](#from-source) section. At this time there are no plans to
introduce Windows support.

### Conda
Conda is the recommended way to install fairseq2. First follow the instructions
[here](https://pytorch.org/get-started/locally/) to install a supported version
of PyTorch in a Conda environment (see the compatibility matrix above); then run
the following command to install fairseq2.

**Stable**
```
conda install -c conda-forge -c <TBD> fairseq2
```

**Nightly**
```
conda install -c conda-forge -c <TBD> fairseq2
```

### PyPI

**Stable**
```
pip install fairseq2 --extra-index-url <TBD>
```

**Nightly**
```
pip install fairseq2 --extra-index-url <TBD> --pre
```

### From Source

#### Prerequisites
1. After cloning the repository make sure to initialize all submodules by
   executing `git submodule update --init --recursive`.
2. The build process requires CMake 3.21 or later. Please refer to your package
   manager or to [cmake.org](https://cmake.org/download/) on how to install
   CMake.
3. (Optional, but strongly recommended) Create a Python virtual environment.
4. Install a version of PyTorch that is supported by fairseq2 (see the
   compatibility matrix above).

Once you have completed all prerequisites run the following commands to install
fairseq2:

```
cmake -B build
cmake --build build
pip install .
```

#### Development
In case you would like to contribute to the project you can slightly modify the
`pip` command listed above:

```
cmake -B build
cmake --build build
pip install -e .
```

With `pip install -e .` you enable edit mode (a.k.a. develop mode) that allows
you to modify Python files in-place.

If you are working in C++, whenever you modify a header or implementation file,
executing `cmake --build build` alone is sufficient. You do not have to execute
`pip install` again.

The project also comes with a [requirements-devel.txt](./requirements-devel.txt)
to set up a Python virtual environment for development.

```
pip install --upgrade -r requirements-devel.txt
```

#### Tip
Note that, if you plan to work in C++, using the Ninja build system and the
ccache tool can significatly speed up your build times. To use them you can
replace the initial CMake command listed above with the following version:

```
cmake -GNinja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build
```

## Getting Started
TBD

## Documentation
For more documentation, see [our docs website](https://pytorch.org/).

## Contributing
Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md).

## License
This project is MIT licensed, as found in the [LICENSE](LICENSE) file. The
license applies to the pre-trained models as well.
