> **_NOTE:_**  fairseq v2 is in heavy development and at this time should not be
> used for anything beyond testing and feedback. Note also that this is a
> temporary repository and the final code will be hosted in the official fairseq
> repository.

<p align="center">
  <img src="docs/src/static/img/fairseq_logo.png" width="150">
</p>

--------------------------------------------------------------------------------

[**Installation**](#installation) | [**Getting Started**](#getting-started) | [**Documentation**](#documentation)

fairseq is a sequence modeling toolkit that allows researchers and developers to
train custom models for translation, summarization, language modeling and other
content generation tasks.

## Dependencies
fairseq versions corresponding to each PyTorch release:

| `fairseq`    | `torch`     | `python`          |
| ------------ | ----------- | ----------------- |
| `main`       | `>=1.11.0`  | `>=3.8`, `<=3.10` |

## Installation
Only Linux and macOS operating systems are supported. Please note though that
pre-built Conda and PyPI packages are *only* available for Linux. For
installation on macOS you can follow the instructions in the
[From Source](#from-source) section. At this time there are no plans to
introduce Windows support.

### Conda
Conda is the recommended way to install fairseq. Running the following command
in a Conda environment will install fairseq and all its dependencies.

**Stable**

For PyTorch CPU:
```
conda install -c pytorch -c conda-forge -c fairseq <TBD> cpuonly
```

For PyTorch with CUDA 10.2:
```
conda install -c pytorch -c conda-forge -c fairseq <TBD> cudatoolkit=10.2
```

For PyTorch with CUDA 11.3:
```
conda install -c pytorch -c conda-forge -c fairseq <TBD> cudatoolkit=11.3
```

For PyTorch with CUDA 11.6:
```
conda install -c pytorch -c conda-forge -c fairseq <TBD> cudatoolkit=11.6
```

**Nightly**

For PyTorch CPU
```
conda install -c pytorch -c conda-forge -c fairseq-nightly <TBD> cpuonly
```

For PyTorch with CUDA 10.2
```
conda install -c pytorch -c conda-forge -c fairseq-nightly <TBD> cudatoolkit=10.2
```

For PyTorch with CUDA 11.3
```
conda install -c pytorch -c conda-forge -c fairseq-nightly <TBD> cudatoolkit=11.3
```

For PyTorch with CUDA 11.6
```
conda install -c pytorch -c conda-forge -c fairseq-nightly <TBD> cudatoolkit=11.6
```
### PyPI

**Stable**

For PyTorch CPU:
```
pip install <TBD> --extra-index-url https://download.pytorch.org/whl/cpu
```

For PyTorch with CUDA 10.2:
```
pip install <TBD> --extra-index-url https://download.pytorch.org/whl/cu102
```

For PyTorch with CUDA 11.3:
```
pip install <TBD> --extra-index-url https://download.pytorch.org/whl/cu113
```

For PyTorch with CUDA 11.6:
```
pip install <TBD> --extra-index-url https://download.pytorch.org/whl/cu116
```

**Nightly**

For PyTorch CPU:
```
pip install <TBD> --pre --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

For PyTorch with CUDA 10.2:
```
pip install <TBD> --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu102
```

For PyTorch with CUDA 11.3:
```
pip install <TBD> --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu113
```

For PyTorch with CUDA 11.6:
```
pip install <TBD> --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116
```

### From Source

#### Prerequisites
- After cloning the repository make sure to initialize all submodules by
  executing `git submodule update --init --recursive`.
- Create a Python virtual environment and install the build dependencies:
 ```
# Build against PyTorch CPU
pip install --upgrade -r requirements.txt -r use-cpu.txt

# Build against PyTorch with CUDA 10.2
pip install --upgrade -r requirements.txt -r use-cu102.txt

# Build against PyTorch with CUDA 11.3
pip install --upgrade -r requirements.txt -r use-cu113.txt

# Build against PyTorch with CUDA 11.6
pip install --upgrade -r requirements.txt -r use-cu116.txt
```
- The build process requires CMake 3.21 or later. You can install an up-to-date
  version by executing `pip install cmake`. For other environments please refer
  to your package manager or [cmake.org](https://cmake.org/download/).

Once you have all prerequisites run the following command to install the fairseq
Python package:

```
pip install .
```

#### Development
In case you would like to contribute to the project you can slightly modify the
command listed above:

```
pip install -e .
```

The project also comes with a [requirements-devel.txt](./requirements-devel.txt)
to set up a Python virtual environment for development.

```
# Build against PyTorch CPU
pip install --upgrade -r requirements-devel.txt -r use-cpu.txt

# Build against PyTorch with CUDA 10.2
pip install --upgrade -r requirements-devel.txt -r use-cu102.txt

# Build against PyTorch with CUDA 11.3
pip install --upgrade -r requirements-devel.txt -r use-cu113.txt

# Build against PyTorch with CUDA 11.6
pip install --upgrade -r requirements-devel.txt -r use-cu116.txt
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
