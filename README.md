<p align="center">
  <img src="doc/static/img/logo.png" width="150"><br />
</p>

[![Release](https://github.com/fairinternal/fairseq2/actions/workflows/release.yaml/badge.svg)](https://github.com/fairinternal/fairseq2/actions/workflows/release.yaml)

--------------------------------------------------------------------------------

[**Getting Started**](#getting-started) | **Installing From: [Conda](#installing-from-conda), [PyPI](#installing-from-pypi), [Source](#installing-from-source)** | [**Contributing**](#contributing) | [**License**](#license)

fairseq2 is a sequence modeling toolkit that allows researchers and developers
to train custom models for translation, summarization, language modeling, and
other content generation tasks.

## Getting Started
You can find our full documentation including tutorials and API reference
[here](https://fairinternal.github.io/fairseq2/nightly)
([nightly](https://fairinternal.github.io/fairseq2/nightly)).

For recent changes, you can check out our [changelog](CHANGELOG.md).

fairseq2 mainly supports Linux. There is partial support for macOS with limited
feature set. Windows is not supported.

## Installing From Conda
coming soon...

## Installing From PyPI
coming soon...

## Installing From Source

By default `make install` will install pytorch and fairseq2 in a new venv.
You'll need to activate it with `. ./venv/bin/activate`.

If you want to reuse an existing venv, you can use `make install VENV=/path/to/venv/bin`.
If you have a conda environment activated, `make install` will install fairseq2 inside it.
If you provide a venv or a conda env,
`make install` will assume you've already installed pytorch inside, and won't try to upgrade it.


For manual/advanced installation steps and detailed instructions please refer to https://fairinternal.github.io/fairseq2/nightly/intro/install.html

To make sure that your installation has no issues, you can run the test suite: `make tests`

By default, the tests will be run on CPU; optionally pass the `--device` (short
form `-d`) argument to run them on a specific device (e.g. NVIDIA GPU).

```
pytest --device cuda:1
```

## Contributing
We always welcome contributions to fairseq2! Please refer to our
[contribution guidelines](./CONTRIBUTING.md) to learn how to format, test, and
submit your work.

## License
This project is MIT licensed, as found in the [LICENSE](LICENSE) file. The
license applies to the pre-trained models as well.

