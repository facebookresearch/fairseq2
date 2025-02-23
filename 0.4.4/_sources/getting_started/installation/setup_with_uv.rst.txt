.. _fairseq2-uv-setup:

:octicon:`lock` UV Setup
========================

Overview
--------

This guide explains how to set up fairseq2 using UV, a modern Python package installer and resolver.
We will show you how to configure the `pyproject.toml` file to install **fairseq2** with various version presets.

With UV, you can:

- Create reproducible environments and avoid conflicts (CPU vs. various CUDA versions).
- Unify environment specifications in a single ``pyproject.toml``.
- Easily pick “extras” for hardware support (``--extra cpu``, ``--extra cu121``, etc.) and “groups” for extra features (``dev``, ``lint``, ``data``, ``experimental``, etc.).

Prerequisites
-------------

1. **Install UV**:

   .. code-block:: sh

      curl -LsSf https://astral.sh/uv/install.sh | sh

2. **Disable Conda base env** (if you used conda before):

   .. code-block:: sh

      conda config --set auto_activate_base false
      # Remember to restart your shell

Example Usage
-------------

We will demonstrate the usage with the following ``pyproject.toml`` file:

.. dropdown:: Example ``pyproject.toml``
    :icon: code
    :animate: fade-in

    .. code-block:: toml

         [dependency-groups]
         dev = [
            "pytest~=7.3",
         ]
         lint = [
            "mypy>=1.14.1",
            "ruff>=0.8.4",
         ]
         data = [
            "nltk>=3.9.1",
            "pyarrow>=18.1.0",
         ]
         experimental = [
            "jupyter>=1.1.1",
            "notebook>=7.3.2",
         ]

         [project.optional-dependencies]
         cpu = [
            "torch==2.2.0+cpu",
            "torchaudio==2.2.0+cpu",
            "torchvision==0.17.0+cpu",
            "fairseq2n>=0.2.0",
            "fairseq2>=0.2.0",
         ]
         cu121 = [
            "torch==2.4.0+cu121",
            "torchaudio==2.4.0+cu121",
            "torchvision==0.19.0+cu121",
            "fairseq2n>=0.2.0",
            "fairseq2>=0.2.0",
         ]
         cu124 = [
            "torch==2.5.1+cu124",
            "torchaudio==2.5.1+cu124",
            "torchvision==0.20.1+cu124",
            "fairseq2n>=0.2.0",
            "fairseq2>=0.2.0",
         ]

         [tool.uv]
         default-groups = ["dev", "lint", "data"]
         conflicts = [
            [
               { extra = "cpu" },
               { extra = "cu121" },
               { extra = "cu124" },
            ],
         ]

         [tool.uv.sources]
         torch = [
            { index = "pytorch-cpu", extra = "cpu" },
            { index = "pytorch-cu121", extra = "cu121" },
            { index = "pytorch-cu124", extra = "cu124" },
         ]
         torchaudio = [
            { index = "pytorch-cpu", extra = "cpu" },
            { index = "pytorch-cu121", extra = "cu121" },
            { index = "pytorch-cu124", extra = "cu124" },
         ]
         torchvision = [
            { index = "pytorch-cpu", extra = "cpu" },
            { index = "pytorch-cu121", extra = "cu121" },
            { index = "pytorch-cu124", extra = "cu124" },
         ]
         fairseq2n = [
            { index = "fairseq2-cpu", extra = "cpu" },
            { index = "fairseq2-cu121", extra = "cu121" },
            { index = "fairseq2-cu124", extra = "cu124" },
         ]
         fairseq2 = [
            { git = "https://github.com/facebookresearch/fairseq2", rev = "b5ae91cadf249e8061c7b6f4afbdb287c01e1624", extra = "cpu" },
            { index = "fairseq2-cu121", extra = "cu121" },
            { index = "fairseq2-cu124", extra = "cu124" },
         ]

         [[tool.uv.index]]
         name = "pytorch-cpu"
         url = "https://download.pytorch.org/whl/cpu"
         explicit = true

         [[tool.uv.index]]
         name = "pytorch-cu121"
         url = "https://download.pytorch.org/whl/cu121"
         explicit = true

         [[tool.uv.index]]
         name = "pytorch-cu124"
         url = "https://download.pytorch.org/whl/cu124"
         explicit = true

         [[tool.uv.index]]
         name = "fairseq2-cpu"
         url = "https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt2.2.0/cpu/"
         explicit = true

         [[tool.uv.index]]
         name = "fairseq2-cu121"
         url = "https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt2.4.0/cu121/"
         explicit = true

         [[tool.uv.index]]
         name = "fairseq2-cu124"
         url = "https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt2.5.1/cu124/"
         explicit = true

    .. note::

         The ``pyproject.toml`` file is a simplified example. You can customize it to fit your needs.
         We will explain the structure of the ``pyproject.toml`` file in the next section.

In the ``pyproject.toml`` file, we have defined the following conflicting extras:

- ``cpu``: CPU-only PyTorch and fairseq2
- ``cu121``: CUDA 12.1 PyTorch and fairseq2
- ``cu124``: CUDA 12.4 PyTorch and fairseq2

We have also defined the following groups:

- ``dev``: Development tools
- ``lint``: Linting tools
- ``data``: Data processing tools
- ``experimental``: Experimental features

For each extra, we have defined the following index sources to pull the corresponding PyTorch and fairseq2 packages:

- ``pytorch-cpu``: CPU-only PyTorch
- ``pytorch-cu121``: CUDA 12.1 PyTorch
- ``pytorch-cu124``: CUDA 12.4 PyTorch
- ``fairseq2-cpu``: CPU-only fairseq2 (pinned to a specific commit in fairseq2)
- ``fairseq2-cu121``: CUDA 12.1 fairseq2
- ``fairseq2-cu124``: CUDA 12.4 fairseq2

Below are a few common workflows. Adjust the commands to suit your environment and cluster:

.. code-block:: sh

   # (1) Sync with CPU extra, which is handy for ci pipelines
   uv sync --extra cpu

   # (2) Lint checks
   uvx ruff check .
   uvx ruff format --check .
   uv run --extra cpu --with=types-PyYAML mypy

   # (3) Run tests
   uv run --extra cpu pytest -rP --verbose

Switch CUDA versions easily:

.. code-block:: sh

   # Switch to CUDA 12.4
   uv sync --extra cu124

   # Run fairseq2 from the new environment
   uv run fairseq2 assets list

   # Or activate .venv
   source .venv/bin/activate
   fairseq2 assets list

For local development or "editable installs" of **fairseq2**:

.. code-block:: sh

   # Once the environment is set up, install fairseq2 in editable mode:
   source .venv/bin/activate
   uv pip install -e /path/to/fairseq2


Key Concepts
~~~~~~~~~~~~

- **Extras**  
  Defined in ``[project.optional-dependencies]``. They are typically `mutually exclusive` variants (`e.g.`, CPU vs. CUDA).

- **Groups**  
  Listed in ``[dependency-groups]`` and combined in ``default-groups`` under ``[tool.uv]``. They are `additive` sets of packages 
  (`e.g.`, dev tools, lint, data).

- **Conflicts**  
  Prevent mixing hardware extras that don’t make sense together (`e.g.`, CPU + CUDA).

- **Index Sources**  
  Let you specify custom or nightly wheels in ``[tool.uv.sources]``. For each extra, UV pulls from the correct index or pinned Git revision.


----

That’s all! With UV, setting up **fairseq2** can be as simple as one line of code. 
Enjoy faster environment creation, conflict-free installs, and effortless reproducibility.
