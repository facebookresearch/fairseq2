.. _fairseq2-uv-setup:

:octicon:`lock` UV Setup
========================

Overview
--------

This guide explains how to set up fairseq2 using `uv <https://docs.astral.sh/uv/>`_, a modern Python package installer and resolver that provides fast, reliable dependency management.

With uv, you can:

- Create reproducible environments with conflict resolution
- Easily switch between hardware variants (CPU, CUDA 12.8, etc.)
- Manage development dependencies with dependency groups
- Ensure version compatibility between PyTorch and fairseq2

Prerequisites
-------------

1. **Install uv**:

    .. code-block:: sh

        curl -LsSf https://astral.sh/uv/install.sh | sh

2. **Disable Conda base environment** (if using Conda):

    .. code-block:: sh

        conda config --set auto_activate_base false
        # Restart your shell after this change

Quick Start
-----------

Add fairseq2 to your project with:

.. code-block:: sh

    uv add fairseq2

A Detailed Project Setup
------------------------

1. **Create a project with fairseq2**:

    .. code-block:: sh

        # Create new project
        mkdir my-fairseq2-project
        cd my-fairseq2-project

        # Initialize with pyproject.toml (see configuration below)
        uv init

2. **Install fairseq2 with CUDA 12.8**:

    .. code-block:: sh

        uv sync --extra cu128

3. **Install fairseq2 with CPU-only**:

   .. code-block:: sh

       uv sync --extra cpu

4. **Run fairseq2 commands**:

   .. code-block:: sh

       uv run python -m fairseq2.assets list --kind model

Configuration
-------------

Add this ``pyproject.toml`` configuration to your project:

.. dropdown:: Complete ``pyproject.toml`` example
    :icon: code
    :animate: fade-in

    .. code-block:: toml

        [project]
        name = "my-fairseq2-project"
        version = "0.1.0"
        requires-python = ">=3.10"
        dependencies = [
            "clusterscope>=0.0.18",
            "pip>=25.2",
            "tensorboard~=2.16",
            "vllm>=0.10.0",
        ]

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
        doc = [
            "sphinx~=7.4.0",
            "sphinxcontrib-bibtex~=2.5.0",
            "sphinx-favicon~=1.0.1",
            "sphinx-design~=0.5.0",
            "myst-parser~=4.0.0",
            "sphinxcontrib-mermaid~=1.0.0",
            "furo==2024.8.6",
            "nbsphinx~=0.9.6",
        ]

        [project.optional-dependencies]
        cpu = [
            "torch==2.7.1+cpu",
            "torchaudio==2.7.1+cpu",
            "torchvision==0.22.1+cpu",
            "fairseq2n==0.5.*",
            "fairseq2==0.5.*",
            "vllm==0.10.1",
        ]
        cu128 = [
            "torch==2.7.1+cu128",
            "torchaudio==2.7.1+cu128",
            "torchvision==0.22.1+cu128",
            "fairseq2n==0.5.*",
            "fairseq2==0.5.*",
            "vllm==0.10.1",
        ]
        experimental = [
            "jupyter>=1.1.1",
            "notebook>=7.3.2",
            "torch==2.7.1+cu128",
            "torchaudio==2.7.1+cu128",
            "torchvision==0.22.1+cu128",
            "fairseq2n==0.5.*",
            "fairseq2==0.5.*",
            "vllm==0.10.1",
        ]
        v04-cu124 = [
            "torch==2.6.0+cu124",
            "torchaudio==2.6.0+cu124",
            "torchvision==0.21.0+cu124",
            "fairseq2n==0.4.5",
            "fairseq2==0.4.5",
            "vllm==0.8.5.post1",
        ]

        [tool.uv]
        default-groups = ["dev", "lint", "data"]
        conflicts = [
            [
                { extra = "cpu" },
                { extra = "cu128" },
                { extra = "experimental" },
                { extra = "v04-cu124" },
            ],
        ]
        prerelease = "allow"

        [tool.uv.sources]
        torch = [
            { index = "pytorch-cpu", extra = "cpu" },
            { index = "pytorch-cu128", extra = "cu128" },
            { index = "pytorch-cu128", extra = "experimental" },
            { index = "pytorch-cu124", extra = "v04-cu124" },
        ]
        torchaudio = [
            { index = "pytorch-cpu", extra = "cpu" },
            { index = "pytorch-cu128", extra = "cu128" },
            { index = "pytorch-cu128", extra = "experimental" },
            { index = "pytorch-cu124", extra = "v04-cu124" },
        ]
        torchvision = [
            { index = "pytorch-cpu", extra = "cpu" },
            { index = "pytorch-cu128", extra = "cu128" },
            { index = "pytorch-cu128", extra = "experimental" },
            { index = "pytorch-cu124", extra = "v04-cu124" },
        ]
        fairseq2n = [
            { index = "fairseq2-cpu", extra = "cpu" },
            { index = "fairseq2-cu128", extra = "cu128" },
            { index = "fairseq2-experimental", extra = "experimental" },
            { index = "fairseq2-v04-cu124", extra = "v04-cu124" },
        ]
        fairseq2 = [
            { git = "https://github.com/facebookresearch/fairseq2", extra = "cpu" },
            { index = "fairseq2-cu128", extra = "cu128" },
            { index = "fairseq2-experimental", extra = "experimental" },
            { index = "fairseq2-v04-cu124", extra = "v04-cu124" },
        ]

        [[tool.uv.index]]
        name = "pytorch-cpu"
        url = "https://download.pytorch.org/whl/cpu"
        explicit = true

        [[tool.uv.index]]
        name = "pytorch-cu124"
        url = "https://download.pytorch.org/whl/cu124"
        explicit = true

        [[tool.uv.index]]
        name = "pytorch-cu128"
        url = "https://download.pytorch.org/whl/cu128"
        explicit = true

        [[tool.uv.index]]
        name = "fairseq2-cpu"
        url = "https://fair.pkg.atmeta.com/fairseq2/whl/pt2.7.1/cpu/"
        explicit = true

        [[tool.uv.index]]
        name = "fairseq2-cu128"
        url = "https://fair.pkg.atmeta.com/fairseq2/whl/pt2.7.1/cu128/"
        explicit = true

        [[tool.uv.index]]
        name = "fairseq2-experimental"
        url = "https://fair.pkg.atmeta.com/fairseq2/whl/pt2.7.1/cu128/"
        explicit = true

        [[tool.uv.index]]
        name = "fairseq2-v04-cu124"
        url = "https://fair.pkg.atmeta.com/fairseq2/whl/pt2.6.0/cu124/"
        explicit = true

Common Workflows
----------------

**Basic Installation & Usage:**

.. code-block:: sh

    # Install with CUDA 12.8 (recommended for GPU users)
    uv sync --extra cu128

    # Install with CPU-only (for development/CI)
    uv sync --extra cpu

    # Run fairseq2 commands
    uv run python -m fairseq2.assets list --kind model
    uv run python -c "from fairseq2.models.hub import load_model; print('✓ fairseq2 works!')"

**Development Workflow:**

.. code-block:: sh

    # Install with development tools
    uv sync --extra cu128 --group dev --group lint

    # Run linting
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src/

    # Run tests
    uv run pytest tests/ -v

**Switching Between Environments:**

.. code-block:: sh

    # Switch to CPU for testing
    uv sync --extra cpu
    uv run pytest tests/

    # Switch back to CUDA for training
    uv sync --extra cu128
    uv run python train_model.py

**Experimental Features:**

.. code-block:: sh

    # Install with Jupyter and experimental features
    uv sync --extra experimental
    uv run jupyter lab

**Legacy Version (v0.4):**

.. code-block:: sh

    # Use fairseq2 v0.4 with CUDA 12.4
    uv sync --extra v04-cu124

**Working with Existing fairseq2 Installation:**

.. code-block:: sh

    # Install fairseq2 from local source in editable mode
    uv sync --extra cu128
    source .venv/bin/activate
    uv pip install -e /path/to/fairseq2/repo

Key Concepts
------------

**Extras (Hardware Variants)**
    Defined in ``[project.optional-dependencies]``. These are mutually exclusive:

    - ``cpu``: CPU-only PyTorch and fairseq2
    - ``cu128``: CUDA 12.8 PyTorch and fairseq2
    - ``experimental``: Latest features with Jupyter support
    - ``v04-cu124``: Legacy fairseq2 v0.4 with CUDA 12.4

**Dependency Groups (Feature Sets)**
    Defined in ``[dependency-groups]``. These are additive:

    - ``dev``: Testing tools (pytest)
    - ``lint``: Code quality tools (mypy, ruff)
    - ``data``: Data processing tools (nltk, pyarrow)
    - ``doc``: Documentation tools (sphinx, etc.)

**Conflict Resolution**
    The ``conflicts`` section prevents mixing incompatible hardware variants.

**Custom Indexes**
    ``[tool.uv.sources]`` and ``[[tool.uv.index]]`` specify where to download PyTorch and fairseq2 packages for each variant.

Troubleshooting
---------------

**Version Mismatch Errors:**

.. code-block:: sh

    # Clear uv cache and reinstall
    uv cache clean
    uv sync --no-cache --extra cu128

**CUDA Version Issues:**

.. code-block:: sh

    # Check your CUDA version
    nvidia-smi

    # Use matching variant:
    # CUDA 12.8 -> --extra cu128
    # CUDA 12.4 -> --extra v04-cu124
    # No GPU -> --extra cpu

**Import Errors:**

.. code-block:: sh

    # Verify installation
    uv run python -c "import fairseq2; print(fairseq2.__version__)"
    uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

**Environment Issues:**

.. code-block:: sh

    # Create fresh environment
    rm -rf .venv uv.lock
    uv sync --extra cu128

Tips & Best Practices
---------------------

1. **Pin Your Environment**: Use ``uv lock`` to create reproducible builds
2. **Use Conflicts**: Prevent accidental mixing of CPU/CUDA variants
3. **Separate Concerns**: Use extras for hardware, groups for features
4. **Test Compatibility**: Always verify PyTorch/fairseq2 versions match
5. **Cache Management**: Use ``uv cache clean`` when troubleshooting

With uv, managing fairseq2 environments becomes fast, reliable, and conflict-free. The configuration above provides a solid foundation for both development and production use.
