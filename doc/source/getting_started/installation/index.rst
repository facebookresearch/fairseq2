.. _installation:

================================
:octicon:`download` Installation
================================

.. dropdown:: System Dependencies

    **Linux (Ubuntu/Debian):**

    .. code-block:: sh

        sudo apt install libsndfile1

    **Linux (Fedora/CentOS):**

    .. code-block:: sh

        sudo dnf install libsndfile

    **macOS:**

    .. code-block:: sh

        brew install libsndfile


Quick Install
-------------

To install fairseq2, simply run:

.. code-block:: sh

    pip install fairseq2

To install fairseq2 with a specific PyTorch version (*e.g.* PyTorch ``2.8.0`` and CUDA ``12.8``):

.. code-block:: sh

    pip install fairseq2\
      --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu128

To install fairseq2 with additional extras (*e.g.* ``arrow``) and a specific commit hash:

.. code-block:: sh

    pip install 'fairseq2[arrow] @ git+https://github.com/facebookresearch/fairseq2.git@<commit-hash>'\
      --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu128

For nightly builds:

.. code-block:: sh

    pip install fairseq2 --pre\
      --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt2.8.0/cu128

For an editable development installation:

.. code-block:: sh

    git clone https://github.com/facebookresearch/fairseq2.git
    cd fairseq2
    pip install -e . --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu128
    pip install -r requirements-devel.txt

Supported Variants
------------------

Replace ``pt2.8.0/cu128`` in the commands above with your desired combination:
**Linux:**

.. list-table:: Supported PyTorch Versions and CUDA Variants (Linux)
   :header-rows: 1

   * - PyTorch Version
     - Variants
   * - 2.8.0
     - ``pt2.8.0/cpu``, ``pt2.8.0/cu126``, ``pt2.8.0/cu128``
   * - 2.7.1
     - ``pt2.7.1/cpu``, ``pt2.7.1/cu126``, ``pt2.7.1/cu128``
   * - 2.6.0
     - ``pt2.6.0/cpu``, ``pt2.6.0/cu124``

**macOS (Apple Silicon):**

.. list-table:: Supported PyTorch Versions (macOS Apple Silicon)
   :header-rows: 1

   * - PyTorch Version
     - Variants
   * - 2.8.0
     - ``pt2.8.0/cpu``
   * - 2.7.1
     - ``pt2.7.1/cpu``

Windows
-------

fairseq2 does not support Windows natively. Use `Windows Subsystem for Linux (WSL) <https://learn.microsoft.com/en-us/windows/wsl/about>`_ and follow the Linux installation instructions.

.. warning::
   fairseq2 relies on the C++ API of PyTorch which has no API/ABI compatibility
   between releases. This means **you have to install the fairseq2 variant that
   exactly matches your PyTorch version**. Otherwise, you might experience issues
   like immediate process crashes or spurious segfaults. For the same reason, if
   you upgrade your PyTorch version, you must also upgrade your fairseq2
   installation.

.. toctree::
   :maxdepth: 1
   :caption: Advanced Installation

   installation_from_source
   setup_with_uv
