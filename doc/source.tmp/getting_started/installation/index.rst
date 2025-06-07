.. _installation:

================================
:octicon:`download` Installation
================================

.. _installation_linux:

Installing on Linux
-------------------

Linux OS System Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, fairseq2 is installed on Linux OS. fairseq2 depends on
`libsndfile <https://github.com/libsndfile/libsndfile>`_, which can be installed
via the system package manager on most Linux distributions. For Ubuntu-based
systems, run:

.. code-block:: sh

    sudo apt install libsndfile1

Similarly, on Fedora, run:

.. code-block:: sh

    sudo dnf install libsndfile

For other Linux distributions, please consult its documentation on how to
install packages.

pip (Linux)
^^^^^^^^^^^

To install fairseq2 on Linux x86-64, run:

.. code-block:: sh

    pip install fairseq2

This command will install a version of fairseq2 that is compatible with PyTorch
hosted on PyPI.

At this time, we do not offer a pre-built package for ARM-based systems such as
Raspberry PI or NVIDIA Jetson. Please refer to :ref:`installation_from_source` to
learn how to build and install fairseq2 on those systems.

Variants (Linux)
^^^^^^^^^^^^^^^^

Besides PyPI, fairseq2 also has pre-built packages available for different
PyTorch and CUDA versions hosted on FAIR's package repository. The following
matrix shows the supported combinations.

.. list-table:: Supported Combinations
    :header-rows: 1
    :widths: 15 15 20 20 10

   * - fairseq2
     - PyTorch
     - Python
     - Variant*
     - Arch
   * - ``HEAD``
     - ``2.4.0``
     - ``>=3.10``, ``<=3.12``
     - ``cpu``, ``cu118``, ``cu121``
     - ``x86_64``
   * - ``HEAD``
     - ``2.3.0``, ``2.3.1``
     - ``>=3.10``, ``<=3.12``
     - ``cpu``, ``cu118``, ``cu121``
     - ``x86_64``
   * - ``HEAD``
     - ``2.2.0``, ``2.2.1``, ``2.2.2``
     - ``>=3.10``, ``<=3.12``
     - ``cpu``, ``cu118``, ``cu121``
     - ``x86_64``
   * - ``0.2.0``
     - ``2.1.1``
     - ``>=3.8``, ``<=3.11``
     - ``cpu``, ``cu118``, ``cu121``
     - ``x86_64``
   * - ``0.2.0``
     - ``2.0.1``
     - ``>=3.8``, ``<=3.11``
     - ``cpu``, ``cu117``, ``cu118``
     - ``x86_64``
   * - ``0.2.0``
     - ``1.13.1``
     - ``>=3.8``, ``<=3.10``
     - ``cpu``, ``cu116``
     - ``x86_64``

*\* cuXYZ refers to CUDA XY.Z (e.g. cu118 means CUDA 11.8)*


To install a specific combination, first follow the installation instructions on
`pytorch.org <https://pytorch.org/get-started/locally>`_ for the desired PyTorch
version, and then use the following command (shown for PyTorch `2.4.0` and
variant `cu121`):

.. code-block:: bash

    pip install fairseq2\
      --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.4.0/cu121

.. warning::

    fairseq2 relies on the C++ API of PyTorch which has no API/ABI compatibility
    between releases. This means **you have to install the fairseq2 variant that
    exactly matches your PyTorch version**. Otherwise, you might experience issues
    like immediate process crashes or spurious segfaults. For the same reason, if
    you upgrade your PyTorch version, you must also upgrade your fairseq2
    installation.

Nightlies
^^^^^^^^^

For Linux, we also host nightly builds on FAIR's package repository. The
supported variants are identical to the ones listed in *Variants* above. Once
you have installed the desired PyTorch version, you can use the following
command to install the corresponding nightly package (shown for PyTorch `2.4.0`
and variant `cu121`):

.. code-block:: sh

    pip install fairseq2\
      --pre --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt2.4.0/cu121

.. _installation_mac:

Installing on macOS
-------------------

macOS System Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^

fairseq2 depends on `libsndfile <https://github.com/libsndfile/libsndfile>`__,
which can be installed via Homebrew:

.. code-block:: sh

    brew install libsndfile


pip (macOS)
^^^^^^^^^^^

To install fairseq2 on ARM64-based (i.e. Apple silicon) Mac computers, run:

.. code-block:: sh

    pip install fairseq2

This command will install a version of fairseq2 that is compatible with PyTorch
hosted on PyPI.

At this time, we do not offer a pre-built package for Intel-based Mac computers.
Please refer to :ref:`installation_from_source` to learn how to build and
install fairseq2 on Intel machines.

Variants (macOS)
^^^^^^^^^^^^^^^^

Besides PyPI, fairseq2 also has pre-built packages available for different
PyTorch versions hosted on FAIR's package repository. The following matrix shows
the supported combinations.

.. list-table:: Supported Combinations
    :header-rows: 1
    :widths: 10 10 10 10

    * - fairseq2
      - PyTorch
      - Python
      - Arch
    * - ``HEAD``
      - ``2.4.0``
      - ``>=3.9``, ``<=3.12``
      - ``arm64``

To install a specific combination, first follow the installation instructions on
`pytorch.org <https://pytorch.org/get-started/locally>`__ for the desired PyTorch
version, and then use the following command (shown for PyTorch `2.4.0`):

.. code-block:: sh

    pip install fairseq2\
      --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.4.0/cpu


.. warning::

    fairseq2 relies on the C++ API of PyTorch which has no API/ABI compatibility
    between releases. This means **you have to install the fairseq2 variant that
    exactly matches your PyTorch version**. Otherwise, you might experience
    issues like immediate process crashes or spurious segfaults. For the same
    reason, if you upgrade your PyTorch version, you must also upgrade your
    fairseq2 installation.

Nightlies (macOS)
^^^^^^^^^^^^^^^^^

For macOS, we also host nightly builds on FAIR's package repository. The
supported variants are identical to the ones listed in *Variants* above. Once
you have installed the desired PyTorch version, you can use the following
command to install the corresponding nightly package  (shown for PyTorch `2.4.0`):

.. code-block:: sh

    pip install fairseq2\
      --pre --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt2.4.0/cpu


.. _installation_windows:

Installing on Windows
---------------------

fairseq2 does not have native support for Windows and there are no plans to
support it in the foreseeable future. However, you can use fairseq2 via the
`Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/about>`__
(a.k.a. WSL) along with full CUDA support introduced in WSL 2. Please follow the
instructions in the :ref:`installation` section for a WSL-based
installation.


.. toctree::
    :maxdepth: 1
    :caption: Other Installation Guides

    installation_from_source
    setup_with_uv
