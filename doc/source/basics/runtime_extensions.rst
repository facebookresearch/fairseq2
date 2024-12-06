.. _basics-runtime-extensions:

:octicon:`plug` Runtime Extensions
==================================

fairseq2 provides a flexible runtime extension system that allows you to extend its functionality without modifying the core codebase. This system leverages Python's setuptools entry points to dynamically load and register extensions during initialization.

Overview
--------

The extension system is built around a dependency injection container (learn more in :ref:`basics-design-philosophy`) that manages fairseq2's components.
Through this system, you can:

* Register new models
* Add custom asset providers
* Extend the runtime context
* Register custom tensor loaders/dumpers
* Add value converters
* And more...

Basic Usage
-----------

Before using any fairseq2 APIs, you must initialize the framework with :meth:`fairseq2.setup_fairseq2`:

.. code-block:: python

    from fairseq2 import setup_fairseq2

    setup_fairseq2()

Creating Extensions
-------------------

To create an extension, define a setup function:

.. code-block:: python

    def setup_my_extension() -> None:
        # Register your custom components here
        pass

Registering Extensions
----------------------

Extensions are registered using setuptools entry points. You can configure them in either ``setup.py`` or ``pyproject.toml``:

Using setup.py:

.. code-block:: python

    setup(
        name="my-fairseq2-extension",
        entry_points={
            "fairseq2": [
                "my_extension = my_package.module:setup_my_extension",
            ],
        },
    )

Using pyproject.toml:

.. code-block:: toml

    [project.entry-points."fairseq2"]
    my_extension = "my_package.module:setup_my_extension"

Extension Loading Process
-------------------------

When ``setup_fairseq2()`` is called, the following steps occur:

2. fairseq2 components are initialized
3. All registered extensions are discovered via entry points
4. Each extension's setup function is called

Complete Example
----------------

Here's a complete example of implementing a fairseq2 extension:

.. code-block:: python

    from fairseq2.assets import default_asset_store

    def setup_my_extension() -> None:
        default_asset_store.add_package_metadata_provider("my_package")

Error Handling
--------------

The extension system includes error handling to maintain system stability:

* Failed extensions log warnings by default
* Set ``FAIRSEQ2_EXTENSION_TRACE`` environment variable for detailed error traces
* Invalid extension functions raise ``RuntimeError``

.. code-block:: bash

    export FAIRSEQ2_EXTENSION_TRACE=1


Best Practices
--------------

We suggest the following best practices for implementing extensions.

Documentation
^^^^^^^^^^^^^

* Document your extension's functionality
* Specify requirements and dependencies
* Include usage examples

Testing
^^^^^^^

* Test extensions in isolation
* Verify integration with fairseq2
* Test error cases and edge conditions

Error Handler
^^^^^^^^^^^^^

* Implement proper error handling
* Fail fast if required dependencies are missing
* Provide meaningful error messages

Configuration
-------------

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

``FAIRSEQ2_EXTENSION_TRACE``
    Set this environment variable to enable detailed stack traces when extensions fail to load.

See Also
--------

* :doc:`/reference/api/fairseq2.assets/index`
