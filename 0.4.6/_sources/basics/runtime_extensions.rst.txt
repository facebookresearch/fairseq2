.. _basics-runtime-extensions:

:octicon:`plug` Runtime Extensions
==================================

fairseq2 provides a flexible runtime extension system that allows you to extend its functionality without modifying the core codebase. This system leverages Python's setuptools entry points to dynamically load and register extensions during initialization.

Overview
--------

The extension system is built around a dependency injection container (learn more in :ref:`basics-design-philosophy`) that manages fairseq2's components.
Through this system, you can:

* Register new models and model cards
* Add custom asset providers and validators  
* Extend the runtime context
* Register custom tensor loaders/dumpers
* Add value converters and type validators
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

    from fairseq2.context import RuntimeContext

    def setup_my_extension(context: RuntimeContext) -> None:
        # Register your custom components here using `context`.
        pass

Registering Extensions
----------------------

Extensions are registered using setuptools entry points. You can configure them in either ``setup.py`` or ``pyproject.toml``:

Using setup.py:

.. code-block:: python

    setup(
        name="my-fairseq2-extension",
        entry_points={
            "fairseq2.extension": [
                "my_extension = my_package.module:setup_my_extension",
            ],
        },
    )

Using pyproject.toml:

.. code-block:: toml

    [project.entry-points."fairseq2.extension"]
    my_extension = "my_package.module:setup_my_extension"

Extension Loading Process
-------------------------

When ``setup_fairseq2()`` is called, the following steps occur:

1. fairseq2 components are initialized
2. All registered extensions are discovered via entry points
3. Each extension's setup function is called

Complete Example
----------------

Here's a complete example of implementing a fairseq2 extension:

.. code-block:: python

    from fairseq2.context import RuntimeContext
    from fairseq2.setup import register_package_metadata_provider

    def setup_my_extension(context: RuntimeContext) -> None:
    
        # Get the global asset store
        asset_store = context.asset_store

        # To manage assets from a custom source, you can append a function that returns the asset source name to the list of environment resolvers
        # For example, the following code registers a function that returns "mycluster" as the asset source name.
        # This allows you to add assets in the asset cards with identifiers that ends with "@mycluster".
        asset_store.env_resolvers.append(lambda: "mycluster")
    
        # Register a package metadata provider for the "my_package" and read the model cards from the "my_package.cards" module.
        register_package_metadata_provider(context, "my_package.cards")

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
