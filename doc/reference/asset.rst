fairseq2.asset
==============
.. body

.. currentmodule:: fairseq2.asset

``fairseq2.asset`` provides API to load the different model using the "model cards" from different "stores".

A model card is a .YAML file that contains information about a model and instructs a 
:pc:class:`fairseq2.models.utils.generic_loaders.ModelLoader` on how to load the model into the memory.
A store is a place where all the model cards are stored. By default, fairseq2 will look up the following stores:

* System asset store: Cards that are shared by all users. By default, the system store is `/etc/fairseq2/assets`,
but this can be changed via the environment variable `FAIRSEQ2_ASSET_DIR`
* User asset store: Cards that are only available to the user. By default, the user store is 
`~/.config/fairseq2/assets`, but this can be changed via the environment variable `FAIRSEQ2_USER_ASSET_DIR`
* (Internal only) Meta asset store: For Meta employees' convenience, we set up a central store that contains
model cards with e.g intermediate checkpoints, extra internal information etc. This store is registered automatically
when one logs into the Fair cluster. If you wish not to use this central store, set the environment variable 
`NO_FAIR_CARD=ON`