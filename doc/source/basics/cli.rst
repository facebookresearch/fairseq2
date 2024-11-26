.. _basics-cli:

:octicon:`terminal` CLI
=======================

The Command-Line Interface (CLI) is a crucial feature in fairseq2, offering users a powerful and flexible way to interact with the framework. With the CLI, users can quickly and easily execute tasks, customize recipes and configurations, and perform complex operations such as sweep runs and benchmarking. For example:

.. code-block:: bash

    # dump the default configuration for instruction finetuning
    fairseq2 lm instruction_finetune --dump-config

    # run a recipe - put the OUTPUT_DIR right after the recipe name
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config-file <YOUR_CONFIG>.yaml
    # or, put the OUTPUT_DIR at the end following a "--"
    fairseq2 lm instruction_finetune --config-file <YOUR_CONFIG>.yaml -- <OUTPUT_DIR>

In this section, we will cover:

1. `More examples of CLI <#cli-examples>`_
2. `How to add your CLI <#cli-add>`_
3. `How the CLI is initialized <#cli-initialize>`_

Key benefits of CLI in fairseq2:
- **Rapid Interaction**: The CLI enables users to rapidly interact with fairseq2, allowing for quick execution of tasks and experiments.
- **Customization**: Users can conveniently customize recipes and configurations by adding or modifying ``--config`` options, making it easy to adapt fairseq2 to specific use cases.
- **Sweep Runs and Benchmarking**: The CLI facilitates sweep runs and benchmarking with a set of different configurations, enabling users to simply customize bash scripts for sbatch or srun.


.. _cli-examples:

More Example CLI-s
------------------
   
   
To get started, you can use the help with ``-h`` whenever you have a question about the CLI at any level:

.. code-block:: console

    $ fairseq2 -h
    usage: fairseq2 [-h] [--version] {assets,lm,llama,mt,wav2vec2,wav2vec2_asr} ...

    command line interface of fairseq2

    positional arguments:
      {assets,lm,llama,mt,wav2vec2,wav2vec2_asr}
        assets              list and show assets (e.g. models, tokenizers, datasets)
        lm                  language model recipes
        llama               LLaMA recipes
        mt                  machine translation recipes
        wav2vec2            wav2vec 2.0 pretraining recipes
        wav2vec2_asr        wav2vec 2.0 ASR recipes

    options:
      -h, --help            show this help message and exit
      --version             show program's version number and exit

Each of the positional arguments listed above (``assets``, ``lm``, ``llama``, ...) is called "cli group". For example, in the ``assets`` group, you can list all assets:

.. code-block:: bash

    $ fairseq2 assets list
    user:
      n/a

    global:
      package:fairseq2.assets.cards
      - dataset:openeft
      - dataset:librilight_asr_10h
      - dataset:librispeech_asr
      - dataset:librispeech_asr_100h
      - dataset:librispeech_960h
      - model:s2t_transformer_mustc_asr_de_s
      - model:s2t_transformer_mustc_asr_es_s
      ...
      - tokenizer:mistral_7b
      - tokenizer:mistral_7b_instruct
      package:fairseq2_ext.cards
      - dataset:librispeech_asr@awscluster
      - dataset:librispeech_asr@faircluster
      ...
      - dataset:openeft@awscluster
      - dataset:openeft@faircluster
      - model:nllb-200@faircluster
      - model:nllb-200_dense_1b@faircluster
      ...
      - tokenizer:llama3_2_3b@awscluster
      - tokenizer:llama3_2_3b_instruct@awscluster

You can also show a specific asset with more verbose information (`e.g.` checkpoint path), for example:

.. code-block:: bash

    $ fairseq2 assets show llama3_1_8b_instruct
    llama3_1_8b_instruct
      source          : 'package:fairseq2_ext.cards'
      base            : 'llama3_instruct'
      model_arch      : 'llama3_1_8b'
      checkpoint      :
    '/fsx-ram/shared/Meta-Llama-3.1-8B-Instruct/original/consolidated.00.pth'

    llama3_instruct
      source          : 'package:fairseq2.assets.cards'
      base            : 'llama3'
      model_config    : {'vocab_info': {'eos_idx': 128009}}

    llama3
      source          : 'package:fairseq2_ext.cards'
      model_family    : 'llama'
      checkpoint      : 'https://ai.meta.com/llama/;gated=true'
      tokenizer       : '/fsx-ram/shared/Meta-Llama-3-8B/tokenizer.model'
      tokenizer_family: 'llama'
      use_v2_tokenizer: True

.. _cli-add:

How to add a CLI?
-----------------

Take ``fairseq2 lm instruction_finetune ...`` as example, the command and handler are registered at :meth:`fairseq2.recipes.lm._setup_lm_cli`.

.. code-block:: python

    def _setup_lm_cli(cli: Cli) -> None:
        group = cli.add_group("lm", help="language model recipes")

        # Instruction Finetune
        instruction_finetune_handler = RecipeCommandHandler(
            loader=load_instruction_finetuner,
            preset_configs=instruction_finetune_presets,
            default_preset="llama3_1_instruct",
        )

        group.add_command(
            name="instruction_finetune",
            handler=instruction_finetune_handler,
            help="instruction-finetune a language model",
        )

The callback function is passed as ``loader``. In this case, you can check the ``load_instruction_finetuner`` :meth:`fairseq2.recipes.lm.instruction_finetune.load_instruction_finetuner` for more details.

.. _cli-initialize:

How is fairseq2 CLI initialized?
--------------------------------

The ``Cli`` class is defined :class:`fairseq2.recipes.CLI` and instantiated in :meth:`fairseq2.recipes.main`. For example:

.. code-block:: python

    def main() -> None:
        """Run the command line fairseq2 program."""
        from fairseq2 import __version__, setup_fairseq2

        with exception_logger(log):
            setup_basic_logging()

            setup_fairseq2()

            cli = Cli(
                name="fairseq2",
                origin_module="fairseq2",
                version=__version__,
                description="command line interface of fairseq2",
            )

            container = get_container()

            _setup_cli(cli, container)
            _setup_cli_extensions(cli, container)

The :meth:`fairseq2.recipes._setup_cli` is where we have all our CLI groups registered:

.. code-block:: python

    def _setup_cli(cli: Cli, resolver: DependencyResolver) -> None:
        _setup_asset_cli(cli)
        _setup_lm_cli(cli)
        _setup_llama_cli(cli)
        _setup_mt_cli(cli)
        _setup_wav2vec2_cli(cli)
        _setup_wav2vec2_asr_cli(cli)
        _setup_hg_cli(cli)

So if you want to add your own group of CLI, don't forget to check it out here.