Creating you own training script
================================

In this tutorial you will write a training script from scratch, to train an English to French translation model.

To train a machine learning model, you need the following components:

* a task: ``task``
* some train data: ``train_data``
* a PyTorch module: ``module``
* some validation data: ``valid_data``
* a PyTorch optimizer: ``optimizer``
* a schedule for the learning rate: ``lr_scheduler``

Following a pre-defined structure allows some things to be automatically handled by fairseq2,
like generating a CLI for the experiment script,
or reloading models for evaluation and inference.

Machine learning models and tasks can be arbitrarily complex,
so Fairseq2 uses Python to write the training scripts.
A training script consists of a list of function instantiating the main components.
Fairseq2 takes care of following the recipe, fetching the ingredients (components),
and proceeding with the cooking (training) and serve the dish (show evaluation results).


Downloading the data
~~~~~~~~~~~~~~~~~~~~

For this task use the `Tatoeba`_ dataset.
Download the dataset locally and unzip it:

.. code-block:: bash

  wget https://www.manythings.org/anki/fra-eng.zip
  unzip fra-eng.zip fra.txt -d examples/
  rm fra-eng.zip
  head examples/fra.txt

The data is tab separated data.
First column contain English sentence, and the second their French translation.

Starting up
~~~~~~~~~~~

Create a python file ``tatoeba.py`` with a simple docstring at the top, and a small Fairseq2 boilerplate::

  """Train eng-fra translation model on Tatoeba dataset"""

Then run

.. code-block:: bash

  fairseq2 help tatoeba.py

This shows the help for the newly created training script.
The first line is the docstring you just wrote at the top of your script.
Then Fairseq2 print more documentation about the experiment CLI.
Your experiment script inherit some fairseq2 defaults, so some things are already defined for you,
notably the ``task``.
More on this later.
But for now focus on the fact that ``module``, ``tokenizer`` and ``train_data`` lines have a warning: ``!!! No XXX specified !!!``.

Go on implementing those in your script.

``tokenizer``
~~~~~~~~~~~~~

Neural networks don't like text.
What they like is good old numbers.
So you need to convert the input text to list of numbers.
Typically you want to assign each word to a unique number, and replace the input sentence by a list of the corresponding numbers.
In traditional machine translation the choice of vocabulary was an important task.
To avoid those issues use a statistical tokenizer using `sentencepiece`_ library.
``sentencepiece`` models don't always split by words, but may use smaller word pieces too.
This allows to decrease the size of the model while still covering a large spectrum of words.

``fairseq2`` asks you to provide a ``tokenizer`` for your model.
Explicitly declaring the ``tokenizer`` is necessary for deploying the model,
and offering a string to string interface.

``sentencepiece`` models being statistical tokenizers, also need to be trained.
Add some code to your script to train a bilingual tokenizer::

  from pathlib import Path
  from fairseq2 import data
  from fairseq2.generate import spm_train

  TSV_FILE = Path("examples/fra.txt")
  SPM_PATH = Path("examples/fra.spm")


  def tokenizer(
      tsv_file: Path = TSV_FILE, vocab_size: int = 5_000, spm: str = SPM_PATTERN
  ) -> data.text.MultilingualTokenizer:
      """eng-fra sentencepiece tokenizer"""
      spm_path = Path(spm.format(vocab_size=vocab_size))
      if not spm_path.exists():
          cfg = spm_train.TrainSpmConfig(vocab_size=vocab_size)
          eng_fra = (
              data.text.read_text(str(tsv_file), rtrim=True, skip_header=1)
              .map(lambda line: "\n".join((str(line).split("\t")[:1])))
              .and_return()
          )
          spm_train.train_from_stream(cfg, eng_fra, spm_path)

      return data.text.MultilingualTokenizer(
          spm_path, "translation", {"eng"}, {"fra"}, "eng", "fra"
      )


Take some time to understand what's going on here.
The `tokenizer` function is first checking if the trained tokenizer already exists.
If it doesn't it creates a dataloader that reads all the English and French sentences from Tatoeba dataset and trains a sentencepiece tokenizer on it.
Note that the script creates one tokenizer model per ``vocab_size`` since this is an important hyper-paramter of this model.
Finally it creates a :py:class:MultilingualTokenizer object that wraps the trained ``sentencepiece`` model
and conform with the :py:class:`Tokenizer` API.

The arguments of the `tokenizer` function
(``tsv_file``, ``vocab_size`` and ``spm``)
are visible from the CLI, and therefore need type annotations.

Now, rerun the help command, it prints:

.. code-block:: text

  **tokenizer** (?): eng-fra sentencepiece tokenizer
    tokenizer.tsv_file (pathlib.Path): (default=examples/fra.txt)
    tokenizer.vocab_size (int): (default=5000)
    tokenizer.spm_path (pathlib.Path): (default=examples/fra.spm)

Here fairseq2 is showing the default value of each of the tokenizer settings and how to set them from CLI.
Note that those values aren't imposed by fairseq2 but are extracted from your code.

You still can't run the training (you are missing ``train_data`` and ``module``),
but you can already test that the ``tokenizer`` function is working as intended.
Try that with:

.. code-block:: bash

  fairseq2 test tatoeba.py -f tokenizer
  ...
  trainer_interface.cc(686) LOG(INFO) Saving model: examples/fra.4000.spm.model
  trainer_interface.cc(698) LOG(INFO) Saving vocabs: examples/fra.4000.spm.vocab
  INFO:fairseq2.generate.spm_train:sentencepiece training completed.
  Success! tokenizer() = <fairseq2.data.text.multilingual_tokenizer.MultilingualTokenizer object at 0x7f4b93cddd50>

You should see some logs from sentencepiece training the model.
You can retry with a smaller vocab size by appending ``vocab_size=4000`` to previous command.
If you retry with the same argument,
the function loads the file from disk and returns immediately.


``train_data``
~~~~~~~~~~~~~~

Now that you have a tokenizer
you can move on to the next step,
loading the data and creating batches.
In fairseq2 the default data type for sequence to sequence tasks is :py:class:`fairseq2.data.Seq2SeqBatch`.
The ``train_data`` ingredient is expected to be an :py:class:`typing.Iterable` of batch.
You're free to use another type of batch as long as you're consistent within the same script.
For this tutorial, stick with the default ``Seq2SeqBatch``
During training, fairseq2 reads ``train_data`` ``Iterable`` several times.
Each iteration, corresponds to one epoch.


Write the following dataloader using :py:mod:`fairseq2.data`.::

  from fairseq2 import data
  from fairseq2.cli import Env

  def train_data(
      tokenizer: data.text.MultilingualTokenizer,
      env: Env,
      batch_size: int = 32,
      tsv_file: Path = TSV_FILE,
  ):
      def _read_tsv_column(encoder, column: int):
          return (
              data.text.read_text(tsv_file, rtrim=True)
              .map(lambda line: str(line).split("\t")[column])
              .map(encoder)
              .and_return()
          )

      src = _read_tsv_column(
          tokenizer.create_encoder(mode="source", lang="eng"), column=0
      )
      tgt = _read_tsv_column(
          tokenizer.create_encoder(mode="target", lang="fra"), column=1
      )

      pad_idx = tokenizer.vocab_info.pad_idx
      device = env.device
      batches = (
          data.zip_data_pipelines([src, tgt])
          .shuffle(10_000)
          .batch(batch_size, pad_idx=pad_idx)
          .map(
              lambda st: data.Seq2SeqBatch(
                  source=st[0].to(device),
                  # TODO: the tokenizer should compute those
                  src_seq_lens=(st[0] != pad_idx).sum(dim=-1).to(device),
                  target=st[1].to(device),
                  tgt_seq_lens=(st[1][:, 1:] != pad_idx).sum(dim=-1).to(device),
              )
          )
          .and_return()
      )
      return batches

First look at the signature of ``train_data``.
The dataloader needs to be able to tokenize the input data,
so it depends on the output of the ``tokenizer`` function.
To explicit this dependency, like in regular python code,
pas ``tokenizer`` as an argument to ``train_data`` function.
Similarly the dataloader need to put the batches on the same device than the model.
This adds a dependency on ``env`` describing the current execution environment.
``env`` is one of the :ref:`reference/cli:Fairseq2 script built-in ingredients`.


``module``
~~~~~~~~~~

``module`` is the :py:class:`torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`.
representing the machine translation model.

To put your model on the right device, add the ``env`` argument to the ``module`` function.
To make sure the model has one embedding by word in the
tokenizer vocabulary,
use the ``tokenizer`` argument too::

  def module(
      model_cfg: fairseq2.models.nllb.NllbConfig,
      env: Env,
  ) -> torch.nn.Module:
      return fairseq2.models.nllb.create_nllb_model(model_cfg, env.device)


  def model_cfg(
      tokenizer: data.text.Tokenizer, model_dim: int = 128, num_layers: int = 4
  ) -> fairseq2.models.nllb.NllbConfig:
      cfg = fairseq2.models.nllb.nllb_archs.get_config("dense_600m")
      return dataclasses.replace(
          cfg,
          vocabulary_size=tokenizer.vocab_info.size,
          pad_idx=tokenizer.vocab_info.pad_idx,
          model_dim=model_dim,
          num_encoder_layers=num_layers,
          num_decoder_layers=num_layers,
          ffn_inner_dim=model_dim * 4,
      )


The scripts introduces a helper function ``model_cfg``.
Since ``model_cfg`` is also an argument to ``module``` its result is passed to ``module``.
Separating this two functions helps Fairseq2 generating a nice yaml file.
Indeed Fairseq2 can't serialize to yaml``module``,
but it can handle ``model_cfg``.
That way the yaml file will contain all the fields of :py:class:`fairseq2.models.nllb.NllbConfig`.

.. note:: Fairseq2 won't implicitly wrap your model for FSDP or DDP.
  You'll need to decide yourself if you want to do it.

As before verify that it's working by running::

  fairseq2 test tatoeba.py -f model_cfg
  fairseq2 test tatoeba.py -f module

Now rerun the help command, and check there is no warning anymore::

  fairseq2 help tatoeba.py

You're now ready to run the training. You can refer to :ref:training ::

  fairseq2 train tatoeba.py

.. TODO:: The tutorial ends here, but we need to explain
  how to customize the loss.

``task``
~~~~~~~~

In fairseq2 a "task" is the code explaining how the module should use the data to make predictions and compute a loss.
The basic task is :py:class:`fairseq2.tasks.Seq2Seq`, which implement classic sequence to sequence translation, and uses negative log likely hood for loss.
If you want another loss, inheriting from :py:class:`.Seq2Seq` and overriding :py:meth:`.compute_loss` method is typically enough.
In any case the task should implement the Unit class from `torchTNT`_ to work with the TNT training loop fairseq2 is using.

The task is also the extension point that allows you to add custom hooks during training.

.. _tatoeba: https://tatoeba.org/
.. _torchTNT: https://pytorch.org/tnt/stable/
.. _sentencepiece: https://github.com/google/sentencepiece/
