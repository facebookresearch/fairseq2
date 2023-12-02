# Mistral 7B
The [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b)  ([arXiv](https://arxiv.org/abs/2310.06825)) large language model is a pretrained generative text model with 7 billion parameters. It uses Grouped-Query Attention for fast inference, and Sliding-Window Attention for handling sequences of arbitrary length.
Its implementation can be found under [`fairseq2.models.mistral`](https://github.com/facebookresearch/fairseq2/tree/main/src/fairseq2/models/mistral).

- [Supported Models](#supported-models)
- [Loading the Checkpoints](#loading-the-checkpoints)
- [Use in Code](#use-in-code)
- [Chatbot](#chatbot)
- [Fine-tuning](#fine-tuning)

## Supported Models
The following Mistral 7B models are available in fairseq2:

- `mistral_7b`
- `mistral_7b_instruct`

## Loading the Checkpoints
The checkpoints of Mistral 7B models are publicly available and, by default, fairseq2 will automatically download them to its cache directory at `~/.cache/fairseq2`. However, if you have already downloaded them in the past, you can instruct fairseq2 to skip the download using one of the following options:

### Option 1
For the recipes under this directory, you can pass the checkpoint path using the `--checkpoint-dir` option.

### Option 2
For general use, you can set them permanently by creating a YAML file (e.g. `mistral.yaml`) with the following template under `~/.config/fairseq2/assets`:

```yaml
name: mistral_7b@user
checkpoint: "<path/to/checkpoint>"
tokenizer: "<path/to/tokenizer>"

---

name: mistral_7b_instruct@user
checkpoint: "<path/to/checkpoint>"
tokenizer: "<path/to/tokenizer>"
```

## Use in Code
The short example below shows how you can complete a small batch of prompts using `mistral_7b`.

```python
import torch

from fairseq2.generation import TopPSampler, SamplingSequenceGenerator, TextCompleter
from fairseq2.models.mistral import load_mistral_model, load_mistral_tokenizer

model = load_mistral_model("mistral_7b", device=torch.device("cuda"), dtype=torch.float16)

tokenizer = load_mistral_tokenizer("mistral_7b")

sampler = TopPSampler(p=0.6)

generator = SamplingSequenceGenerator(model, sampler, echo_prompt=True)

text_completer = TextCompleter(generator, tokenizer)

prompts = [
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that ",
]

output, _ = text_completer.batch_complete(prompts)

for text in output:
  print(text)
```

## Chatbot
The [`chatbot.py`](./chatbot.py) script demonstrates how you can build a simple terminal-based chat application using Mistral 7B and fairseq2. To try it out, run:

```sh
python recipes/mistral/chatbot.py
```

## Fine-tuning
Coming soon. Stay tuned.
