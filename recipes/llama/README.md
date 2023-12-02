# LLaMA and LLaMA 2
[LLaMA](https://ai.meta.com/llama/) ([arXiv](https://arxiv.org/abs/2302.13971)) and [LLaMA 2](https://ai.meta.com/llama/) ([arXiv](https://arxiv.org/abs/2307.09288)) family of large language models by Meta are a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. Their implementation can be found under [`fairseq2.models.llama`](https://github.com/facebookresearch/fairseq2/tree/main/src/fairseq2/models/llama).

- [Supported Models](#supported-models)
- [Loading the Checkpoints](#loading-the-checkpoints)
- [Use in Code](#use-in-code)
- [Chatbot](#chatbot)
- [Fine-tuning](#fine-tuning)

## Supported Models
As of today the following LLaMA 7B models are available in fairseq2. The larger variants will be made available soon:

- `llama_7b`
- `llama2_7b`
- `llama2_7b_chat`

## Loading the Checkpoints
LLaMA checkpoints are gated and can only be downloaded after filling out the form [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/). Once downloaded, there are two simple ways to access them in fairseq2:

### Option 1
For the recipes under this directory, you can pass the checkpoint path using the `--checkpoint-dir` option.

### Option 2
For general use, you can set them permanently by creating a YAML file (e.g. `llama.yaml`) with the following template under `~/.config/fairseq2/assets` (shown for `llama2_7b` and `llama2_7b_chat`):

```yaml
name: llama2_7b@user
checkpoint: "<path/to/checkpoint>"
tokenizer: "<path/to/tokenizer>"

---

name: llama2_7b_chat@user
checkpoint: "<path/to/checkpoint>"
tokenizer: "<path/to/tokenizer>"
```

## Use in Code
The short example below shows how you can complete a small batch of prompts using `llama2_7b`.

```python
import torch

from fairseq2.generation import TopPSampler, SamplingSequenceGenerator, TextCompleter
from fairseq2.models.llama import load_llama_model, load_llama_tokenizer

model = load_llama_model("llama2_7b", device=torch.device("cuda"), dtype=torch.float16)

tokenizer = load_llama_tokenizer("llama2_7b")

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
The [`chatbot.py`](./chatbot.py) script demonstrates how you can build a simple terminal-based chat application using LLaMA 2 and fairseq2. To try it out, run:

```sh
python recipes/llama/chatbot.py --checkpoint-dir <path/to/llama/checkpoint/dir>
```

Note that you can omit `--checkpoint-dir` if you have set the checkpoint in a YAML file (option 2 above).

## Fine-tuning
Coming soon. Stay tuned.
