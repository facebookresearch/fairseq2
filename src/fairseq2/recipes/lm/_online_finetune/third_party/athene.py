from typing import Any, Dict, List, cast

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaModel,
    LlamaPreTrainedModel,
    TextClassificationPipeline,
)


class AtheneForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.CLS_ID = 128003
        # Initialize weights and apply final processing
        self.post_init()

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = transformer_outputs.hidden_states[-1]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)

        bs = int(input_ids.shape[0])

        for i in range(bs):
            c_inds = (input_ids[i] == self.CLS_ID).nonzero()
            if len(c_inds) == 0:
                # FIXME why is this happening? should always be a CLS_ID token
                scores.append(torch.Tensor([0.0]))
            else:
                c_ind = c_inds[-1].item()
                scores.append(rewards[i, c_ind])
        scores = torch.stack(scores)
        return {"scores": scores}


class AtheneRewardPipeline(TextClassificationPipeline):
    def __init__(self, *args, **kwargs):
        model = AtheneForSequenceClassification.from_pretrained(
            "Nexusflow/Athene-RM-8B", torch_dtype="bfloat16"
        )
        tokenizer = AutoTokenizer.from_pretrained("Nexusflow/Athene-RM-8B")

        super().__init__(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )

    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, torch.Tensor]:
        return_tensors = self.framework

        formatted = self.tokenizer.apply_chat_template(inputs, tokenize=False)

        formatted = formatted + self.tokenizer.cls_token

        return self.tokenizer(
            formatted,
            return_tensors=return_tensors,
            max_length=4096,  # FIXME use config len
            padding="longest",
            truncation=True,
        )

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        return model_outputs["scores"].cpu().float().item()
