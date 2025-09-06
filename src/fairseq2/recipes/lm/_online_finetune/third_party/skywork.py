import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fairseq2.logging import log


class SkyworkRMPipeline:
    def __init__(self, *args, **kwargs):
        model_path = "/datasets/pretrained-llms/Skywork-Reward-V2-Llama-3.1-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, prompt_chunk):
        inputs = self.tokenizer(
            prompt_chunk, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)

        outputs = self.model(**inputs)[0]
        rewards = [output[0] for output in outputs]

        return rewards
