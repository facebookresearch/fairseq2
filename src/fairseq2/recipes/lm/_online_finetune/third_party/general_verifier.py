import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert_decisions_to_binary(string_list):
    binary_list = []
    for s in string_list:
        if "Final Decision: Yes" in s:
            binary_list.append(1)
        else:
            binary_list.append(0)
    return binary_list


class GeneralVerifierPipeline:
    def __init__(self, *args, **kwargs):
        model_path = "TIGER-Lab/general-verifier"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).cuda()

    def __call__(self, prompt_chunk):
        cleaned_prompt_chunk = [
            x.replace("[", "\[").replace("]", "\]") for x in prompt_chunk
        ]

        inputs = self.tokenizer(cleaned_prompt_chunk, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        text_out = self.tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        rewards = convert_decisions_to_binary(text_out)

        return rewards
