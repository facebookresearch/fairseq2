from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


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
        # Replace with your model path
        model_path = "TIGER-Lab/general-verifier"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).cuda()

    def __call__(self, prompt_chunk):
        cleaned_prompt_chunk = [
            x.replace("[", "\[").replace("]", "\]") for x in prompt_chunk
        ]

        # Tokenize and generate
        inputs = self.tokenizer(
            cleaned_prompt_chunk, return_tensors="pt", padding=True
        ).to(self.model.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=1024, temperature=0.0, do_sample=False
        )

        # Decode and print output
        # text_out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        text_out = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        rewards = convert_decisions_to_binary(text_out)

        return rewards
