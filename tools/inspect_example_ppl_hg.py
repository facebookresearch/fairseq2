import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import ast
import tqdm
import statistics
import json

# Example: load model
model_path = "/fsx-ram/lidli/checkpoints/llama3_1_1b_midtraining_data3_lr1e5_warm2000_cosdecay_8nodes_v6/checkpoints/step_32000/hg"
# model_path = "/datasets/pretrained-llms/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()


def compute_suffix_ppl(example_full_token_list, prefix_len_list):
    ppl_list = []
    for i, (tokens, prefix_len) in enumerate(
        zip(example_full_token_list, prefix_len_list)
    ):
        print(((f"rollout {i}" if i > 0 else "no thinking") + "=" * 20))
        print(
            f"prefix: {tokenizer.decode(tokens[:prefix_len], skip_special_tokens=True)}"
        )
        print("-" * 20)
        print(
            f"completion: {tokenizer.decode(tokens[prefix_len:], skip_special_tokens=True)}"
        )

        input_ids = torch.tensor([tokens])  # shape (1, seq_len)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            logits = outputs.logits  # (1, seq_len, vocab)

        # Shift for causal LM loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Suffix starts after prefix_len
        suffix_start = prefix_len - 1  # -1 because labels are shifted

        suffix_logits = shift_logits[:, suffix_start:, :]
        suffix_labels = shift_labels[:, suffix_start:]

        # Compute cross-entropy loss on suffix
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            suffix_logits.view(-1, suffix_logits.size(-1)), suffix_labels.view(-1)
        )

        mean_loss = loss.mean()
        perplexity = torch.exp(mean_loss)
        print("-" * 20)
        print(f"Suffix perplexity: {perplexity.item():.4f}")
        ppl_list.append(perplexity.item())
        print(ppl_list)
    return ppl_list


vllm_inputs = []
all_input_tok_lens = []
compute_suffix_ppl(vllm_inputs, all_input_tok_lens)
