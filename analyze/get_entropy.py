import json
import torch
from transformers import set_seed
from tqdm import tqdm
from scipy.stats import entropy
import numpy as np
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM

def calculate_entropy(attention_weights):
    average_attention = torch.mean(attention_weights, dim=1).squeeze(0)
    token_importance = average_attention.sum(dim=0)
    token_importance_list = np.array(token_importance.tolist())
    token_importance_list = token_importance_list / np.sum(token_importance_list)
    attention_entropy = entropy(token_importance_list)
    return attention_entropy

def main():
    set_seed(42)
    
    model_name_or_path = "Meta-Llama-3.1-8B"
    data_path = "sampled_data.json"
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map='auto', attn_implementation="eager", use_safetensors=True)
    model = model.to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    with open(data_path, "r") as json_file:
        data = json.load(json_file)

    entropy_lists = []
    for _, text in tqdm(enumerate(data), total=len(data)):

        input_ids = tokenizer.encode(text)
        input_ids = input_ids[:2048]
        input_ids = torch.tensor([input_ids]).to(model.device)
        CausalLMOutput = model(input_ids=input_ids, return_dict=False, output_hidden_states=False, output_attentions=True, use_cache=False)
        assert len(CausalLMOutput) == 2

        entropy_list = []
        for layer_attn in CausalLMOutput[1]:
            assert layer_attn.size(0) == 1
            entropy = float(calculate_entropy(layer_attn))
            entropy_list.append(entropy)
        entropy_lists.append(entropy_list)

    average_list = [sum(elements) / len(elements) for elements in zip(*entropy_lists)]
    print(average_list)

if __name__ == "__main__":
    main()